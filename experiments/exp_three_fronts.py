"""三條線並行診斷（主人 2026-08-23 指定三件一起跑）

A. head — 為什麼餵【真】e_target 也只比沒有 u 好 15%？
   只留 anchor loss（拿掉 refine 與 null，排除互相干擾），掃 head 容量。
   對照：同容量但【只給 cond、不給 u】。
     差距隨容量拉開 => head 太小，u 的資訊沒被榨出來
     差距不變       => e_target 能給的就這麼多，問題不在 head
   ⚠️ 另外比「只練 anchor」vs「三種輸入一起練」——現在 head 同時要吃
      真 et／refine 後的 u／全零，三者差很遠，可能互相打架。

B. 隨機輪數 — 讓 test-time scaling 真的能 scale。
   `[實測]` 固定練 3 輪的結果：最佳點【剛好落在第 3 輪】，第 4 輪起一路退
   （資訊 0.0437->0.1134、cos 0.826->0.686）。那不是巧合 ——
   refine 學到的是「這三步怎麼走」，不是「一步通用的改進」。
   => 訓練時每個 batch 隨機抽輪數，逼它不能假設自己是第幾步。

C. 寬距離 — eval 是 OOD。
   訓練 geometric(p=0.02) 平均 ~50 步；eval 要 140+ 步；官方 uniform trajgoal
   平均 ~500 步。我們用了三者中【最窄】的。改成官方式均勻取樣。

用 MODE 環境變數選要跑哪一格。
"""
import os, sys, copy, json, numpy as np, torch
from torch import nn
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lacot.model import LaCoTActorState
from lacot.heads import ContinuousActionHead, MixerActionHead, TokenActionHead
from lacot.e_target import PerceiverPooler

OGB_DATA = os.environ.get("OGBENCH_DATA_DIR", "/archive/cymaxwelllee/data/ogbench")
device = "cuda" if torch.cuda.is_available() else "cpu"
ENV = "pointmaze-medium-navigate-v0"
K, D_MODEL, T_CAP, COND, CHUNK, ADIM, B = 4, 256, 16, 256, 4, 2, 64
GEOM_P, TEMP, WANDER_MAX = 0.02, 0.1, 3.0
DIM = K * D_MODEL
STEPS1 = int(os.environ.get("LACOT_STEPS1", 1200))
STEPS2 = int(os.environ.get("LACOT_STEPS2", 3000))
PROBE_STEPS = int(os.environ.get("LACOT_PROBE_STEPS", 1200))
SEED = int(os.environ.get("LACOT_SEED", 0))
EMA_M = 0.996
MODE = os.environ.get("MODE", "")

d = np.load(f"{OGB_DATA}/{ENV}.npz")
OBS = np.asarray(d["observations"], np.float32)
ACT = np.asarray(d["actions"], np.float32)
TERM = np.asarray(d["terminals"], bool)
N = OBS.shape[0]
ends = np.flatnonzero(TERM); starts = np.concatenate([[0], ends[:-1] + 1])
traj_end = np.empty(N, np.int64)
for s0, e0 in zip(starts, ends):
    traj_end[s0:e0 + 1] = e0
mu, sd = OBS.mean(0), OBS.std(0) + 1e-6
flat = lambda x: x.reshape(x.shape[0], -1)


def make_batch(rng, b=B, goal_mode="geom"):
    """goal_mode: geom = 現在的 geometric(0.02)；uniform = 官方式軌跡內均勻取。"""
    rows, goals = [], []
    while len(rows) < b:
        r = int(rng.integers(0, N)); te = int(traj_end[r])
        if te - r < 8:
            continue
        if goal_mode == "uniform":
            gr = int(round(r + rng.random() * (te - r)))    # 官方 GCDataset 的做法
        else:
            gr = min(r + int(rng.geometric(GEOM_P)), te)
        if gr - r < 8:
            continue
        path = OBS[r:gr + 1]
        if np.linalg.norm(np.diff(path, axis=0), axis=1).sum() / (np.linalg.norm(path[-1] - path[0]) + 1e-6) > WANDER_MAX:
            continue
        rows.append(r); goals.append(gr)
    rows, goals = np.array(rows), np.array(goals)
    idxs = [np.unique(np.linspace(rows[i], goals[i], min(T_CAP, goals[i] - rows[i] + 1)).round().astype(int)) for i in range(b)]
    Tmax = max(len(ix) for ix in idxs)
    traj = np.zeros((b, Tmax, 2), np.float32); mask = np.ones((b, Tmax), bool)
    mids = np.zeros((b, 2), np.float32)
    for i, ix in enumerate(idxs):
        traj[i, :len(ix)] = (OBS[ix] - mu) / sd; mask[i, :len(ix)] = False
        mids[i] = (OBS[ix[len(ix) // 2]] - mu) / sd
    s = (OBS[rows] - mu) / sd; g = (OBS[goals] - mu) / sd
    act = np.stack([ACT[r:r + CHUNK] for r in rows]).astype(np.float32)
    T_ = lambda x: torch.from_numpy(x.astype(np.float32)).to(device)
    return (T_(traj), torch.from_numpy(mask).to(device), T_(s), T_(g), T_(act), T_(mids),
            (goals - rows))


def mlp(i, h, o, n=2):
    L, p = [], i
    for _ in range(n):
        lin = nn.Linear(p, h); nn.init.xavier_uniform_(lin.weight); nn.init.zeros_(lin.bias)
        L += [lin, nn.GELU(), nn.LayerNorm(h)]; p = h
    lin = nn.Linear(p, o); nn.init.xavier_uniform_(lin.weight); nn.init.zeros_(lin.bias)
    return nn.Sequential(*L, lin)


def pair_cos(x):
    z = F.normalize(flat(x), dim=1); m = z @ z.t(); n = z.shape[0]
    return ((m.sum() - m.diag().sum()) / (n * (n - 1))).item()


def train_frontend(rng, goal_mode="geom"):
    """stage 1：contrastive e_target，各模式共用。"""
    model = LaCoTActorState(state_dim=2, d_model=D_MODEL, k=K, action_dim=ADIM,
                            chunk_len=CHUNK, cond_dim=COND).to(device)
    sg_c = mlp(2, 512, 512).to(device)
    q_pooler = PerceiverPooler(512, D_MODEL, K, 2, 4).to(device)
    opt = torch.optim.Adam(list(model.traj_enc.parameters()) + list(model.e_pooler.parameters())
                           + list(sg_c.parameters()) + list(q_pooler.parameters()), lr=1e-3)
    lab = torch.arange(B, device=device)
    for _ in range(STEPS1):
        traj, mask, s, g, _, _, _ = make_batch(rng, goal_mode=goal_mode)
        et = model.e_target(traj, mask)
        q = q_pooler(torch.stack([sg_c(s), sg_c(g)], 1))
        lg = (F.normalize(q.reshape(B, -1), dim=1) @ F.normalize(et.reshape(B, -1), dim=1).t()) / TEMP
        loss = 0.5 * (F.cross_entropy(lg, lab) + F.cross_entropy(lg.t(), lab))
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    model.freeze_front_end()
    return model, (lg.argmax(1) == lab).float().mean().item()


# ═══════════ A. head 容量 ═══════════
def run_head_token(tag, n_layers, n_heads, **hkw):
    """head 保留 u 的 token 結構（⛔ 不 flatten），對照同樣只給 cond 的版本。

    `hkw` 透傳給 TokenActionHead 的開關（deep_readout / wide / u_proj / readout_mode）。
    ⚠️ 第一版（全部關掉）輸給 concat 的 +12.2%，但那時候三個瑕疵是綁在一起的，
    分不出是「token 結構本身不好」還是「這三個瑕疵」。所以做成 2^3 全因子：
    八格＝三個開關的所有組合，同時看得到「單獨加一個有多少」跟「拿掉一個掉多少」。
    """
    torch.manual_seed(SEED); rng = np.random.default_rng(SEED)
    model, macc = train_frontend(rng)
    head_u = TokenActionHead(COND, D_MODEL, K, ADIM, CHUNK, n_layers, n_heads, **hkw).to(device)
    # 對照：同一個 head，但 u 全部餵零 —— 等於只有 cond 那個 token
    mods = [model.cond_enc, model.cond_head, head_u]
    opt = torch.optim.Adam([p for m in mods for p in m.parameters()], lr=5e-4)
    for _ in range(STEPS2):
        traj, mask, s, g, act, _, _ = make_batch(rng)
        with torch.no_grad():
            et = model.e_target(traj, mask)
        cond = model.encode_cond(s, g)
        l = (head_u(cond, et) - act).pow(2).mean()
        l = l + (head_u(cond, torch.zeros_like(et)) - act).pow(2).mean()   # cond-only 分支
        opt.zero_grad(set_to_none=True); l.backward()
        torch.nn.utils.clip_grad_norm_([p for m in mods for p in m.parameters()], 1.0)
        opt.step()
    for m in mods:
        m.eval()
    er = np.random.default_rng(4242)
    E_traj, E_mask, E_s, E_g, E_act, _, _ = make_batch(er, b=512)
    with torch.no_grad():
        et = model.e_target(E_traj, E_mask); c = model.encode_cond(E_s, E_g)
        a_u = (head_u(c, et) - E_act).pow(2).mean().item()
        a_c = (head_u(c, torch.zeros_like(et)) - E_act).pow(2).mean().item()
        base = ((E_act - E_act.mean(0, keepdim=True)) ** 2).mean().item()
    params = sum(p.numel() for p in head_u.parameters())
    print(f"  {tag:<22} 真et {a_u:.4f} | 只cond {a_c:.4f} | 增益 {100*(1-a_u/a_c):+.1f}% "
          f"| 猜平均 {base:.4f} | head參數 {params/1e6:.1f}M", flush=True)
    return dict(tag=tag, mode="head", hidden=0, layers=n_layers, isolate=True,
                anchor=a_u, cond_only=a_c, gain=1 - a_u / a_c, base=base,
                params=params, macc=macc, switches=hkw)


def run_cond_size(tag, cond_dim, enc_hidden):
    """cond 縮小 —— 看 head 會不會被迫去用 u。

    ⚠️ 為什麼要測：cond 的輸入只有 s,g 兩個 (x,y)＝【4 個數字】，卻用約 1.4M 參數
    擴張成 256 維；而 u 帶著整條軌跡的資訊，是【凍結的 e_target 直接餵】、沒有任何加工。
    ⇒ 兩個輸入的起跑點差太多：cond 有人鋪好路，u 是生的。head 偏好 cond 是理性的。
    `[實測]` 這個 256 是昨晚寫 state 版時從影像版沿用的（commit 6d707d9）——
    影像版那裡是「兩張 64x64 過 CNN 的特徵」，256 有道理；state 版沒有人重新想過。
    而設計文件明寫 "matched-capacity to the SOTA baselines, not inherited from
    any prior in-house model"。
    """
    torch.manual_seed(SEED); rng = np.random.default_rng(SEED)
    model, macc = train_frontend(rng)
    cond_enc = mlp(2, enc_hidden, enc_hidden).to(device)
    cond_head = mlp(2 * enc_hidden, enc_hidden, cond_dim).to(device)
    head_u = TokenActionHead(cond_dim, D_MODEL, K, ADIM, CHUNK, 2, 4).to(device)
    mods = [cond_enc, cond_head, head_u]
    opt = torch.optim.Adam([p for m in mods for p in m.parameters()], lr=5e-4)
    enc = lambda s, g: cond_head(torch.cat([cond_enc(s), cond_enc(g)], -1))
    for _ in range(STEPS2):
        traj, mask, s, g, act, _, _ = make_batch(rng)
        with torch.no_grad():
            et = model.e_target(traj, mask)
        c = enc(s, g)
        l = (head_u(c, et) - act).pow(2).mean() + (head_u(c, torch.zeros_like(et)) - act).pow(2).mean()
        opt.zero_grad(set_to_none=True); l.backward()
        torch.nn.utils.clip_grad_norm_([p for m in mods for p in m.parameters()], 1.0)
        opt.step()
    for m in mods:
        m.eval()
    er = np.random.default_rng(4242)
    E_traj, E_mask, E_s, E_g, E_act, _, _ = make_batch(er, b=512)
    with torch.no_grad():
        et = model.e_target(E_traj, E_mask); c = enc(E_s, E_g)
        a_u = (head_u(c, et) - E_act).pow(2).mean().item()
        a_c = (head_u(c, torch.zeros_like(et)) - E_act).pow(2).mean().item()
        base = ((E_act - E_act.mean(0, keepdim=True)) ** 2).mean().item()
    params = sum(p.numel() for m in (cond_enc, cond_head) for p in m.parameters())
    print(f"  {tag:<22} 真et {a_u:.4f} | 只cond {a_c:.4f} | 增益 {100*(1-a_u/a_c):+.1f}% "
          f"| 猜平均 {base:.4f} | cond網路 {params/1e6:.2f}M", flush=True)
    return dict(tag=tag, mode="head", hidden=cond_dim, layers=2, isolate=True,
                anchor=a_u, cond_only=a_c, gain=1 - a_u / a_c, base=base,
                params=params, macc=macc)


def run_head(tag, hidden, layers, isolate):
    """isolate=True：只練 anchor（排除 refine/null 干擾）；False：三種輸入一起練。"""
    torch.manual_seed(SEED); rng = np.random.default_rng(SEED)
    model, macc = train_frontend(rng)
    head_u = ContinuousActionHead(COND + DIM, ADIM, CHUNK, hidden=hidden, n_layers=layers).to(device)
    head_c = ContinuousActionHead(COND, ADIM, CHUNK, hidden=hidden, n_layers=layers).to(device)  # ⛔ 只給 cond
    mods = [model.cond_enc, model.cond_head, model.flow, model.refine, head_u, head_c]
    opt = torch.optim.Adam([p for m in mods for p in m.parameters()], lr=5e-4)
    for _ in range(STEPS2):
        traj, mask, s, g, act, _, _ = make_batch(rng)
        with torch.no_grad():
            et = model.e_target(traj, mask)
        cond = model.encode_cond(s, g)
        l_anchor = (head_u(torch.cat([cond, flat(et)], -1)) - act).pow(2).mean()
        l_condonly = (head_c(cond) - act).pow(2).mean()
        loss = l_anchor + l_condonly
        if not isolate:                      # 三種輸入一起練（現況）
            u = model.flow.sample(B, cond).detach()
            for _ in range(3):
                u = model.refine(cond, u)
            loss = loss + (head_u(torch.cat([cond, flat(u)], -1)) - act).pow(2).mean()
            Z = torch.zeros_like(flat(et))
            loss = loss + (head_u(torch.cat([cond, Z], -1)) - act).pow(2).mean()
            loss = loss + model.flow.nll(et, cond) / DIM
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_([p for m in mods for p in m.parameters()], 1.0)
        opt.step()
    for m in mods:
        m.eval()
    er = np.random.default_rng(4242)
    E_traj, E_mask, E_s, E_g, E_act, _, _ = make_batch(er, b=512)
    with torch.no_grad():
        et = model.e_target(E_traj, E_mask); c = model.encode_cond(E_s, E_g)
        a_u = (head_u(torch.cat([c, flat(et)], -1)) - E_act).pow(2).mean().item()
        a_c = (head_c(c) - E_act).pow(2).mean().item()
        base = ((E_act - E_act.mean(0, keepdim=True)) ** 2).mean().item()
    params = sum(p.numel() for p in head_u.parameters())
    print(f"  {tag:<22} 真et {a_u:.4f} | 只cond {a_c:.4f} | 增益 {100*(1-a_u/a_c):+.1f}% "
          f"| 猜平均 {base:.4f} | head參數 {params/1e6:.1f}M", flush=True)
    return dict(tag=tag, mode="head", hidden=hidden, layers=layers, isolate=isolate,
                anchor=a_u, cond_only=a_c, gain=1 - a_u / a_c, base=base,
                params=params, macc=macc)


# ═══════════ A2. 協定對齊（2x2）═══════════
def run_proto(tag, arch, protocol, cond_dim=None, enc_hidden=None, **hkw):
    """把「增益」這個量法本身拆出來當變數。

    🚨 2026-08-23 發現：concat 版跟 token 版一直不是同一種比法，而這個差別
       比 head 的容量／寬度／投影層那三項都大。

       ded（專用）  head_u 只看過「u 在」的情況，另外有一顆專用的 head_c 只吃 cond。
                    ＝ A 同學永遠有課本、B 同學永遠沒有，比兩個人的分數。
       shared（共用）同一顆 head 要同時服務「u 在」跟「u 是 0」兩種輸入。
                    ＝ C 同學一個人考兩次，而且兩次都要考好。

    ⚠️ 為什麼會差很多：shared 底下，一顆網路要在兩種輸入下都交代得過去，
       最省事的解就是【乾脆不要用 u】—— 兩邊輸出一樣就好。訓練本身在壓
       「別依賴 u」，而增益量的正好是「有多依賴 u」。⇒ 這不是架構的差別，
       是規則的差別。原本 concat 跑 ded、token 跑 shared，所以 concat 看起來贏。

    arch: "concat" | "token"　protocol: "ded" | "shared"
    """
    torch.manual_seed(SEED); rng = np.random.default_rng(SEED)
    model, macc = train_frontend(rng)

    # cond_dim 有給就自己建一個 cond 編碼器（跟 run_cond_size 同一套），
    # 沒給就用 model 內建的（COND=256）。⚠️ 兩者實作不同 ⇒ ⛔ 不要跨組比絕對值。
    if cond_dim is None:
        CD, enc, extra = COND, model.encode_cond, [model.cond_enc, model.cond_head]
    else:
        cond_enc = mlp(2, enc_hidden, enc_hidden).to(device)
        cond_head = mlp(2 * enc_hidden, enc_hidden, cond_dim).to(device)
        CD, extra = cond_dim, [cond_enc, cond_head]
        enc = lambda s, g: cond_head(torch.cat([cond_enc(s), cond_enc(g)], -1))

    if arch == "concat":
        mk_u = lambda: ContinuousActionHead(CD + DIM, ADIM, CHUNK, hidden=512, n_layers=3).to(device)
        fwd = lambda h, c, u: h(torch.cat([c, flat(u)], -1))
    elif arch == "mixer":
        mk_u = lambda: MixerActionHead(CD, D_MODEL, K, ADIM, CHUNK, 2, **hkw).to(device)
        fwd = lambda h, c, u: h(c, u)
    else:
        mk_u = lambda: TokenActionHead(CD, D_MODEL, K, ADIM, CHUNK, 2, 4, **hkw).to(device)
        fwd = lambda h, c, u: h(c, u)

    head_u = mk_u()
    # ded 要第二顆【獨立】的網路吃 cond；shared 就是同一顆
    head_c = mk_u() if protocol == "ded" else head_u
    mods = extra + [head_u] + ([head_c] if protocol == "ded" else [])
    opt = torch.optim.Adam([p for m in mods for p in m.parameters()], lr=5e-4)

    for _ in range(STEPS2):
        traj, mask, s, g, act, _, _ = make_batch(rng)
        with torch.no_grad():
            et = model.e_target(traj, mask)
        cond = enc(s, g)
        Z = torch.zeros_like(et)
        # ⛔ 兩種協定的【總損失項數一樣】（都兩項），差別只在第二項餵給誰。
        l = (fwd(head_u, cond, et) - act).pow(2).mean() \
          + (fwd(head_c, cond, Z) - act).pow(2).mean()
        opt.zero_grad(set_to_none=True); l.backward()
        torch.nn.utils.clip_grad_norm_([p for m in mods for p in m.parameters()], 1.0)
        opt.step()

    for m in mods:
        m.eval()
    er = np.random.default_rng(4242)
    E_traj, E_mask, E_s, E_g, E_act, _, _ = make_batch(er, b=512)
    with torch.no_grad():
        et = model.e_target(E_traj, E_mask); c = enc(E_s, E_g)
        a_u = (fwd(head_u, c, et) - E_act).pow(2).mean().item()
        a_c = (fwd(head_c, c, torch.zeros_like(et)) - E_act).pow(2).mean().item()
        base = ((E_act - E_act.mean(0, keepdim=True)) ** 2).mean().item()
    params = sum(p.numel() for p in head_u.parameters())
    print(f"  {tag:<24} 真et {a_u:.4f} | 只cond {a_c:.4f} | 增益 {100*(1-a_u/a_c):+.1f}% "
          f"| 猜平均 {base:.4f} | head參數 {params/1e6:.1f}M", flush=True)
    return dict(tag=tag, mode="proto", arch=arch, protocol=protocol, switches=hkw,
                cond_dim=CD, cond_params=sum(p.numel() for m in extra for p in m.parameters()),
                anchor=a_u, cond_only=a_c, gain=1 - a_u / a_c, base=base,
                params=params, macc=macc)


# ═══════════ B. 隨機輪數 / C. 寬距離 ═══════════
def run_train(tag, rand_rounds, goal_mode):
    torch.manual_seed(SEED); rng = np.random.default_rng(SEED)
    model, macc = train_frontend(rng, goal_mode=goal_mode)
    refine_ema = copy.deepcopy(model.refine)
    for p in refine_ema.parameters():
        p.requires_grad_(False)
    mods = [model.cond_enc, model.cond_head, model.flow, model.refine, model.action_head]
    opt = torch.optim.Adam([p for m in mods for p in m.parameters()], lr=5e-4)
    ZERO = torch.zeros(B, K, D_MODEL, device=device)
    for _ in range(STEPS2):
        traj, mask, s, g, act, _, _ = make_batch(rng, goal_mode=goal_mode)
        with torch.no_grad():
            et = model.e_target(traj, mask)
        cond = model.encode_cond(s, g)
        cat = lambda uu: torch.cat([cond, flat(uu)], -1)
        R = int(rng.integers(1, 9)) if rand_rounds else 3     # ← B 的唯一差別
        l = model.flow.nll(et, cond) / DIM
        l = l + model.action_head.nll(model.action_head(cat(et)), act).mean()
        u = model.flow.sample(B, cond).detach(); us = [u]
        for _ in range(R):
            u = model.refine(cond, u); us.append(u)
        l = l + sum(model.action_head.nll(model.action_head(cat(us[r + 1])), act).mean()
                    for r in range(R)) / R
        lc = us[0].new_zeros(())
        for r in range(R):
            with torch.no_grad():
                tgt = refine_ema(cond, us[r])
            lc = lc + (us[r + 1] - tgt).pow(2).mean()
        l = l + 0.5 * lc / R
        l = l + model.action_head.nll(model.action_head(torch.cat([cond, flat(ZERO)], -1)), act).mean()
        l.backward()
        torch.nn.utils.clip_grad_norm_([p for m in mods for p in m.parameters()], 1.0)
        opt.step(); opt.zero_grad(set_to_none=True)
        with torch.no_grad():
            for pe, p in zip(refine_ema.parameters(), model.refine.parameters()):
                pe.mul_(EMA_M).add_(p, alpha=1 - EMA_M)
    model.eval()

    pr_u = mlp(DIM, 512, 2, n=3).to(device); pr_i = mlp(4, 512, 2, n=3).to(device)
    op = torch.optim.Adam(list(pr_u.parameters()) + list(pr_i.parameters()), lr=1e-3)
    for _ in range(PROBE_STEPS):
        traj, mask, s, g, _, mid, _ = make_batch(rng, goal_mode=goal_mode)
        with torch.no_grad():
            c = model.encode_cond(s, g); uu = model.sample_u(c)
            for _ in range(3):
                uu = model.refine(c, uu)
        ll = (pr_u(flat(uu)) - mid).pow(2).mean() + (pr_i(torch.cat([s, g], 1)) - mid).pow(2).mean()
        op.zero_grad(set_to_none=True); ll.backward(); op.step()
    pr_u.eval(); pr_i.eval()

    er = np.random.default_rng(4242)
    E_traj, E_mask, E_s, E_g, _, E_mid, dist = make_batch(er, b=512, goal_mode=goal_mode)
    rows = []
    with torch.no_grad():
        E_et = model.e_target(E_traj, E_mask); E_c = model.encode_cond(E_s, E_g)
        interp = (pr_i(torch.cat([E_s, E_g], 1)) - E_mid).pow(2).mean().item()
        u = model.sample_u(E_c)
        for r in range(13):
            if r > 0:
                u = model.refine(E_c, u)
            rows.append(dict(round=r, collapse=pair_cos(u),
                             info=(pr_u(flat(u)) - E_mid).pow(2).mean().item(),
                             cos_et=F.cosine_similarity(flat(u), flat(E_et), dim=1).mean().item()))
    best = min(rows, key=lambda r: r["info"])
    print(f"  {tag:<22} 最佳第 {best['round']} 輪 info {best['info']:.4f}（內插 {interp:.4f}）"
          f" cos {best['cos_et']:+.3f} | 第12輪 info {rows[12]['info']:.4f} cos {rows[12]['cos_et']:+.3f}"
          f" | goal距離中位數 {int(np.median(dist))}", flush=True)
    return dict(tag=tag, mode="train", rand_rounds=rand_rounds, goal_mode=goal_mode,
                interp=interp, rows=rows, best_round=best["round"],
                goal_dist_median=float(np.median(dist)), macc=macc)


CASES = {
    # A：head 容量（只練 anchor，排除干擾）
    "head_s":     lambda: run_head("head 3層×512 (現況)",  512,  3, True),
    "head_m":     lambda: run_head("head 4層×1024",        1024, 4, True),
    "head_l":     lambda: run_head("head 5層×2048",        2048, 5, True),
    "head_mix":   lambda: run_head("head 3層×512 三輸入混練", 512, 3, False),
    # 3：token-based head（不 flatten u）—— 對照上面那三格的 concat 版
    "head_tok1":  lambda: run_head_token("token head 1層", 1, 4),
    "head_tok2":  lambda: run_head_token("token head 2層", 2, 4),
    "head_tok4":  lambda: run_head_token("token head 4層", 4, 8),
    # ── token head 的 2^3 全因子 ablation（主人 2026-08-23：「我想看 ablation」）──
    #    r = deep_readout（3 層讀出）　w = wide（內部 512）　u = u_proj（u 的投影層）
    #    ⛔ 全部固定 2 層 / 4 頭，只有這三個開關在動，跟 head_tok2 是同一格起點。
    "tok_000":    lambda: run_head_token("tok 基準(三個都關)", 2, 4),
    "tok_r00":    lambda: run_head_token("tok +readout",       2, 4, deep_readout=True),
    "tok_0w0":    lambda: run_head_token("tok +wide",          2, 4, wide=512),
    "tok_00u":    lambda: run_head_token("tok +uproj",         2, 4, u_proj=True),
    "tok_rw0":    lambda: run_head_token("tok +readout+wide",  2, 4, deep_readout=True, wide=512),
    "tok_r0u":    lambda: run_head_token("tok +readout+uproj", 2, 4, deep_readout=True, u_proj=True),
    "tok_0wu":    lambda: run_head_token("tok +wide+uproj",    2, 4, wide=512, u_proj=True),
    "tok_rwu":    lambda: run_head_token("tok 三個全開",        2, 4, deep_readout=True, wide=512, u_proj=True),
    # ── 協定對齊 2x2（+2 格）：架構 × 量法，主人 2026-08-23 核可 ──
    "pr_cat_ded":   lambda: run_proto("concat / 專用",    "concat", "ded"),
    "pr_cat_shr":   lambda: run_proto("concat / 共用",    "concat", "shared"),
    "pr_tok_ded":   lambda: run_proto("token基準 / 專用", "token",  "ded"),
    "pr_tok_shr":   lambda: run_proto("token基準 / 共用", "token",  "shared"),
    "pr_tokru_ded": lambda: run_proto("token r0u / 專用", "token",  "ded",
                                      deep_readout=True, u_proj=True),
    "pr_tokru_shr": lambda: run_proto("token r0u / 共用", "token",  "shared",
                                      deep_readout=True, u_proj=True),
    # ── B1：動作從哪顆 token 讀出來（協定固定 ded，跟 concat 那格同規則）──
    #    ⚠️ 2x2 已證實協定不影響數字，所以這裡只留一種協定，把變數壓到只剩讀出位置。
    "b1_cond":    lambda: run_proto("B1 讀 cond 顆（現況）", "token", "ded", readout_mode="cond"),
    "b1_pool":    lambda: run_proto("B1 五顆平均",           "token", "ded", readout_mode="pool"),
    "b1_query":   lambda: run_proto("B1 專用 query token",   "token", "ded", readout_mode="query"),
    # ── MLP-Mixer head（主人 2026-08-23 同意加進待跑清單）──
    #    跨 token 的混合改成普通線性層（自由加減），⛔ 不是 attention 的加權平均。
    "mx_pool":    lambda: run_proto("mixer 五顆平均",  "mixer", "ded", readout_mode="pool"),
    "mx_cond":    lambda: run_proto("mixer 讀 cond 顆", "mixer", "ded", readout_mode="cond"),
    # ── concat × cond 大小（主人 2026-08-23：「留 concat，來試試看 concat + cond64」）──
    #    ⚠️ 全部用【自建】的 cond 編碼器，跟 token 那組的 cond sweep 同一套實作，
    #       這樣 concat 跟 token 才比得起來。⛔ 不要拿去跟 pr_cat_ded 比絕對值
    #       —— 那格用的是 model 內建的編碼器，不同實作。
    "cc256":      lambda: run_proto("concat cond 256", "concat", "ded", cond_dim=256, enc_hidden=512),
    "cc64":       lambda: run_proto("concat cond 64",  "concat", "ded", cond_dim=64,  enc_hidden=128),
    "cc16":       lambda: run_proto("concat cond 16",  "concat", "ded", cond_dim=16,  enc_hidden=64),
    "cc4":        lambda: run_proto("concat cond 4",   "concat", "ded", cond_dim=4,   enc_hidden=32),
    # ── token × cond 大小，跟 cc* 同規格重跑（3 seed、ded 協定）──
    #    ⚠️ 舊的 cond4/16/64/256 四格只有 1 seed、且用 shared 協定 ⇒ ⛔ 不能跟 cc* 對比。
    #    ★ 追這一格的理由：舊資料裡 token cond64 的真et 0.2375 比 concat 最好的 0.2421
    #      還低。若不是運氣，「留 concat」的結論就要反過來。
    "tc256":      lambda: run_proto("token cond 256", "token", "ded", cond_dim=256, enc_hidden=512),
    "tc64":       lambda: run_proto("token cond 64",  "token", "ded", cond_dim=64,  enc_hidden=128),
    "tc16":       lambda: run_proto("token cond 16",  "token", "ded", cond_dim=16,  enc_hidden=64),
    "tc4":        lambda: run_proto("token cond 4",   "token", "ded", cond_dim=4,   enc_hidden=32),
    # cond 縮小（都配 token head 2 層，只有 cond 大小不同）
    "cond256":    lambda: run_cond_size("cond 256（現況）", 256, 512),
    "cond64":     lambda: run_cond_size("cond 64",          64, 128),
    "cond16":     lambda: run_cond_size("cond 16",          16,  64),
    "cond4":      lambda: run_cond_size("cond 4（幾乎原始）",  4,  32),
    # B：隨機輪數
    "rounds_fix": lambda: run_train("固定 3 輪（現況）", False, "geom"),
    "rounds_rnd": lambda: run_train("隨機 1~8 輪",       True,  "geom"),
    # C：寬距離
    "goal_geom":  lambda: run_train("geometric goal（現況）", False, "geom"),
    "goal_unif":  lambda: run_train("uniform goal（官方式）", False, "uniform"),
    "both":       lambda: run_train("隨機輪數 ＋ uniform goal", True, "uniform"),
}

if MODE not in CASES:
    print(f"用法：MODE=<name> ...  可選：{', '.join(CASES)}")
    sys.exit(2)
print(f"=== {MODE} | seed {SEED} ===", flush=True)
row = CASES[MODE]()
out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "results", f"fronts_{MODE}_s{SEED}.json")
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w") as f:
    json.dump(row, f, indent=1)
print(f"寫入 {out}", flush=True)
