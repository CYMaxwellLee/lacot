"""哪一種 consistency loss 擋得住 refine 塌掉？（主人 2026-08-23：EMA / BYOL / JEPA 那系列）

現況（exp_refine_process 實測）：refine 一輪就把所有 u 壓成 batch 平均 ——
batch 內兩兩 cosine 0.9999（真 e_target 是 0.025），離 batch 平均 31.1 -> 0.2，
路徑資訊從 0.041 掉到 0.873（＝猜平均的水準）。

為什麼現在這個擋不住：`l_cons = (u^r - u^{r+1}.detach())²` 比較的兩邊都是
同一個網路、同一個當下的輸出 —— 它們可以【一起】往同一點塌，差距是 0、loss 完美。
而且塌成常數之後只有第一輪要付代價，後兩輪全免費。

四種比：
  A. self（現在這個）  權重掃 0 / 0.5 / 2 / 8
  B. ema     BYOL 那套 —— 跟 refine 的 EMA 副本比，EMA 更新慢、扯得住
  C. center  DINO 那套 —— 比之前先減掉 batch 平均，塌向平均就失去獎勵
  D. var     VICReg/JEPA 那套 —— 直接懲罰 batch 內標準差太小

⛔ 兩個都要過才算好：
  ① 不塌      —— 兩兩 cosine 要離 0.9999 遠、往 0.025 靠
  ② 資訊還在  —— probe 從 u 還原路徑中點，要贏過「只用 (s,g) 內插」
     （只有 ① 沒有 ② 可能只是變成隨機噪聲，一樣沒用）

先用這兩個指標快篩，不跑 rollout；活下來的才值得花時間進環境。
"""
import os, sys, copy, numpy as np, torch
from torch import nn
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lacot.model import LaCoTActorState
from lacot.e_target import PerceiverPooler

OGB_DATA = os.environ.get("OGBENCH_DATA_DIR", "/archive/cymaxwelllee/data/ogbench")
device = "cuda" if torch.cuda.is_available() else "cpu"
ENV = "pointmaze-medium-navigate-v0"
K, D_MODEL, T_CAP, COND, CHUNK, ADIM, B = 4, 256, 16, 256, 4, 2, 64
GEOM_P, TEMP, WANDER_MAX = 0.02, 0.1, 3.0
STEPS1 = int(os.environ.get("LACOT_STEPS1", 1200))
STEPS2 = int(os.environ.get("LACOT_STEPS2", 3000))
PROBE_STEPS = int(os.environ.get("LACOT_PROBE_STEPS", 1200))
ROUNDS = 3
DIM = K * D_MODEL
EMA_M = float(os.environ.get("LACOT_EMA_M", 0.99))
VAR_GAMMA = 1.0

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
print(f"device {device} | K={K} dim={DIM}", flush=True)


def make_batch(rng, b=B):
    rows, goals = [], []
    while len(rows) < b:
        r = int(rng.integers(0, N)); te = int(traj_end[r])
        if te - r < 8:
            continue
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
    return T_(traj), torch.from_numpy(mask).to(device), T_(s), T_(g), T_(act), T_(mids)


def mlp(i, h, o, n=2):
    L, p = [], i
    for _ in range(n):
        lin = nn.Linear(p, h); nn.init.xavier_uniform_(lin.weight); nn.init.zeros_(lin.bias)
        L += [lin, nn.GELU(), nn.LayerNorm(h)]; p = h
    lin = nn.Linear(p, o); nn.init.xavier_uniform_(lin.weight); nn.init.zeros_(lin.bias)
    return nn.Sequential(*L, lin)


flat = lambda x: x.reshape(x.shape[0], -1)


def pair_cos(x):
    z = F.normalize(flat(x), dim=1)
    m = z @ z.t()
    n = z.shape[0]
    return ((m.sum() - m.diag().sum()) / (n * (n - 1))).item()


def eff_dim(x, thresh=0.99):
    """撐起 thresh 變異需要幾個主成分 —— 量【維度層面】的塌。

    ⚠️ 加這一項是因為 Rei 在 LeWM 上實測到「SIGReg 防了樣本塌，latent 卻塌到
    192 維裡的 63 維」。樣本彼此不像（pair_cos 低）不代表沒塌 —— 有可能只是
    在少數幾個維度上互相不同，其餘維度全是死的。
    """
    z = flat(x) - flat(x).mean(0, keepdim=True)
    sv = torch.linalg.svdvals(z.double())
    var = (sv ** 2)
    ratio = torch.cumsum(var, 0) / var.sum()
    return int((ratio < thresh).sum().item()) + 1


def barlow(a, b):
    """Barlow Twins：cross-correlation 矩陣逼近單位矩陣。

    防的是【維度之間冗餘】—— 對角線=1（同一維在兩邊要一致）、
    非對角線=0（不同維不能編同一件事）。這跟 pair_cos 防的是不同的塌。
    """
    za = (a - a.mean(0, keepdim=True)) / (a.std(0, keepdim=True) + 1e-6)
    zb = (b - b.mean(0, keepdim=True)) / (b.std(0, keepdim=True) + 1e-6)
    c = (za.t() @ zb) / a.shape[0]
    on = (torch.diagonal(c) - 1).pow(2).sum()
    off = (c.pow(2).sum() - torch.diagonal(c).pow(2).sum())
    return (on + 0.005 * off) / a.shape[1]


class SIGReg(nn.Module):
    """Sketch Isotropic Gaussian Regularizer（arXiv:2511.08544，LeWM 用的）。

    逐字對照 Rei 的 porting/test_drifting_lewm.py（該檔註明是
    stable_worldmodel/wm/loss.py::SIGReg 的 verbatim copy）。
    做法：把表徵隨機投影到 num_proj 個一維方向，檢定每個方向上的分布
    像不像標準高斯（比較特徵函數的實部 cos 與虛部 sin，用高斯窗加權積分）。
    塌掉的東西不可能是高斯 => 擋得住。
    ⚠️ 已知限制（LeWM Appendix H 自承）：只約束【每個時間點各自】的分布，
       時間維度無約束，會 temporal collapse。
    """

    def __init__(self, knots=17, num_proj=256):
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots)
        dt = 3 / (knots - 1)
        w = torch.full((knots,), 2 * dt)
        w[[0, -1]] = dt
        win = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", win)
        self.register_buffer("weights", w * win)

    def forward(self, proj):
        A = torch.randn(proj.size(-1), self.num_proj, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        return ((err @ self.weights) * proj.size(-2)).mean()


def cons_loss(kind, us, refine_ema, cond, predictor=None, sigreg=None):
    """各種 consistency 的差別只在這裡。"""
    L = us[0].new_zeros(())
    # ── 有 _ema 後綴的：目標換成 EMA 的輸出（主人 08-23：「其他方法也都可以有 EMA」）
    use_ema = kind.endswith("_ema")
    base = kind[:-4] if use_ema else kind

    def target(r):
        """要拿什麼當比較對象 —— 這一格就是 EMA 與否的唯一差別。"""
        if use_ema:
            with torch.no_grad():
                return refine_ema(cond, us[r])
        return us[r].detach()

    if base == "sigreg":
        for r in range(1, ROUNDS + 1):
            L = L + sigreg(flat(us[r]).unsqueeze(0))
        if use_ema:                      # ＋ 跟 EMA 對齊
            for r in range(ROUNDS):
                L = L + (us[r + 1] - target(r)).pow(2).mean()
        return L / ROUNDS
    if base == "barlow":
        for r in range(ROUNDS):
            L = L + barlow(flat(us[r + 1]), flat(target(r)))
        return L / ROUNDS
    if base == "center" and use_ema:
        for r in range(ROUNDS):
            a = us[r + 1] - us[r + 1].mean(0, keepdim=True)
            t = target(r); b = (t - t.mean(0, keepdim=True))
            L = L + (a - b).pow(2).mean()
        return L / ROUNDS
    if base == "var" and use_ema:
        for r in range(ROUNDS):
            L = L + (us[r + 1] - target(r)).pow(2).mean()
            L = L + F.relu(VAR_GAMMA - flat(us[r + 1]).std(dim=0)).mean()
        return L / ROUNDS
    if kind == "barlow":
        for r in range(ROUNDS):
            L = L + barlow(flat(us[r + 1]), flat(us[r]).detach())
        return L / ROUNDS
    if kind == "ema_var":               # EMA 管方向 ＋ variance 管散度
        for r in range(ROUNDS):
            with torch.no_grad():
                tgt = refine_ema(cond, us[r])
            L = L + (us[r + 1] - tgt).pow(2).mean()
            std = flat(us[r + 1]).std(dim=0)
            L = L + F.relu(VAR_GAMMA - std).mean()
        return L / ROUNDS
    if kind == "byol":                  # 真正的 BYOL：online 多一個 predictor
        for r in range(ROUNDS):
            with torch.no_grad():
                tgt = refine_ema(cond, us[r])
            pred = predictor(flat(us[r + 1])).reshape_as(tgt)
            L = L + (F.normalize(flat(pred), dim=1)
                     - F.normalize(flat(tgt), dim=1)).pow(2).sum(1).mean()
        return L / ROUNDS
    if kind == "var":
        # VICReg/JEPA：直接要求 batch 內每個維度的標準差不能太小
        for r in range(1, ROUNDS + 1):
            std = flat(us[r]).std(dim=0)
            L = L + F.relu(VAR_GAMMA - std).mean()
        return L / ROUNDS
    for r in range(ROUNDS):
        a, b = us[r], us[r + 1]
        if kind == "self":                      # 現在這個
            L = L + (a - b.detach()).pow(2).mean()
        elif kind == "center":                  # DINO：先減掉 batch 平均
            am = a - a.mean(0, keepdim=True)
            bm = (b - b.mean(0, keepdim=True)).detach()
            L = L + (am - bm).pow(2).mean()
        elif kind == "ema":                     # BYOL：跟 EMA 副本比
            with torch.no_grad():
                tgt = refine_ema(cond, us[r])
            L = L + (b - tgt).pow(2).mean()
    return L / ROUNDS


def run(kind, lam, tag):
    torch.manual_seed(0); rng = np.random.default_rng(0)
    model = LaCoTActorState(state_dim=2, d_model=D_MODEL, k=K, action_dim=ADIM,
                            chunk_len=CHUNK, cond_dim=COND).to(device)
    sg_c = mlp(2, 512, 512).to(device)
    q_pooler = PerceiverPooler(512, D_MODEL, K, 2, 4).to(device)
    opt1 = torch.optim.Adam(
        list(model.traj_enc.parameters()) + list(model.e_pooler.parameters())
        + list(sg_c.parameters()) + list(q_pooler.parameters()), lr=1e-3)
    lab = torch.arange(B, device=device)
    for stp in range(STEPS1):
        traj, mask, s, g, _, _ = make_batch(rng)
        et = model.e_target(traj, mask)
        q = q_pooler(torch.stack([sg_c(s), sg_c(g)], 1))
        lg = (F.normalize(q.reshape(B, -1), dim=1) @ F.normalize(et.reshape(B, -1), dim=1).t()) / TEMP
        loss = 0.5 * (F.cross_entropy(lg, lab) + F.cross_entropy(lg.t(), lab))
        opt1.zero_grad(set_to_none=True); loss.backward(); opt1.step()
    model.freeze_front_end()
    macc = (lg.argmax(1) == lab).float().mean().item()

    needs_ema = kind in ("ema", "ema_var", "byol") or kind.endswith("_ema")
    refine_ema = copy.deepcopy(model.refine) if needs_ema else None
    # ⚠️ predictor 要【刻意弱】——BYOL 的防塌靠「online 得保留真資訊才預測得準」，
    #    predictor 一旦強到什麼都能擬合，online 丟光資訊輸出常數也照樣被預測出來，
    #    約束就失效了。實測：hidden=1024 的 byol(0.4706) 比【沒有 predictor】的
    #    ema(0.1060) 還糟 —— 等於把煞車拆了。
    #    我們這個問題很小：u 名目 1024 維，但真 e_target 有效維只有 47、
    #    軌跡真實資訊量只有 32（16 點 × 2 維）。所以往小的搜。
    PRED_H = int(os.environ.get("LACOT_PRED_HIDDEN", 64))
    predictor = mlp(DIM, PRED_H, DIM, n=int(os.environ.get("LACOT_PRED_LAYERS", 1))).to(device) if kind == "byol" else None
    sigreg = SIGReg().to(device) if "sigreg" in kind else None
    if refine_ema is not None:
        for p in refine_ema.parameters():
            p.requires_grad_(False)

    mods = [model.cond_enc, model.cond_head, model.flow, model.refine, model.action_head]
    if predictor is not None:
        mods = mods + [predictor]
    opt2 = torch.optim.Adam([p for m in mods for p in m.parameters()], lr=5e-4)
    ZERO = torch.zeros(B, K, D_MODEL, device=device)
    for stp in range(STEPS2):
        traj, mask, s, g, act, _ = make_batch(rng)
        with torch.no_grad():
            et = model.e_target(traj, mask)
        cond = model.encode_cond(s, g)
        l_nf = model.flow.nll(et, cond) / DIM
        cat = lambda uu: torch.cat([cond, flat(uu)], -1)
        l_anchor = model.action_head.nll(model.action_head(cat(et)), act).mean()
        u = model.flow.sample(B, cond).detach()
        us = [u]
        for _ in range(ROUNDS):
            u = model.refine(cond, u); us.append(u)
        l_refine = sum(model.action_head.nll(model.action_head(cat(us[r + 1])), act).mean()
                       for r in range(ROUNDS)) / ROUNDS
        l_cons = cons_loss(kind, us, refine_ema, cond, predictor, sigreg)
        l_null = model.action_head.nll(model.action_head(torch.cat([cond, flat(ZERO)], -1)), act).mean()
        (l_nf + l_anchor + l_refine + lam * l_cons + l_null).backward()
        torch.nn.utils.clip_grad_norm_([p for m in mods for p in m.parameters()], 1.0)
        opt2.step(); opt2.zero_grad(set_to_none=True)
        if refine_ema is not None:
            with torch.no_grad():
                for pe, p in zip(refine_ema.parameters(), model.refine.parameters()):
                    pe.mul_(EMA_M).add_(p, alpha=1 - EMA_M)
    model.eval()

    # probe：refine 後的 u 還剩多少路徑資訊
    pr_u = mlp(DIM, 512, 2, n=3).to(device)
    pr_i = mlp(4, 512, 2, n=3).to(device)
    opt_p = torch.optim.Adam(list(pr_u.parameters()) + list(pr_i.parameters()), lr=1e-3)
    for stp in range(PROBE_STEPS):
        traj, mask, s, g, _, mid = make_batch(rng)
        with torch.no_grad():
            c = model.encode_cond(s, g)
            uu = model.sample_u(c)
            for _ in range(ROUNDS):
                uu = model.refine(c, uu)
        l = (pr_u(flat(uu)) - mid).pow(2).mean() + (pr_i(torch.cat([s, g], 1)) - mid).pow(2).mean()
        opt_p.zero_grad(set_to_none=True); l.backward(); opt_p.step()
    pr_u.eval(); pr_i.eval()

    er = np.random.default_rng(4242)
    E_traj, E_mask, E_s, E_g, _, E_mid = make_batch(er, b=512)
    with torch.no_grad():
        E_et = model.e_target(E_traj, E_mask)
        E_c = model.encode_cond(E_s, E_g)
        u0 = model.sample_u(E_c)
        ur = u0
        for _ in range(ROUNDS):
            ur = model.refine(E_c, ur)
        mse = lambda a, b: (a - b).pow(2).mean().item()
        row = dict(
            tag=tag, macc=macc,
            collapse=pair_cos(ur), collapse_flow=pair_cos(u0), collapse_et=pair_cos(E_et),
            edim=eff_dim(ur), edim_et=eff_dim(E_et),
            info=mse(pr_u(flat(ur)), E_mid), info_interp=mse(pr_i(torch.cat([E_s, E_g], 1)), E_mid),
            cos_et=F.cosine_similarity(flat(ur), flat(E_et), dim=1).mean().item(),
        )
    print(f"  {tag:<22} 塌度 {row['collapse']:.4f} | 有效維 {row['edim']:>4} | 資訊 {row['info']:.4f} "
          f"(內插 {row['info_interp']:.4f}) | cos(真et) {row['cos_et']:+.3f}", flush=True)
    return row




# ────────────────────────────────────────────────────────────
# 單變體模式：由 slurm job array 每格跑一個，結果寫 JSON，最後彙整
# ────────────────────────────────────────────────────────────
import json

# 參數搜尋（主人 2026-08-23 裁示：predictor 往小搜、EMA momentum 一起掃）
VARIANTS = {
    # predictor 大小（byol）—— 往小的方向，對照真 e_target 的有效維 47
    "byol_h32":    ("byol", 0.5, dict(LACOT_PRED_HIDDEN="32")),
    "byol_h64":    ("byol", 0.5, dict(LACOT_PRED_HIDDEN="64")),
    "byol_h128":   ("byol", 0.5, dict(LACOT_PRED_HIDDEN="128")),
    "byol_h256":   ("byol", 0.5, dict(LACOT_PRED_HIDDEN="256")),
    "byol_h1024":  ("byol", 0.5, dict(LACOT_PRED_HIDDEN="1024")),   # 舊值，當對照
    # EMA momentum
    "ema_m99":     ("ema",        0.5, dict(LACOT_EMA_M="0.99")),
    "ema_m996":    ("ema",        0.5, dict(LACOT_EMA_M="0.996")),
    "ema_m999":    ("ema",        0.5, dict(LACOT_EMA_M="0.999")),
    "cema_m99":    ("center_ema", 0.5, dict(LACOT_EMA_M="0.99")),
    "cema_m996":   ("center_ema", 0.5, dict(LACOT_EMA_M="0.996")),
    "cema_m999":   ("center_ema", 0.5, dict(LACOT_EMA_M="0.999")),
}

name = os.environ.get("VARIANT")
if name is None or name not in VARIANTS:
    print(f"用法：VARIANT=<name> python exp_anticollapse_one.py")
    print(f"可選：{', '.join(VARIANTS)}")
    sys.exit(2)

kind, lam, extra = VARIANTS[name]
for k, v in extra.items():          # 該格自己的超參數
    os.environ[k] = v
EMA_M = float(os.environ.get("LACOT_EMA_M", 0.99))
print(f"\n=== 變體 {name}（kind={kind} λ={lam} {extra}）===", flush=True)
row = run(kind, lam, name)
row.update(extra)
out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "results", f"anticollapse_{name}.json")
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w") as f:
    json.dump(row, f, indent=1)
print(f"寫入 {out}", flush=True)
