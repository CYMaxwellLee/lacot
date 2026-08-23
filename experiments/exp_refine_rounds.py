"""想越多輪，u 會越好嗎？—— test-time scaling 的直接檢驗（主人 2026-08-23）

為什麼現在才有意義：早上測的時候 refine 是塌的（一輪就把所有 u 壓成 batch 平均），
多跑幾輪只是塌得更徹底。ema_m996 把塌擋住之後，這個問題才重新有答案可找。

⚠️ 訓練只練 ROUNDS_TRAIN=3 輪，推論跑到 12 輪本身就是 OOD ——
   所以要看的是【曲線形狀】，不是單一個點：
     一路升        => 真的有 test-time scaling，想越久越準
     升到某輪掉頭  => 有最佳輪數，超過就開始壞（記下那個轉折點）
     一開始就平    => refine 只是一步到位，多想沒用

每一輪都量三件（跟前面的實驗同一套尺，才可比）：
  塌度      batch 內兩兩 cosine（真 e_target ≈ 0.027）
  路徑資訊  probe 從 u 還原路徑中點，⛔ 對照「只用 (s,g) 內插」
  cos(真et) 方向上離正確答案多遠
"""
import os, sys, copy, json, numpy as np, torch
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
DIM = K * D_MODEL
STEPS1 = int(os.environ.get("LACOT_STEPS1", 1200))
STEPS2 = int(os.environ.get("LACOT_STEPS2", 3000))
PROBE_STEPS = int(os.environ.get("LACOT_PROBE_STEPS", 1200))
SEED = int(os.environ.get("LACOT_SEED", 0))
EMA_M = float(os.environ.get("LACOT_EMA_M", 0.996))     # 上一輪掃出來的最佳
ROUNDS_TRAIN = 3
MAX_ROUNDS = int(os.environ.get("LACOT_MAX_ROUNDS", 12))

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
print(f"device {device} | seed {SEED} | ema_m {EMA_M} | 訓練 {ROUNDS_TRAIN} 輪、推論到 {MAX_ROUNDS} 輪", flush=True)


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
    z = F.normalize(flat(x), dim=1); m = z @ z.t(); n = z.shape[0]
    return ((m.sum() - m.diag().sum()) / (n * (n - 1))).item()


torch.manual_seed(SEED); rng = np.random.default_rng(SEED)
model = LaCoTActorState(state_dim=2, d_model=D_MODEL, k=K, action_dim=ADIM,
                        chunk_len=CHUNK, cond_dim=COND).to(device)
sg_c = mlp(2, 512, 512).to(device)
q_pooler = PerceiverPooler(512, D_MODEL, K, 2, 4).to(device)
opt1 = torch.optim.Adam(list(model.traj_enc.parameters()) + list(model.e_pooler.parameters())
                        + list(sg_c.parameters()) + list(q_pooler.parameters()), lr=1e-3)
lab = torch.arange(B, device=device)
print("stage 1 ...", flush=True)
for stp in range(STEPS1):
    traj, mask, s, g, _, _ = make_batch(rng)
    et = model.e_target(traj, mask)
    q = q_pooler(torch.stack([sg_c(s), sg_c(g)], 1))
    lg = (F.normalize(q.reshape(B, -1), dim=1) @ F.normalize(et.reshape(B, -1), dim=1).t()) / TEMP
    loss = 0.5 * (F.cross_entropy(lg, lab) + F.cross_entropy(lg.t(), lab))
    opt1.zero_grad(set_to_none=True); loss.backward(); opt1.step()
model.freeze_front_end()
print(f"  match-acc {(lg.argmax(1)==lab).float().mean().item():.3f}", flush=True)

refine_ema = copy.deepcopy(model.refine)
for p in refine_ema.parameters():
    p.requires_grad_(False)
mods = [model.cond_enc, model.cond_head, model.flow, model.refine, model.action_head]
opt2 = torch.optim.Adam([p for m in mods for p in m.parameters()], lr=5e-4)
ZERO = torch.zeros(B, K, D_MODEL, device=device)

print(f"stage 2（ema consistency, m={EMA_M}）...", flush=True)
for stp in range(STEPS2):
    traj, mask, s, g, act, _ = make_batch(rng)
    with torch.no_grad():
        et = model.e_target(traj, mask)
    cond = model.encode_cond(s, g)
    cat = lambda uu: torch.cat([cond, flat(uu)], -1)
    l_nf = model.flow.nll(et, cond) / DIM
    l_anchor = model.action_head.nll(model.action_head(cat(et)), act).mean()
    u = model.flow.sample(B, cond).detach(); us = [u]
    for _ in range(ROUNDS_TRAIN):
        u = model.refine(cond, u); us.append(u)
    l_refine = sum(model.action_head.nll(model.action_head(cat(us[r + 1])), act).mean()
                   for r in range(ROUNDS_TRAIN)) / ROUNDS_TRAIN
    l_cons = us[0].new_zeros(())
    for r in range(ROUNDS_TRAIN):
        with torch.no_grad():
            tgt = refine_ema(cond, us[r])
        l_cons = l_cons + (us[r + 1] - tgt).pow(2).mean()
    l_cons = l_cons / ROUNDS_TRAIN
    l_null = model.action_head.nll(model.action_head(torch.cat([cond, flat(ZERO)], -1)), act).mean()
    (l_nf + l_anchor + l_refine + 0.5 * l_cons + l_null).backward()
    torch.nn.utils.clip_grad_norm_([p for m in mods for p in m.parameters()], 1.0)
    opt2.step(); opt2.zero_grad(set_to_none=True)
    with torch.no_grad():
        for pe, p in zip(refine_ema.parameters(), model.refine.parameters()):
            pe.mul_(EMA_M).add_(p, alpha=1 - EMA_M)
model.eval()

# probe：用【訓練時的輪數】訓練，之後拿去量所有輪數（⛔ 不能每輪各訓一個 probe，
# 那樣 probe 的容量差異會混進結果裡）
pr_u = mlp(DIM, 512, 2, n=3).to(device)
pr_i = mlp(4, 512, 2, n=3).to(device)
opt_p = torch.optim.Adam(list(pr_u.parameters()) + list(pr_i.parameters()), lr=1e-3)
print("probe ...", flush=True)
for stp in range(PROBE_STEPS):
    traj, mask, s, g, _, mid = make_batch(rng)
    with torch.no_grad():
        c = model.encode_cond(s, g)
        uu = model.sample_u(c)
        for _ in range(ROUNDS_TRAIN):
            uu = model.refine(c, uu)
    l = (pr_u(flat(uu)) - mid).pow(2).mean() + (pr_i(torch.cat([s, g], 1)) - mid).pow(2).mean()
    opt_p.zero_grad(set_to_none=True); l.backward(); opt_p.step()
pr_u.eval(); pr_i.eval()

er = np.random.default_rng(4242)
E_traj, E_mask, E_s, E_g, E_act, E_mid = make_batch(er, b=512)
rows = []
with torch.no_grad():
    E_et = model.e_target(E_traj, E_mask)
    E_c = model.encode_cond(E_s, E_g)
    interp = (pr_i(torch.cat([E_s, E_g], 1)) - E_mid).pow(2).mean().item()
    et_col, et_info = pair_cos(E_et), (pr_u(flat(E_et)) - E_mid).pow(2).mean().item()
    u = model.sample_u(E_c)
    for r in range(MAX_ROUNDS + 1):
        if r > 0:
            u = model.refine(E_c, u)
        rows.append(dict(
            round=r, collapse=pair_cos(u),
            info=(pr_u(flat(u)) - E_mid).pow(2).mean().item(),
            cos_et=F.cosine_similarity(flat(u), flat(E_et), dim=1).mean().item(),
            norm=flat(u).norm(dim=1).mean().item()))

print("\n" + "=" * 68)
print(f"{'輪數':>5} {'塌度':>9} {'路徑資訊':>10} {'贏內插':>7} {'cos(真et)':>10} {'‖u‖':>8}")
print("-" * 68)
for r in rows:
    win = "✔" if r["info"] < interp * 0.95 else "✘"
    mark = "  <- 訓練到這輪" if r["round"] == ROUNDS_TRAIN else ""
    print(f"{r['round']:>5} {r['collapse']:>9.4f} {r['info']:>10.4f} {win:>7} "
          f"{r['cos_et']:>+10.3f} {r['norm']:>8.1f}{mark}")
print("-" * 68)
print(f"{'真值':>5} {et_col:>9.4f} {et_info:>10.4f} {'—':>7} {1.0:>+10.3f}")
print(f"{'內插':>5} {'—':>9} {interp:>10.4f}")
print("=" * 68)

best = min(rows, key=lambda r: r["info"])
print(f"\n路徑資訊最好的是第 {best['round']} 輪（{best['info']:.4f}）")
print(f"cos 最好的是第 {max(rows, key=lambda r: r['cos_et'])['round']} 輪")
out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "results", f"rounds_seed{SEED}.json")
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w") as f:
    json.dump(dict(seed=SEED, ema_m=EMA_M, interp=interp, et_collapse=et_col,
                   et_info=et_info, rows=rows), f, indent=1)
print(f"寫入 {out}")
