"""降低 u 的維度，flow 追得上 regression 嗎？（主人 2026-08-23 要的）

背景: u 是 K*D_MODEL 維。K=64, D=256 -> 16384 維，而它描述的軌跡只有 T_CAP*2 = 32 維
      真實資訊。exp_u_why 量到 flow(mode) 0.523 vs regression 0.947 -> 懷疑是維度壓死 flow。

掃 K ∈ {64, 32, 16, 8, 4}（D_MODEL 固定 256），每個 K 從頭訓一次，量:
  - e_target match-acc     <- ⛔ 關鍵控制: K 太小 e_target 自己就爛了，那 flow 變好也沒意義
  - flow mode  cosine      <- T=0，flow 最理想的採樣
  - flow T=1   cosine      <- 實際推論用的
  - regression cosine      <- 對照組，同 cond、同步數
  - 打亂 cond 的 cosine    <- 退化控制，該掉到 0

讀法:
  flow 隨 K 下降而爬升、且 match-acc 沒掉  => 維度是病根，降維就能修
  flow 不動                                => 維度不是病根，flow 不適合這個位置
  match-acc 跟著掉                          => 該 K 的結果不算數（e_target 已經壞了）
"""
import os, sys, math, numpy as np, torch
from torch import nn
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lacot.e_target import PerceiverPooler
from lacot.nf_head import Flow

OGB_DATA = os.environ.get("OGBENCH_DATA_DIR", "/archive/cymaxwelllee/data/ogbench")
device = "cuda" if torch.cuda.is_available() else "cpu"
ENV = "pointmaze-medium-navigate-v0"
d = np.load(f"{OGB_DATA}/{ENV}.npz")
OBS = np.asarray(d["observations"], np.float32)
TERM = np.asarray(d["terminals"], bool)
N = OBS.shape[0]
ends = np.flatnonzero(TERM); starts = np.concatenate([[0], ends[:-1] + 1])
traj_end = np.empty(N, np.int64)
for s0, e0 in zip(starts, ends):
    traj_end[s0:e0 + 1] = e0
mu, sd = OBS.mean(0), OBS.std(0) + 1e-6

B, T_CAP, D_MODEL, GEOM_P, TEMP, COND = 64, 16, 256, 0.02, 0.1, 256
WANDER_MAX = 3.0
K_LIST = [64, 32, 16, 8, 4]
STEPS1, STEPS2 = 1500, 2000

print(f"device: {device}   軌跡真實資訊量 = T_CAP*2 = {T_CAP*2} 維", flush=True)


def make_batch(rng, b=B):
    rows, goals = [], []
    while len(rows) < b:
        r = int(rng.integers(0, N)); te = int(traj_end[r])
        if te - r < 4:
            continue
        gr = min(r + int(rng.geometric(GEOM_P)), te)
        if gr - r < 4:
            continue
        path = OBS[r:gr + 1]
        if np.linalg.norm(np.diff(path, axis=0), axis=1).sum() / (np.linalg.norm(path[-1] - path[0]) + 1e-6) > WANDER_MAX:
            continue
        rows.append(r); goals.append(gr)
    rows, goals = np.array(rows), np.array(goals)
    idxs = [np.unique(np.linspace(rows[i], goals[i], min(T_CAP, goals[i] - rows[i] + 1)).round().astype(int)) for i in range(b)]
    Tmax = max(len(ix) for ix in idxs)
    traj = np.zeros((b, Tmax, 2), np.float32); mask = np.ones((b, Tmax), bool)
    for i, ix in enumerate(idxs):
        traj[i, :len(ix)] = (OBS[ix] - mu) / sd; mask[i, :len(ix)] = False
    s = (OBS[rows] - mu) / sd; g = (OBS[goals] - mu) / sd
    T_ = lambda x: torch.from_numpy(x.astype(np.float32)).to(device)
    return T_(traj), torch.from_numpy(mask).to(device), T_(s), T_(g)


def sota_mlp(i, h, o, n=2):
    L, p = [], i
    for _ in range(n):
        lin = nn.Linear(p, h); nn.init.xavier_uniform_(lin.weight); nn.init.zeros_(lin.bias)
        L += [lin, nn.GELU(), nn.LayerNorm(h)]; p = h
    lin = nn.Linear(p, o); nn.init.xavier_uniform_(lin.weight); nn.init.zeros_(lin.bias)
    return nn.Sequential(*L, lin)


def cos(a, b):
    return F.cosine_similarity(a.reshape(a.shape[0], -1), b.reshape(b.shape[0], -1), dim=1).mean().item()


def run_one(K):
    DIM = K * D_MODEL
    torch.manual_seed(0); rng = np.random.default_rng(0)
    traj_enc = sota_mlp(2, 512, 512).to(device)
    e_pooler = PerceiverPooler(512, D_MODEL, K, 2, 4).to(device)
    sg_c = sota_mlp(2, 512, 512).to(device)
    q_pooler = PerceiverPooler(512, D_MODEL, K, 2, 4).to(device)
    opt1 = torch.optim.Adam([p for m in (traj_enc, e_pooler, sg_c, q_pooler) for p in m.parameters()], lr=1e-3)
    lab = torch.arange(B, device=device)

    def etarget(traj, mask):
        Bc, Tc, _ = traj.shape
        return e_pooler(traj_enc(traj.reshape(Bc * Tc, 2)).reshape(Bc, Tc, 512), key_padding_mask=mask)

    accs = []
    for stp in range(STEPS1):
        traj, mask, s, g = make_batch(rng)
        et = etarget(traj, mask); q = q_pooler(torch.stack([sg_c(s), sg_c(g)], 1))
        logits = (F.normalize(q.reshape(B, -1), dim=1) @ F.normalize(et.reshape(B, -1), dim=1).t()) / TEMP
        loss = 0.5 * (F.cross_entropy(logits, lab) + F.cross_entropy(logits.t(), lab))
        opt1.zero_grad(set_to_none=True); loss.backward(); opt1.step()
        if stp >= STEPS1 - 50:
            accs.append((logits.argmax(1) == lab).float().mean().item())
    match_acc = float(np.mean(accs))          # 最後 50 步平均，⛔ 不用單一 batch
    for m in (traj_enc, e_pooler):
        m.eval()
        for p in m.parameters():
            p.requires_grad_(False)

    cond_enc = sota_mlp(2, 512, 512).to(device)
    cond_head = sota_mlp(1024, 512, COND).to(device)
    flow = Flow(token_dim=D_MODEL, seq_len=K, n_blocks=4, cond_dim=COND).to(device)
    reg = sota_mlp(COND, 1024, DIM, n=3).to(device)
    opt_f = torch.optim.Adam([p for m in (cond_enc, cond_head, flow) for p in m.parameters()], lr=5e-4)
    opt_r = torch.optim.Adam(reg.parameters(), lr=5e-4)

    def condvec(s, g):
        return cond_head(torch.cat([cond_enc(s), cond_enc(g)], 1))

    for stp in range(STEPS2):
        traj, mask, s, g = make_batch(rng)
        with torch.no_grad():
            et = etarget(traj, mask)
        cond = condvec(s, g)
        l_nf = flow.nll(et, cond) / DIM
        opt_f.zero_grad(set_to_none=True); l_nf.backward()
        torch.nn.utils.clip_grad_norm_([p for m in (cond_enc, cond_head, flow) for p in m.parameters()], 1.0)
        opt_f.step()
        l_reg = (reg(cond.detach()).reshape(-1, K, D_MODEL) - et).pow(2).mean()
        opt_r.zero_grad(set_to_none=True); l_reg.backward(); opt_r.step()
    for m in (cond_enc, cond_head, flow, reg):
        m.eval()

    @torch.no_grad()
    def sample_at_T(cond, T):
        z = torch.randn(cond.shape[0], K, D_MODEL, device=device) * T
        u = z
        for i in reversed(range(len(flow.blocks))):
            if i < len(flow.blocks) - 1:
                u = flow.perm.inverse(u)
            u = flow.blocks[i].inverse(u, cond)
        return u

    with torch.no_grad():
        traj, mask, s, g = make_batch(rng, b=256)
        et = etarget(traj, mask); cond = condvec(s, g)
        shuf = cond[torch.randperm(cond.shape[0], device=device)]
        return dict(
            K=K, dim=DIM, acc=match_acc,
            flow_mode=cos(sample_at_T(cond, 0.0), et),
            flow_t1=cos(sample_at_T(cond, 1.0), et),
            reg=cos(reg(cond).reshape(-1, K, D_MODEL), et),
            flow_shuf=cos(sample_at_T(shuf, 0.0), et),
            reg_shuf=cos(reg(shuf).reshape(-1, K, D_MODEL), et),
            nll_dim=(flow.nll(et, cond) / DIM).item(),
        )


rows = []
for K in K_LIST:
    print(f"\n--- K={K}  (dim={K*D_MODEL}) 訓練中 ...", flush=True)
    r = run_one(K)
    rows.append(r)
    print(f"    match-acc {r['acc']:.3f} | flow_mode {r['flow_mode']:.3f} | "
          f"flow_T1 {r['flow_t1']:.3f} | reg {r['reg']:.3f} | shuf {r['flow_shuf']:+.3f}/{r['reg_shuf']:+.3f}", flush=True)

print("\n" + "=" * 84, flush=True)
print(f"{'K':>4} {'dim':>7} {'match-acc':>10} {'flow mode':>10} {'flow T=1':>9} {'regress':>8} "
      f"{'shuf-flow':>10} {'shuf-reg':>9}")
print("-" * 84)
for r in rows:
    flag = "  <- e_target 已壞，不算數" if r["acc"] < 0.80 else ""
    print(f"{r['K']:>4} {r['dim']:>7} {r['acc']:>10.3f} {r['flow_mode']:>10.3f} {r['flow_t1']:>9.3f} "
          f"{r['reg']:>8.3f} {r['flow_shuf']:>+10.3f} {r['reg_shuf']:>+9.3f}{flag}")
print("=" * 84)
best = max([r for r in rows if r["acc"] >= 0.80], key=lambda r: r["flow_mode"], default=None)
if best:
    base = rows[0]
    print(f"\n[結論] e_target 還健康(acc>=0.80)的範圍內，flow mode 最好的是 K={best['K']} "
          f"({best['flow_mode']:.3f})，相對 K=64 的 {base['flow_mode']:.3f} "
          f"{'上升' if best['flow_mode'] > base['flow_mode'] else '沒有上升'}。")
    print(f"       同一格的 regression 是 {best['reg']:.3f} —— 差距 {best['reg']-best['flow_mode']:+.3f}")
