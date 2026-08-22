"""flow 只是還沒訓練夠嗎？—— 訓練步數掃描（主人 2026-08-23）

背景: 官方 GCBC 用 batch 1024 × 1,000,000 步；我們今晚的所有實驗是 batch 64 × 3,500 步
      （compute 差約 4500 倍）。而訓練 log 顯示 flow 的 l_nf/dim 在 2000 步時
      仍在下降（-1.91 -> -2.70，沒收斂）。
      ⇒ 「flow 0.523 vs regression 0.947」可能只是【在對 flow 不利的時間點量的】。

做法: 單一 run 訓練到 MAX_STEPS，在多個 checkpoint 各量一次 flow / regression 的
      對齊度，畫出兩條曲線。密度估計本來就比點估計收斂慢，所以要看的是【趨勢】:

  flow 一直爬、regression 早就平  => 今晚的結論作廢，flow 只是需要時間
  flow 也平了、差距不變           => 差距是結構性的，不是訓練量的問題
  flow 爬但爬不完                 => 量化差距還剩多少，決定值不值得繼續投

⛔ 每個 checkpoint 都帶「打亂 cond」的退化控制，沒有控制的數字不算數。
"""
import os, sys, time, json, numpy as np, torch
from torch import nn
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lacot.e_target import PerceiverPooler
from lacot.nf_head import Flow

OGB_DATA = os.environ.get("OGBENCH_DATA_DIR", "/archive/cymaxwelllee/data/ogbench")
device = "cuda" if torch.cuda.is_available() else "cpu"

B = int(os.environ.get("LACOT_BATCH", 64))
MAX_STEPS = int(os.environ.get("LACOT_MAX_STEPS", 100_000))
STEPS1 = int(os.environ.get("LACOT_STEPS1", 5_000))
CKPTS = [1_000, 2_000, 3_500, 5_000, 10_000, 20_000, 35_000, 50_000, 75_000, 100_000]
CKPTS = [c for c in CKPTS if c <= MAX_STEPS]

T_CAP, D_MODEL, K, GEOM_P, TEMP, COND = 16, 256, 64, 0.02, 0.1, 256
DIM = K * D_MODEL
WANDER_MAX = 3.0
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "u_steps_curve.json")

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
print(f"device {device} | batch {B} | stage1 {STEPS1} | stage2 {MAX_STEPS} | dim {DIM}", flush=True)


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


def cosm(a, b):
    return F.cosine_similarity(a.reshape(a.shape[0], -1), b.reshape(b.shape[0], -1), dim=1).mean().item()


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


print(f"stage 1: contrastive e_target ({STEPS1} 步) ...", flush=True)
accs = []
t0 = time.time()
for stp in range(STEPS1):
    traj, mask, s, g = make_batch(rng)
    et = etarget(traj, mask); q = q_pooler(torch.stack([sg_c(s), sg_c(g)], 1))
    logits = (F.normalize(q.reshape(B, -1), dim=1) @ F.normalize(et.reshape(B, -1), dim=1).t()) / TEMP
    loss = 0.5 * (F.cross_entropy(logits, lab) + F.cross_entropy(logits.t(), lab))
    opt1.zero_grad(set_to_none=True); loss.backward(); opt1.step()
    if stp >= STEPS1 - 50:
        accs.append((logits.argmax(1) == lab).float().mean().item())
MATCH_ACC = float(np.mean(accs))
print(f"  match-acc {MATCH_ACC:.3f}  ({time.time()-t0:.0f}s)", flush=True)
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


@torch.no_grad()
def sample_at_T(cond, T):
    z = torch.randn(cond.shape[0], K, D_MODEL, device=device) * T
    u = z
    for i in reversed(range(len(flow.blocks))):
        if i < len(flow.blocks) - 1:
            u = flow.perm.inverse(u)
        u = flow.blocks[i].inverse(u, cond)
    return u


# 固定的評估批，讓每個 checkpoint 量在同一批上（⛔ 不同批會混入抽樣噪聲）
eval_rng = np.random.default_rng(12345)
E_traj, E_mask, E_s, E_g = make_batch(eval_rng, b=256)


@torch.no_grad()
def measure(step, nll_dim, reg_loss):
    for m in (cond_enc, cond_head, flow, reg):
        m.eval()
    et = etarget(E_traj, E_mask)
    cond = condvec(E_s, E_g)
    shuf = cond[torch.randperm(cond.shape[0], device=device)]
    row = dict(
        step=step, nll_dim=nll_dim, reg_loss=reg_loss, match_acc=MATCH_ACC,
        flow_mode=cosm(sample_at_T(cond, 0.0), et),
        flow_t1=cosm(sample_at_T(cond, 1.0), et),
        reg=cosm(reg(cond).reshape(-1, K, D_MODEL), et),
        flow_shuf=cosm(sample_at_T(shuf, 0.0), et),
        reg_shuf=cosm(reg(shuf).reshape(-1, K, D_MODEL), et),
        elapsed=round(time.time() - t0),
    )
    for m in (cond_enc, cond_head, flow, reg):
        m.train()
    print(f"  [{step:>6}] flow_mode {row['flow_mode']:.3f} | flow_T1 {row['flow_t1']:.3f} | "
          f"reg {row['reg']:.3f} | nll/dim {nll_dim:.4f} | shuf {row['flow_shuf']:+.3f}/{row['reg_shuf']:+.3f} "
          f"| {row['elapsed']}s", flush=True)
    return row


print(f"stage 2: flow ‖ regression, 到 {MAX_STEPS} 步，checkpoints={CKPTS}", flush=True)
curve = []
t0 = time.time()
for stp in range(1, MAX_STEPS + 1):
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
    if stp in CKPTS:
        curve.append(measure(stp, l_nf.item(), l_reg.item()))
        with open(OUT, "w") as f:
            json.dump(curve, f, indent=1)

print("\n" + "=" * 78, flush=True)
print(f"{'step':>7} {'flow mode':>10} {'flow T=1':>9} {'regress':>8} {'gap':>7} {'nll/dim':>9} {'shuf':>16}")
print("-" * 78)
for r in curve:
    print(f"{r['step']:>7} {r['flow_mode']:>10.3f} {r['flow_t1']:>9.3f} {r['reg']:>8.3f} "
          f"{r['reg']-r['flow_mode']:>+7.3f} {r['nll_dim']:>9.4f} "
          f"{r['flow_shuf']:>+7.3f}/{r['reg_shuf']:>+7.3f}")
print("=" * 78)
if len(curve) >= 2:
    a, b = curve[0], curve[-1]
    print(f"\n[趨勢] flow_mode {a['flow_mode']:.3f} -> {b['flow_mode']:.3f} ({b['flow_mode']-a['flow_mode']:+.3f})")
    print(f"       regress   {a['reg']:.3f} -> {b['reg']:.3f} ({b['reg']-a['reg']:+.3f})")
    print(f"       差距      {a['reg']-a['flow_mode']:+.3f} -> {b['reg']-b['flow_mode']:+.3f}")
    print(f"\n曲線存到 {OUT}")
