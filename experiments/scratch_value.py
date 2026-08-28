"""Step 0 of 'our version': a goal-conditioned V-function, GCIVL-style expectile
regression, on pointmaze STATE data. This is the frozen anchor every value idea
(AWR, value-shaped e_target, value-directed refine) will stand on.
  V(s,g) <- expectile_tau[ r + gamma*(1-done)*V(s',g) ],  r=0 at goal else -1.
HEALTH CHECK (must pass before trusting it): V rises along real trajectories toward
the goal (monotone-ish), and V(s, near-goal) > V(s, far-goal). Saves weights on pass.
"""
import os, sys, numpy as np, torch
from torch import nn
import torch.nn.functional as F

# 資料位置：預設走官方 OGBENCH_DATA_DIR，沒設才用本機 archive
CKPT_DIR = os.environ.get("LACOT_CKPT_DIR",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints"))
os.makedirs(CKPT_DIR, exist_ok=True)
OGB_DATA = os.environ.get("OGBENCH_DATA_DIR", "/archive/cymaxwelllee/data/ogbench")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # lacot repo root

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device, flush=True)
ENV_NAME = "pointmaze-medium-navigate-v0"
d = np.load(f"{OGB_DATA}/{ENV_NAME}.npz")
OBS = np.asarray(d["observations"], np.float32)
TERM = np.asarray(d["terminals"], bool)
N = OBS.shape[0]; ends = np.flatnonzero(TERM); starts = np.concatenate([[0], ends[:-1] + 1])
traj_end = np.empty(N, np.int64)
for s0, e0 in zip(starts, ends):
    traj_end[s0:e0 + 1] = e0
nonend = np.flatnonzero(traj_end != np.arange(N))  # i where i+1 is the same trajectory (s' = OBS[i+1])
mu, sd = OBS.mean(0), OBS.std(0) + 1e-6
GAMMA, TAU, GOAL_TOL, B = 0.99, 0.9, 0.5, 512

def nrm(x):
    return (x - mu) / sd

def make_v_batch(rng):
    i = nonend[rng.integers(0, len(nonend), size=B)]
    te = traj_end[i]
    s = OBS[i]; sp = OBS[i + 1]
    # goal relabel: 70% hindsight future on same traj (geometric), 30% random state
    g = np.empty_like(s)
    hind = rng.random(B) < 0.7
    for k in range(B):
        if hind[k]:
            gi = min(i[k] + 1 + int(rng.geometric(0.02)), int(te[k]))
            g[k] = OBS[gi]
        else:
            g[k] = OBS[rng.integers(0, N)]
    done = (np.linalg.norm(sp - g, axis=1) < GOAL_TOL).astype(np.float32)  # s' reached g?
    r = done - 1.0  # 0 at goal, -1 otherwise
    T = lambda x: torch.from_numpy(np.asarray(x, np.float32)).to(device)
    return T(nrm(s)), T(nrm(sp)), T(nrm(g)), T(r), T(done)

def sota_mlp(i, h, o, n=3):
    L, p = [], i
    for _ in range(n):
        lin = nn.Linear(p, h); nn.init.xavier_uniform_(lin.weight); nn.init.zeros_(lin.bias)
        L += [lin, nn.GELU(), nn.LayerNorm(h)]; p = h
    lin = nn.Linear(p, o); nn.init.xavier_uniform_(lin.weight); nn.init.zeros_(lin.bias)
    return nn.Sequential(*L, lin)

torch.manual_seed(0); rng = np.random.default_rng(0)
V = sota_mlp(4, 512, 1).to(device)
V_targ = sota_mlp(4, 512, 1).to(device); V_targ.load_state_dict(V.state_dict())
for p in V_targ.parameters():
    p.requires_grad_(False)
opt = torch.optim.Adam(V.parameters(), lr=3e-4)
POLYAK = 0.005
def Vsg(s, g):
    return V(torch.cat([s, g], -1)).squeeze(-1)
def Vsg_targ(s, g):
    return V_targ(torch.cat([s, g], -1)).squeeze(-1)
def expectile_loss(diff, tau):
    w = torch.where(diff > 0, tau, 1 - tau)
    return (w * diff.pow(2)).mean()

print("training V (GCIVL expectile, target net) ...", flush=True)
for stp in range(15000):
    s, sp, g, r, done = make_v_batch(rng)
    with torch.no_grad():
        target = r + GAMMA * (1 - done) * Vsg_targ(sp, g)  # bootstrap off the TARGET net
    diff = target - Vsg(s, g)
    loss = expectile_loss(diff, TAU)
    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    with torch.no_grad():  # Polyak update target net
        for pt, p in zip(V_targ.parameters(), V.parameters()):
            pt.mul_(1 - POLYAK).add_(POLYAK * p)
    if (stp + 1) % 3000 == 0:
        print(f"  step {stp+1}  loss {loss.item():.4f}  V(s,g) mean {Vsg(s,g).mean().item():.3f}", flush=True)
V.eval()

# ---- HEALTH CHECK (metrics appropriate to how AWR uses V) ----
print("\n=== health check (must pass to trust V) ===", flush=True)
# 1) along-trajectory rise + CHUNK-level advantage sign (this is exactly what AWR reads:
#    A = V(s_{t+CHUNK},g) - V(s_t,g); it must be positive as the agent nears its goal).
# HINDSIGHT-goal advantage (the correct frame: g is a state the agent WAS heading to,
# not the multi-goal episode endpoint). AWR reads exactly A = V(s_{t+CH},g) - V(s_t,g).
rng2 = np.random.default_rng(1); adv_pos = []; adv_pos_ref = []
CH = 4
for _ in range(6000):
    e0 = ends[rng2.integers(len(ends))]; s0 = starts[np.searchsorted(ends, e0)]
    if e0 - s0 < 60:
        continue
    t = rng2.integers(s0, e0 - 40)
    H = rng2.integers(CH + 4, 40)                 # hindsight goal H steps ahead of s_t
    g = OBS[min(t + H, e0)]
    st, stc = OBS[t], OBS[t + CH]                  # s_t and s_{t+CH} (a step toward g)
    with torch.no_grad():
        vt = Vsg(torch.from_numpy(nrm(st[None])).float().to(device), torch.from_numpy(nrm(g[None])).float().to(device)).item()
        vtc = Vsg(torch.from_numpy(nrm(stc[None])).float().to(device), torch.from_numpy(nrm(g[None])).float().to(device)).item()
    adv_pos.append(vtc > vt)                       # advancing CH steps toward g should RAISE V
    adv_pos_ref.append(np.linalg.norm(stc - g) < np.linalg.norm(st - g))  # euclidean ref
print(f"  1) HINDSIGHT-goal advantage sign (AWR reads this): {np.mean(adv_pos):.2f}  (euclidean ref {np.mean(adv_pos_ref):.2f}; want >0.8)", flush=True)
chunk_up = adv_pos
# 2) near goal has higher V than far goal
rng3 = np.random.default_rng(2); near_hi = []
for _ in range(500):
    e0 = ends[rng3.integers(len(ends))]; s0 = starts[np.searchsorted(ends, e0)]
    if e0 - s0 < 20:
        continue
    t = rng3.integers(s0, e0 - 10)
    g_near = OBS[min(t + 5, e0)]; g_far = OBS[e0]
    with torch.no_grad():
        vn = Vsg(torch.from_numpy(nrm(OBS[t:t+1])).float().to(device), torch.from_numpy(nrm(g_near[None])).float().to(device)).item()
        vf = Vsg(torch.from_numpy(nrm(OBS[t:t+1])).float().to(device), torch.from_numpy(nrm(g_far[None])).float().to(device)).item()
    near_hi.append(vn > vf)
print(f"  2) V(near goal) > V(far goal): {np.mean(near_hi):.2f} (want ~>0.8)", flush=True)

ok = np.mean(chunk_up) > 0.75 and np.mean(near_hi) > 0.8
print(f"\n=> HEALTH {'PASS' if ok else 'FAIL'}  (hindsight-advantage-sign, near>far)", flush=True)
if ok:
    torch.save({"state": V.state_dict(), "mu": mu, "sd": sd, "gamma": GAMMA, "tau": TAU}, os.path.join(CKPT_DIR, "scratch_value.pt"))
    print("   saved V -> scratch_value.pt", flush=True)
