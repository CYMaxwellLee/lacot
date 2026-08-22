"""M4 ORACLE-future gate (per 主人): feed the head the TRUE e_target (encoded from
the expert's real future trajectory to the goal), bypassing the noisy flow, and roll
out. This is the CEILING of the head+e_target design.
  * ORACLE      = head(cond, e_target_from_expert_future)   <- upper bound
  * GCBC floor  = standalone (s,g)->action, no u
  * null floor  = head(cond, 0), trained (null-token)
  * M4 R=3      = head(cond, flow-sampled + refined u)       <- the real inference path
Reading: oracle high & M4 low  => flow is the bottleneck (train flow/refine).
         oracle also low        => head/e_target design itself is limited.
Includes an expert-path sanity check (does the simulated future reach the goal?).
"""
import os, sys, numpy as np, torch
from torch import nn
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # m4 repo root
from lacot.e_target import PerceiverPooler
from lacot.nf_head import Flow
from lacot.model import RefineOperator
import ogbench

# 資料位置：預設走官方 OGBENCH_DATA_DIR，沒設才用本機 archive
OGB_DATA = os.environ.get("OGBENCH_DATA_DIR", "/archive/cymaxwelllee/data/ogbench")

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device, flush=True)
ENV_NAME = "pointmaze-medium-navigate-v0"
d = np.load(f"{OGB_DATA}/{ENV_NAME}.npz")
OBS = np.asarray(d["observations"], np.float32); ACT = np.asarray(d["actions"], np.float32); TERM = np.asarray(d["terminals"], bool)
N = OBS.shape[0]; ends = np.flatnonzero(TERM); starts = np.concatenate([[0], ends[:-1] + 1])
traj_end = np.empty(N, np.int64)
for s0, e0 in zip(starts, ends):
    traj_end[s0:e0 + 1] = e0
mu, sd = OBS.mean(0), OBS.std(0) + 1e-6
MU = torch.tensor(mu, device=device); SD = torch.tensor(sd, device=device)
B, T_CAP, D_MODEL, K, GEOM_P, TEMP, COND, CHUNK, ADIM = 64, 16, 256, 64, 0.02, 0.1, 256, 4, 2
DIM = K * D_MODEL

def make_batch(rng):
    rows, goals = [], []
    while len(rows) < B:
        r = int(rng.integers(0, N)); te = int(traj_end[r])
        if te - r < CHUNK:
            continue
        gr = min(r + int(rng.geometric(GEOM_P)), te)
        if gr - r < CHUNK:
            continue
        rows.append(r); goals.append(gr)
    rows, goals = np.array(rows), np.array(goals)
    idxs = [np.unique(np.linspace(rows[i], goals[i], min(T_CAP, goals[i] - rows[i] + 1)).round().astype(int)) for i in range(B)]
    Tmax = max(len(ix) for ix in idxs)
    traj = np.zeros((B, Tmax, 2), np.float32); mask = np.ones((B, Tmax), bool)
    for i, ix in enumerate(idxs):
        traj[i, :len(ix)] = (OBS[ix] - mu) / sd; mask[i, :len(ix)] = False
    s = (OBS[rows] - mu) / sd; g = (OBS[goals] - mu) / sd
    act = np.stack([ACT[r:r + CHUNK] for r in rows]).astype(np.float32)
    T = lambda x: torch.from_numpy(x.astype(np.float32)).to(device)
    return T(traj), torch.from_numpy(mask).to(device), T(s), T(g), T(act)

def sota_mlp(i, h, o, n=2):
    L, p = [], i
    for _ in range(n):
        lin = nn.Linear(p, h); nn.init.xavier_uniform_(lin.weight); nn.init.zeros_(lin.bias)
        L += [lin, nn.GELU(), nn.LayerNorm(h)]; p = h
    lin = nn.Linear(p, o); nn.init.xavier_uniform_(lin.weight); nn.init.zeros_(lin.bias)
    return nn.Sequential(*L, lin)

class ActionMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = sota_mlp(COND + DIM, 512, CHUNK * ADIM, n=3)
    def forward(self, cond, u):
        return self.net(torch.cat([cond, u.reshape(u.shape[0], -1)], -1)).reshape(-1, CHUNK, ADIM)

torch.manual_seed(0); rng = np.random.default_rng(0)
traj_enc = sota_mlp(2, 512, 512).to(device); e_pooler = PerceiverPooler(512, D_MODEL, K, 2, 4).to(device)
sg_c = sota_mlp(2, 512, 512).to(device); q_pooler = PerceiverPooler(512, D_MODEL, K, 2, 4).to(device)
opt1 = torch.optim.Adam([p for m in (traj_enc, e_pooler, sg_c, q_pooler) for p in m.parameters()], lr=1e-3)
lab = torch.arange(B, device=device)
def etarget(traj, mask):
    Bc, Tc, _ = traj.shape
    return e_pooler(traj_enc(traj.reshape(Bc * Tc, 2)).reshape(Bc, Tc, 512), key_padding_mask=mask)
print("stage 1 contrastive e_target ...", flush=True)
for stp in range(1500):
    traj, mask, s, g, _ = make_batch(rng)
    et = etarget(traj, mask); q = q_pooler(torch.stack([sg_c(s), sg_c(g)], 1))
    logits = (F.normalize(q.reshape(B, -1), dim=1) @ F.normalize(et.reshape(B, -1), dim=1).t()) / TEMP
    loss = 0.5 * (F.cross_entropy(logits, lab) + F.cross_entropy(logits.t(), lab))
    opt1.zero_grad(set_to_none=True); loss.backward(); opt1.step()
for m in (traj_enc, e_pooler):
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
print(f"  e_target match-acc {(logits.argmax(1)==lab).float().mean().item():.3f}", flush=True)

cond_enc = sota_mlp(2, 512, 512).to(device); cond_head = sota_mlp(1024, 512, COND).to(device)
flow = Flow(token_dim=D_MODEL, seq_len=K, n_blocks=4, cond_dim=COND).to(device)
refine = RefineOperator(COND, K, D_MODEL, hidden=256).to(device)
ahead = ActionMLP().to(device)
f_mods = [cond_enc, cond_head, flow, refine, ahead]
opt2 = torch.optim.Adam([p for m in f_mods for p in m.parameters()], lr=5e-4)
gcbc_enc = sota_mlp(2, 512, 512).to(device); gcbc_head = sota_mlp(1024, 512, CHUNK * ADIM, n=3).to(device)
opt_g = torch.optim.Adam([p for m in (gcbc_enc, gcbc_head) for p in m.parameters()], lr=5e-4)
def condvec(s, g):
    return cond_head(torch.cat([cond_enc(s), cond_enc(g)], 1))
def gcbc(s, g):
    return gcbc_head(torch.cat([gcbc_enc(s), gcbc_enc(g)], 1)).reshape(-1, CHUNK, ADIM)
mse = lambda p, a: (p - a).pow(2).mean()
ZERO = torch.zeros(B, K, D_MODEL, device=device)
print("stage 2 flow+refine+action(+null-u) & GCBC ...", flush=True)
for stp in range(2000):
    traj, mask, s, g, act = make_batch(rng)
    with torch.no_grad():
        et = etarget(traj, mask)
    cond = condvec(s, g)
    l_nf = flow.nll(et, cond) / DIM
    l_anchor = mse(ahead(cond, et), act)
    l_null = mse(ahead(cond, ZERO), act)
    u = flow.sample(B, cond).detach(); us = [u]
    for _ in range(3):
        u = refine(cond, u); us.append(u)
    l_cons = sum((us[r] - us[r + 1].detach()).pow(2).mean() for r in range(3)) / 3
    l_refine = sum(mse(ahead(cond, us[r + 1]), act) for r in range(3)) / 3
    total = l_nf + l_anchor + l_refine + l_null + 0.5 * l_cons
    opt2.zero_grad(set_to_none=True); total.backward()
    torch.nn.utils.clip_grad_norm_([p for m in f_mods for p in m.parameters()], 1.0); opt2.step()
    l_gcbc = mse(gcbc(s, g), act)
    opt_g.zero_grad(set_to_none=True); l_gcbc.backward(); opt_g.step()
    if (stp + 1) % 1000 == 0:
        print(f"  step {stp+1}  l_anchor {l_anchor.item():.4f} l_null {l_null.item():.4f} l_gcbc {l_gcbc.item():.4f}", flush=True)
for m in f_mods + [gcbc_enc, gcbc_head]:
    m.eval()

os.environ.setdefault("OGBENCH_DATA_DIR", OGB_DATA)
env, _, _ = ogbench.make_env_and_datasets(ENV_NAME)
# ⛔ 官方標準（impls/utils/evaluation.py + main.py）：跑滿 env 自己的 horizon，
# 每個 task 20 集。舊版自己砍成 500/6 集 = 難度自訂、數字不能跟官方並排。
MAXH = int(os.environ.get("LACOT_EVAL_MAXH", env.spec.max_episode_steps or 1000))
N_TASKS = len(env.unwrapped.task_infos); SEEDS = int(os.environ.get("LACOT_EVAL_EPISODES", 20))  # 官方 eval_episodes=20
GAIN = 5.0

def expert_positions(obs, goal, horizon=150):
    """Simulate the env's own BFS proportional expert from obs to goal (point dynamics
    qpos += 0.2*a). Returns the future position trajectory — the 'true future' e_target."""
    xy = np.asarray(obs, np.float64); g = np.asarray(goal, np.float64)
    poss = [xy.copy()]
    for _ in range(horizon):
        subgoal, bfs = env.unwrapped.get_oracle_subgoal(xy, g)
        here = env.unwrapped.xy_to_ij(xy)
        target = g if bfs[here[0], here[1]] == 0 else np.asarray(subgoal)
        a = np.clip(GAIN * (target - xy), -1, 1)
        xy = xy + 0.2 * a
        poss.append(xy.copy())
        if np.linalg.norm(xy - g) < 0.5:
            break
    return np.array(poss)

@torch.no_grad()
def oracle_etarget(obs, goal):
    poss = expert_positions(obs, goal)
    idx = np.unique(np.linspace(0, len(poss) - 1, min(T_CAP, len(poss))).round().astype(int))
    traj = ((poss[idx] - mu) / sd).astype(np.float32)
    tt = torch.tensor(traj, device=device)[None]
    return e_pooler(traj_enc(tt.reshape(-1, 2)).reshape(1, -1, 512), key_padding_mask=None)

# --- sanity: does the simulated expert future actually reach the goal? ---
print("\n=== expert-path sanity (must reach goal, else oracle is meaningless) ===", flush=True)
for task in (1, 2, 3):
    obs, info = env.reset(seed=task, options={"task_id": task, "render_goal": False})
    goal = np.asarray(info["goal"]); poss = expert_positions(obs, goal, horizon=300)
    print(f"  task{task}: {len(poss)} steps, start-dist {np.linalg.norm(poss[0]-goal):.2f} -> end-dist {np.linalg.norm(poss[-1]-goal):.2f}", flush=True)

def normstate(x):
    return ((torch.tensor(np.asarray(x, np.float32), device=device) - MU) / SD)[None]
@torch.no_grad()
def chunk(obs, goal, kind, R=0):
    s = normstate(obs); g = normstate(goal); cond = condvec(s, g)
    if kind == "gcbc":
        a = gcbc(s, g)[0]
    elif kind == "null":
        a = ahead(cond, torch.zeros(1, K, D_MODEL, device=device))[0]
    elif kind == "oracle":
        a = ahead(cond, oracle_etarget(obs, goal))[0]
    else:
        u = flow.sample(1, cond)
        for _ in range(R):
            u = refine(cond, u)
        a = ahead(cond, u)[0]
    return np.clip(a.cpu().numpy(), -1.0, 1.0).astype(np.float32)

def rollout(kind, R, tag):
    succ = ep = 0
    for task in range(1, N_TASKS + 1):
        for sd_ in range(SEEDS):
            obs, info = env.reset(seed=1000 * task + sd_, options={"task_id": task, "render_goal": False})
            goal = info["goal"]; success = False; steps = 0
            torch.manual_seed(7 * task + sd_)
            while steps < MAXH and not success:
                for a in chunk(obs, goal, kind, R):
                    obs, rew, term, trunc, info = env.step(a)
                    steps += 1
                    if info.get("success"):
                        success = True
                    if success or term or trunc or steps >= MAXH:
                        break
            succ += int(success); ep += 1
    print(f"  {tag}: success {succ}/{ep} = {succ/ep:.3f}", flush=True)
    return succ / ep

print(f"\n==== SUCCESS RATE + ORACLE  env={ENV_NAME}, {N_TASKS}x{SEEDS} eps, MAXH {MAXH} ====", flush=True)
rollout("oracle", 0, "ORACLE  head(cond, TRUE e_target)   <-- ceiling")
rollout("gcbc", 0, "GCBC floor (standalone, no u)")
rollout("null", 0, "null-u floor (head(cond,0), trained)")
rollout("m4", 0, "M4 refine R=0 (flow sample)")
rollout("m4", 3, "M4 refine R=3")
print("=> oracle high & M4 low => flow is the bottleneck; oracle low => head/e_target design is.", flush=True)
