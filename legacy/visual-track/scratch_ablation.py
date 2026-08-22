"""各種 ablation (per 主人): fpo's REAL boundary detector for proper same-segment goal
sampling (②), N=100 for significance (①), full policy sweep.
Data conditions: naive (goal capped at 1001-step traj end, crosses sub-goals)
                 segment (goal capped at the sub-trajectory end -- 'same segment')
Policy: GCBC / null-u / ORACLE(true et) / M4 R=0,1,3,5 (test-time scaling curve).
"""
import os, sys, numpy as np, torch
from torch import nn
import torch.nn.functional as F
sys.path.insert(0, "/archive/cymaxwelllee/fpo")
from wpm.models.e_target import PerceiverPooler
from wpm.models.nf_head import Flow
from wpm.models.m4 import RefineOperator
import wpm.data.pipeline as P
import ogbench

# 資料位置：預設走官方 OGBENCH_DATA_DIR，沒設才用本機 archive
OGB_DATA = os.environ.get("OGBENCH_DATA_DIR", "/archive/cymaxwelllee/data/ogbench")

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device, flush=True)
ENV_NAME = "pointmaze-medium-navigate-v0"
d = np.load(f"{OGB_DATA}/{ENV_NAME}.npz")
OBS = np.asarray(d["observations"], np.float32); ACT = np.asarray(d["actions"], np.float32)
TERM = np.asarray(d["terminals"], bool); QPOS = np.asarray(d["qpos"], np.float32)
N = OBS.shape[0]; ends = np.flatnonzero(TERM); starts = np.concatenate([[0], ends[:-1] + 1])
traj_end = np.empty(N, np.int64)
for s0, e0 in zip(starts, ends):
    traj_end[s0:e0 + 1] = e0
mu, sd = OBS.mean(0), OBS.std(0) + 1e-6
MU = torch.tensor(mu, device=device); SD = torch.tensor(sd, device=device)
B, T_CAP, D_MODEL, K, GEOM_P, TEMP, COND, CHUNK, ADIM = 64, 16, 256, 64, 0.02, 0.1, 256, 4, 2
DIM = K * D_MODEL

# --- REAL fpo sub-trajectory boundaries (maze_arrival) -> segment_end per row ---
spec = P.dataset_spec(ENV_NAME)
cfg = P.PipelineConfig(dataset_name=ENV_NAME, boundary_mode="maze_arrival", maze_reversal_window=45)
print("resolved boundary_mode:", cfg.resolved_boundary_mode(), "reversal_window", cfg.maze_reversal_window, flush=True)
boundaries = P.derive_subtrajectory_boundaries(cfg, spec, QPOS, starts, ends)
SEG_END, _ = P.subtrajectory_maps(starts, ends, boundaries, N)
nseg = np.mean([len(b) + 1 for b in boundaries]); seglen = np.mean(SEG_END - np.arange(N))
print(f"boundaries: {nseg:.2f} sub-traj/episode, mean fwd-to-segment-end {seglen:.1f} steps (vs 1001)", flush=True)
GOAL_CAP = {"naive": traj_end, "segment": SEG_END}

def make_batch(rng, mode, want_raw=False):
    cap = GOAL_CAP[mode]
    rows, goals = [], []
    while len(rows) < B:
        r = int(rng.integers(0, N)); te = int(cap[r])
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
    out = [T(traj), torch.from_numpy(mask).to(device), T(s), T(g), T(act)]
    if want_raw:
        out += [OBS[rows].copy(), OBS[goals].copy()]
    return out

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

os.environ.setdefault("OGBENCH_DATA_DIR", OGB_DATA)
env, _, _ = ogbench.make_env_and_datasets(ENV_NAME)
MAXH = min(int(env.spec.max_episode_steps or 1000), 500)
N_TASKS = len(env.unwrapped.task_infos); SEEDS = 20; GAIN = 5.0  # N = 5*20 = 100

def expert_positions(obs, goal, horizon=150):
    xy = np.asarray(obs, np.float64); gg = np.asarray(goal, np.float64); poss = [xy.copy()]
    for _ in range(horizon):
        subgoal, bfs = env.unwrapped.get_oracle_subgoal(xy, gg)
        here = env.unwrapped.xy_to_ij(xy)
        target = gg if bfs[here[0], here[1]] == 0 else np.asarray(subgoal)
        a = np.clip(GAIN * (target - xy), -1, 1); xy = xy + 0.2 * a; poss.append(xy.copy())
        if np.linalg.norm(xy - gg) < 0.5:
            break
    return np.array(poss)

def train_and_rollout(mode, variants):
    torch.manual_seed(0); rng = np.random.default_rng(0)
    traj_enc = sota_mlp(2, 512, 512).to(device); e_pooler = PerceiverPooler(512, D_MODEL, K, 2, 4).to(device)
    sg_c = sota_mlp(2, 512, 512).to(device); q_pooler = PerceiverPooler(512, D_MODEL, K, 2, 4).to(device)
    opt1 = torch.optim.Adam([p for m in (traj_enc, e_pooler, sg_c, q_pooler) for p in m.parameters()], lr=1e-3)
    lab = torch.arange(B, device=device)
    def etarget(traj, mask):
        Bc, Tc, _ = traj.shape
        return e_pooler(traj_enc(traj.reshape(Bc * Tc, 2)).reshape(Bc, Tc, 512), key_padding_mask=mask)
    for _ in range(1500):
        traj, mask, s, g, _ = make_batch(rng, mode)
        et = etarget(traj, mask); q = q_pooler(torch.stack([sg_c(s), sg_c(g)], 1))
        logits = (F.normalize(q.reshape(B, -1), dim=1) @ F.normalize(et.reshape(B, -1), dim=1).t()) / TEMP
        loss = 0.5 * (F.cross_entropy(logits, lab) + F.cross_entropy(logits.t(), lab))
        opt1.zero_grad(set_to_none=True); loss.backward(); opt1.step()
    macc = (logits.argmax(1) == lab).float().mean().item()
    for m in (traj_enc, e_pooler):
        m.eval()
        for p in m.parameters():
            p.requires_grad_(False)
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
    for _ in range(2500):
        traj, mask, s, g, act = make_batch(rng, mode)
        with torch.no_grad():
            et = etarget(traj, mask)
        cond = condvec(s, g)
        l_nf = flow.nll(et, cond) / DIM
        l_anchor = mse(ahead(cond, et), act); l_null = mse(ahead(cond, ZERO), act)
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
    for m in f_mods + [gcbc_enc, gcbc_head]:
        m.eval()
    def normstate(x):
        return ((torch.tensor(np.asarray(x, np.float32), device=device) - MU) / SD)[None]
    @torch.no_grad()
    def oracle_et(obs, goal):
        poss = expert_positions(obs, goal)
        idx = np.unique(np.linspace(0, len(poss) - 1, min(T_CAP, len(poss))).round().astype(int))
        tt = torch.tensor(((poss[idx] - mu) / sd).astype(np.float32), device=device)[None]
        return e_pooler(traj_enc(tt.reshape(-1, 2)).reshape(1, -1, 512), key_padding_mask=None)
    @torch.no_grad()
    def chunk(obs, goal, kind, R):
        s = normstate(obs); g = normstate(goal); cond = condvec(s, g)
        if kind == "gcbc":
            a = gcbc(s, g)[0]
        elif kind == "null":
            a = ahead(cond, torch.zeros(1, K, D_MODEL, device=device))[0]
        elif kind == "oracle":
            a = ahead(cond, oracle_et(obs, goal))[0]
        else:
            u = flow.sample(1, cond)
            for _ in range(R):
                u = refine(cond, u)
            a = ahead(cond, u)[0]
        return np.clip(a.cpu().numpy(), -1.0, 1.0).astype(np.float32)
    def rollout(kind, R):
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
        return succ / ep
    print(f"\n---- mode={mode}  (match-acc {macc:.3f}, N={N_TASKS*SEEDS}) ----", flush=True)
    for kind, R, tag in variants:
        print(f"  {tag}: {rollout(kind, R):.3f}", flush=True)

SEG_VARIANTS = [("gcbc", 0, "GCBC floor"), ("null", 0, "null-u floor"), ("oracle", 0, "ORACLE (true et)"),
                ("m4", 0, "M4 R=0"), ("m4", 1, "M4 R=1"), ("m4", 3, "M4 R=3"), ("m4", 5, "M4 R=5")]
NAIVE_VARIANTS = [("gcbc", 0, "GCBC floor"), ("null", 0, "null-u floor"), ("m4", 3, "M4 R=3")]
print("\n==== ABLATION: naive vs segment data, full policy sweep, N=100 ====", flush=True)
train_and_rollout("segment", SEG_VARIANTS)
train_and_rollout("naive", NAIVE_VARIANTS)
print("\n=> compare: segment vs naive (data), ORACLE/M4 vs floors (u helps?), M4 R-curve (scaling).", flush=True)
