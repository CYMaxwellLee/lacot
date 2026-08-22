"""Diagnose the oracle-37%: isolate the OOD variable (per 主人 '排除各種變因').
Same (s, g, act); feed head TWO kinds of GT u:
  * dataset e_target  = encode(the DATASET future trajectory s->g)   [in-distribution]
  * expert  e_target  = encode(the EXPERT BFS path s->g)             [what the oracle used]
If MSE_expert >> MSE_dataset, the oracle rollout was OOD-confounded (a tunable/fixable
variable), not proof the design is limited. Also reports null (u=0) for reference.
Lean: trains only the e_target encoder + cond + head (no flow/refine), open-loop MSE.
"""
import os, sys, numpy as np, torch
from torch import nn
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # m4 repo root
from lacot.e_target import PerceiverPooler
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
B, T_CAP, D_MODEL, K, GEOM_P, TEMP, COND, CHUNK, ADIM = 64, 16, 256, 64, 0.02, 0.1, 256, 4, 2

def make_batch(rng, want_raw=False):
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
    out = [T(traj), torch.from_numpy(mask).to(device), T(s), T(g), T(act)]
    if want_raw:
        out += [OBS[rows], OBS[goals]]
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
        self.net = sota_mlp(COND + K * D_MODEL, 512, CHUNK * ADIM, n=3)
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
print("stage 1 contrastive ...", flush=True)
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
print(f"  match-acc {(logits.argmax(1)==lab).float().mean().item():.3f}", flush=True)

cond_enc = sota_mlp(2, 512, 512).to(device); cond_head = sota_mlp(1024, 512, COND).to(device)
ahead = ActionMLP().to(device)
mods = [cond_enc, cond_head, ahead]
opt2 = torch.optim.Adam([p for m in mods for p in m.parameters()], lr=5e-4)
def condvec(s, g):
    return cond_head(torch.cat([cond_enc(s), cond_enc(g)], 1))
mse = lambda p, a: (p - a).pow(2).mean()
ZERO = torch.zeros(B, K, D_MODEL, device=device)
print("stage 2 head (anchor + null) ...", flush=True)
for stp in range(1500):
    traj, mask, s, g, act = make_batch(rng)
    with torch.no_grad():
        et = etarget(traj, mask)
    cond = condvec(s, g)
    loss = mse(ahead(cond, et), act) + mse(ahead(cond, ZERO), act)
    opt2.zero_grad(set_to_none=True); loss.backward(); opt2.step()
for m in mods:
    m.eval()

os.environ.setdefault("OGBENCH_DATA_DIR", OGB_DATA)
env, _, _ = ogbench.make_env_and_datasets(ENV_NAME)
GAIN = 5.0
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
@torch.no_grad()
def expert_et_batch(raw_s, raw_g):
    ets = []
    for i in range(len(raw_s)):
        poss = expert_positions(raw_s[i], raw_g[i])
        idx = np.unique(np.linspace(0, len(poss) - 1, min(T_CAP, len(poss))).round().astype(int))
        tt = torch.tensor(((poss[idx] - mu) / sd).astype(np.float32), device=device)[None]
        ets.append(e_pooler(traj_enc(tt.reshape(-1, 2)).reshape(1, -1, 512), key_padding_mask=None))
    return torch.cat(ets, 0)

print("\n=== DIAGNOSIS: OOD isolation (dataset-et vs expert-et, same (s,g,act)) ===", flush=True)
md, me, mn, cosee = [], [], [], []
with torch.no_grad():
    for _ in range(5):
        traj, mask, s, g, act, raw_s, raw_g = make_batch(np.random.default_rng(1000 + _), want_raw=True)
        et_data = etarget(traj, mask)
        et_exp = expert_et_batch(raw_s, raw_g)
        cond = condvec(s, g)
        md.append(mse(ahead(cond, et_data), act).item())
        me.append(mse(ahead(cond, et_exp), act).item())
        mn.append(mse(ahead(cond, ZERO[:s.shape[0]]), act).item())
        cosee.append(F.cosine_similarity(et_data.reshape(B, -1), et_exp.reshape(B, -1)).mean().item())
base = 0.485
print(f"  MSE  head(cond, DATASET e_target)  {np.mean(md):.4f}   [in-distribution GT u]", flush=True)
print(f"  MSE  head(cond, EXPERT  e_target)  {np.mean(me):.4f}   [what the ORACLE rollout used]", flush=True)
print(f"  MSE  head(cond, ZERO u)            {np.mean(mn):.4f}   [null / (s,g)-only]", flush=True)
print(f"  predict-mean baseline              {base:.4f}", flush=True)
print(f"  cos(dataset_et, expert_et)         {np.mean(cosee):.3f}   [1=same, low=OOD]", flush=True)
print("=> expert_et MSE >> dataset_et MSE  => oracle-37% was OOD-confounded (fixable), not a design ceiling.", flush=True)
