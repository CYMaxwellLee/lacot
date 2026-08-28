"""LaCoT full loop v2 (fix action head, per 主人): action head = deep MLP that takes
(s,g)-cond AND u  -> continuous 2D action chunk (MSE). Fixes: (1) MLP not 1 Linear,
(2) action sees (s,g) directly, (3) continuous output (drop 256 bins).
Checks: action MSE anchor(from true e_target) vs refined vs baseline; test-time scaling.
Also ablation: does u help? (head(cond, ZERO-u) vs head(cond, real u)).
"""
import os
import sys, numpy as np, torch
from torch import nn
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # lacot repo root
from lacot.e_target import PerceiverPooler
from lacot.nf_head import Flow
from lacot.model import RefineOperator

# 資料位置：預設走官方 OGBENCH_DATA_DIR，沒設才用本機 archive
OGB_DATA = os.environ.get("OGBENCH_DATA_DIR", "/archive/cymaxwelllee/data/ogbench")

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device, flush=True)
d = np.load(f"{OGB_DATA}/pointmaze-medium-navigate-v0.npz")
OBS = np.asarray(d["observations"], np.float32); ACT = np.asarray(d["actions"], np.float32); TERM = np.asarray(d["terminals"], bool)
N = OBS.shape[0]; ends = np.flatnonzero(TERM); starts = np.concatenate([[0], ends[:-1] + 1])
traj_end = np.empty(N, np.int64)
for s0, e0 in zip(starts, ends):
    traj_end[s0:e0 + 1] = e0
mu, sd = OBS.mean(0), OBS.std(0) + 1e-6
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

class ActionMLP(nn.Module):  # (s,g)-cond + u -> continuous action chunk
    def __init__(self):
        super().__init__()
        self.net = sota_mlp(COND + DIM, 512, CHUNK * ADIM, n=3)
    def forward(self, cond, u):
        x = torch.cat([cond, u.reshape(u.shape[0], -1)], -1)
        return self.net(x).reshape(-1, CHUNK, ADIM)

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
def condvec(s, g):
    return cond_head(torch.cat([cond_enc(s), cond_enc(g)], 1))
mse = lambda p, a: (p - a).pow(2).mean()

print("stage 2 flow+refine+action(MLP, sees s,g) ...", flush=True)
for stp in range(2000):
    traj, mask, s, g, act = make_batch(rng)
    with torch.no_grad():
        et = etarget(traj, mask)
    cond = condvec(s, g)
    l_nf = flow.nll(et, cond) / DIM
    l_anchor = mse(ahead(cond, et), act)
    u = flow.sample(B, cond).detach(); us = [u]
    for _ in range(3):
        u = refine(cond, u); us.append(u)
    l_cons = sum((us[r] - us[r + 1].detach()).pow(2).mean() for r in range(3)) / 3
    l_refine = sum(mse(ahead(cond, us[r + 1]), act) for r in range(3)) / 3
    total = l_nf + l_anchor + l_refine + 0.5 * l_cons
    opt2.zero_grad(set_to_none=True); total.backward()
    torch.nn.utils.clip_grad_norm_([p for m in f_mods for p in m.parameters()], 1.0); opt2.step()
    if (stp + 1) % 500 == 0:
        print(f"  step {stp+1}  l_nf/dim {l_nf.item():.3f} l_anchor(MSE) {l_anchor.item():.4f} l_refine {l_refine.item():.4f} l_cons {l_cons.item():.4f}", flush=True)

for m in f_mods:
    m.eval()
with torch.no_grad():
    traj, mask, s, g, act = make_batch(np.random.default_rng(9999))
    et = etarget(traj, mask); cond = condvec(s, g)
    base = (act.mean(0, keepdim=True) - act).pow(2).mean().item()
    err_anchor = mse(ahead(cond, et), act).item()
    zero = torch.zeros(B, K, D_MODEL, device=device)
    err_sg_only = mse(ahead(cond, zero), act).item()   # ablation: (s,g) + ZERO u  -> does u help?
    print("\n==== LaCoT v2 RESULTS ====", flush=True)
    print(f"baseline (predict-mean) MSE {base:.4f}", flush=True)
    print(f"ANCHOR head(s,g, TRUE e_target) MSE {err_anchor:.4f}   (ceiling)", flush=True)
    print(f"ablation head(s,g, ZERO-u)      MSE {err_sg_only:.4f}   (u helps if anchor < this)", flush=True)
    for R in (0, 1, 3, 5, 8):
        u = flow.sample(B, cond)
        for _ in range(R):
            u = refine(cond, u)
        print(f"  refine rounds {R}:  action MSE {mse(ahead(cond, u), act).item():.4f}", flush=True)
    print("=> want: anchor << baseline (action decodes); refined drops with rounds (test-time scaling).", flush=True)
