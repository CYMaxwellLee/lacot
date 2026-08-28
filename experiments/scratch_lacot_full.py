"""LaCoT full loop on STATE: e_target (contrastive, frozen) -> flow -> refine -> action head.
Three losses (l_nf + l_act_anchor + l_act_refine + lam*l_cons), like LaCoTActor.losses_given.
Checks: action error from the TRUE e_target (ceiling) vs from the REFINED sampled u (inference);
u convergence (l_cons); TEST-TIME SCALING = action error vs number of refine rounds.
"""
import os
import sys, numpy as np, torch
from torch import nn
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # lacot repo root
from lacot.e_target import PerceiverPooler
from lacot.nf_head import Flow
from lacot.model import RefineOperator
from lacot.heads import DiscretizedActionHead

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
    act = np.stack([ACT[r:r + CHUNK] for r in rows]).astype(np.float32)  # [B,CHUNK,2] in [-1,1]
    T = lambda x: torch.from_numpy(x).to(device)
    return T(traj), T(mask), T(s.astype(np.float32)), T(g.astype(np.float32)), T(act)

def sota_mlp(i, h, o, n=2):
    L, p = [], i
    for _ in range(n):
        lin = nn.Linear(p, h); nn.init.xavier_uniform_(lin.weight); nn.init.zeros_(lin.bias)
        L += [lin, nn.GELU(), nn.LayerNorm(h)]; p = h
    lin = nn.Linear(p, o); nn.init.xavier_uniform_(lin.weight); nn.init.zeros_(lin.bias)
    return nn.Sequential(*L, lin)

torch.manual_seed(0); rng = np.random.default_rng(0)
# ---- stage 1: contrastive e_target ----
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

# ---- stage 2: flow + refine + action head ----
cond_enc = sota_mlp(2, 512, 512).to(device); cond_head = sota_mlp(1024, 512, COND).to(device)
flow = Flow(token_dim=D_MODEL, seq_len=K, n_blocks=4, cond_dim=COND).to(device)
refine = RefineOperator(COND, K, D_MODEL, hidden=256).to(device)
ahead = DiscretizedActionHead(DIM, ADIM, CHUNK, 256).to(device)
f_mods = [cond_enc, cond_head, flow, refine, ahead]
opt2 = torch.optim.Adam([p for m in f_mods for p in m.parameters()], lr=5e-4)
def condvec(s, g):
    return cond_head(torch.cat([cond_enc(s), cond_enc(g)], 1))
def act_err(logits, act):  # greedy decode -> continuous -> MSE vs true
    return (ahead.decode(logits, "greedy") - act).pow(2).mean().item()

print("stage 2 flow+refine+action ...", flush=True)
for stp in range(2000):
    traj, mask, s, g, act = make_batch(rng)
    with torch.no_grad():
        et = etarget(traj, mask)
    cond = condvec(s, g)
    l_nf = flow.nll(et, cond) / DIM
    l_anchor = ahead.nll(ahead(et.reshape(B, -1)), act).mean()
    u = flow.sample(B, cond).detach(); us = [u]
    for _ in range(3):
        u = refine(cond, u); us.append(u)
    l_cons = sum((us[r] - us[r + 1].detach()).pow(2).mean() for r in range(3)) / 3
    l_refine = sum(ahead.nll(ahead(us[r + 1].reshape(B, -1)), act).mean() for r in range(3)) / 3
    total = l_nf + l_anchor + l_refine + 0.5 * l_cons
    opt2.zero_grad(set_to_none=True); total.backward()
    torch.nn.utils.clip_grad_norm_([p for m in f_mods for p in m.parameters()], 1.0); opt2.step()
    if (stp + 1) % 500 == 0:
        print(f"  step {stp+1}  l_nf/dim {l_nf.item():.3f} l_anchor {l_anchor.item():.3f} l_refine {l_refine.item():.3f} l_cons {l_cons.item():.4f}", flush=True)

# ---- checks ----
for m in f_mods:
    m.eval()
with torch.no_grad():
    traj, mask, s, g, act = make_batch(np.random.default_rng(9999))
    et = etarget(traj, mask); cond = condvec(s, g)
    err_anchor = act_err(ahead(et.reshape(B, -1)), act)                       # ceiling: from true e_target
    acc_anchor = ahead.accuracy(ahead(et.reshape(B, -1)), act).mean().item()
    # test-time scaling: action error vs refine rounds (inference path: sample u0 -> refine)
    print("\n==== LaCoT FULL RESULTS ====", flush=True)
    print(f"ANCHOR (action from TRUE e_target): MSE {err_anchor:.4f}  bin-acc {acc_anchor:.3f}   (ceiling)", flush=True)
    base = ((act.mean(0, keepdim=True) - act).pow(2).mean().item())
    print(f"baseline (predict-mean-action) MSE {base:.4f}", flush=True)
    for R in (0, 1, 3, 5, 8):
        u = flow.sample(B, cond)
        for _ in range(R):
            u = refine(cond, u)
        e = act_err(ahead(u.reshape(B, -1)), act)
        print(f"  refine rounds {R}:  action MSE {e:.4f}", flush=True)
    print("=> test-time scaling works if MSE drops as rounds increase.", flush=True)
