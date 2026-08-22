"""M4 connect the flow (option a): stage-1 contrastive-pretrain e_target -> FREEZE;
stage-2 train the TARFlow to model p(u|s,g) with u=e_target. Checks:
  - NLL/dim drops
  - NLL with WRONG (shuffled) cond is worse  => conditioning is really used
  - sampled u | (s,g) is close to the true e_target (mean slot cosine)
"""
import os
import sys, numpy as np, torch
from torch import nn
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # m4 repo root
from lacot.e_target import PerceiverPooler
from lacot.nf_head import Flow

# 資料位置：預設走官方 OGBENCH_DATA_DIR，沒設才用本機 archive
OGB_DATA = os.environ.get("OGBENCH_DATA_DIR", "/archive/cymaxwelllee/data/ogbench")

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device, flush=True)
d = np.load(f"{OGB_DATA}/pointmaze-medium-navigate-v0.npz")
OBS = np.asarray(d["observations"], np.float32); TERM = np.asarray(d["terminals"], bool)
N = OBS.shape[0]; ends = np.flatnonzero(TERM); starts = np.concatenate([[0], ends[:-1] + 1])
traj_end = np.empty(N, np.int64)
for s0, e0 in zip(starts, ends):
    traj_end[s0:e0 + 1] = e0
mu, sd = OBS.mean(0), OBS.std(0) + 1e-6
B, T_CAP, D_MODEL, K, GEOM_P, TEMP, COND = 64, 16, 256, 64, 0.02, 0.1, 256

def make_batch(rng):
    rows, goals = [], []
    while len(rows) < B:
        r = int(rng.integers(0, N)); te = int(traj_end[r])
        if te - r < 2:
            continue
        gr = min(r + int(rng.geometric(GEOM_P)), te)
        if gr > r:
            rows.append(r); goals.append(gr)
    rows, goals = np.array(rows), np.array(goals)
    idxs = [np.unique(np.linspace(rows[i], goals[i], min(T_CAP, goals[i] - rows[i] + 1)).round().astype(int)) for i in range(B)]
    Tmax = max(len(ix) for ix in idxs)
    traj = np.zeros((B, Tmax, 2), np.float32); mask = np.ones((B, Tmax), bool)
    for i, ix in enumerate(idxs):
        traj[i, :len(ix)] = (OBS[ix] - mu) / sd; mask[i, :len(ix)] = False
    s = (OBS[rows] - mu) / sd; g = (OBS[goals] - mu) / sd
    return (torch.from_numpy(traj).to(device), torch.from_numpy(mask).to(device),
            torch.from_numpy(s.astype(np.float32)).to(device), torch.from_numpy(g.astype(np.float32)).to(device))

def sota_mlp(i, h, o, n=2):
    L, p = [], i
    for _ in range(n):
        lin = nn.Linear(p, h); nn.init.xavier_uniform_(lin.weight); nn.init.zeros_(lin.bias)
        L += [lin, nn.GELU(), nn.LayerNorm(h)]; p = h
    lin = nn.Linear(p, o); nn.init.xavier_uniform_(lin.weight); nn.init.zeros_(lin.bias)
    return nn.Sequential(*L, lin)

torch.manual_seed(0); rng = np.random.default_rng(0)

# ---------- Stage 1: contrastive-pretrain e_target ----------
traj_enc = sota_mlp(2, 512, 512).to(device); e_pooler = PerceiverPooler(512, D_MODEL, K, 2, 4).to(device)
sg_c = sota_mlp(2, 512, 512).to(device); q_pooler = PerceiverPooler(512, D_MODEL, K, 2, 4).to(device)
c_mods = [traj_enc, e_pooler, sg_c, q_pooler]
opt1 = torch.optim.Adam([p for m in c_mods for p in m.parameters()], lr=1e-3)
lab = torch.arange(B, device=device)
def etarget(traj, mask):
    Bc, Tc, _ = traj.shape
    return e_pooler(traj_enc(traj.reshape(Bc * Tc, 2)).reshape(Bc, Tc, 512), key_padding_mask=mask)
print("stage 1: contrastive pretrain e_target ...", flush=True)
for stp in range(800):
    traj, mask, s, g = make_batch(rng)
    et = etarget(traj, mask); q = q_pooler(torch.stack([sg_c(s), sg_c(g)], 1))
    logits = (F.normalize(q.reshape(B, -1), dim=1) @ F.normalize(et.reshape(B, -1), dim=1).t()) / TEMP
    loss = 0.5 * (F.cross_entropy(logits, lab) + F.cross_entropy(logits.t(), lab))
    opt1.zero_grad(set_to_none=True); loss.backward(); opt1.step()
    if (stp + 1) % 200 == 0:
        acc = (logits.argmax(1) == lab).float().mean().item()
        print(f"  c-step {stp+1} loss {loss.item():.3f} match-acc {acc:.3f}", flush=True)
for m in (traj_enc, e_pooler):
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)

# ---------- Stage 2: flow p(u | s,g), u = frozen e_target ----------
cond_enc = sota_mlp(2, 512, 512).to(device); cond_head = sota_mlp(1024, 512, COND).to(device)
flow = Flow(token_dim=D_MODEL, seq_len=K, n_blocks=4, d_hidden=256, n_layers=2, n_heads=4, cond_dim=COND).to(device)
f_mods = [cond_enc, cond_head, flow]
opt2 = torch.optim.Adam([p for m in f_mods for p in m.parameters()], lr=5e-4)
DIM = K * D_MODEL
def condvec(s, g):
    return cond_head(torch.cat([cond_enc(s), cond_enc(g)], 1))
print("stage 2: train flow L_NF ...", flush=True)
for stp in range(2000):
    traj, mask, s, g = make_batch(rng)
    with torch.no_grad():
        et = etarget(traj, mask)
    loss = flow.nll(et, condvec(s, g)) / DIM      # nats per dim
    opt2.zero_grad(set_to_none=True); loss.backward()
    torch.nn.utils.clip_grad_norm_([p for m in f_mods for p in m.parameters()], 1.0)
    opt2.step()
    if (stp + 1) % 200 == 0:
        print(f"  f-step {stp+1}  NLL/dim {loss.item():.4f}", flush=True)

# ---------- checks ----------
for m in f_mods:
    m.eval()
with torch.no_grad():
    traj, mask, s, g = make_batch(np.random.default_rng(9999))
    et = etarget(traj, mask); cond = condvec(s, g)
    nll_ok = (flow.nll(et, cond) / DIM).item()
    nll_bad = (flow.nll(et, cond[torch.randperm(B, device=device)]) / DIM).item()  # WRONG cond
    u = flow.sample(B, cond)                              # sample u | (s,g)
    cos = F.cosine_similarity(F.normalize(u, dim=-1), F.normalize(et, dim=-1), dim=-1).mean().item()
    cos_rand = F.cosine_similarity(F.normalize(u, dim=-1), F.normalize(et[torch.randperm(B, device=device)], dim=-1), dim=-1).mean().item()
print("\n==== FLOW CONNECT RESULTS ====", flush=True)
print(f"NLL/dim  correct-cond {nll_ok:.4f}  |  WRONG-cond {nll_bad:.4f}   (wrong should be worse => conditioning used)", flush=True)
print(f"sampled u vs TRUE e_target: slot-cosine {cos:.3f}  |  vs random e_target {cos_rand:.3f}   (higher-for-true => flow generates the right u)", flush=True)
