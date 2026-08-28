"""LaCoT state e_target, v2: NO pooling of the K slots -- compare the full [K,d] vectors
DIRECTLY (flatten K*d, normalize, dot). Mirrors how u ([K,d]) aligns to e_target ([K,d]).
query (s,g)->K slots ; e_target trajectory->K slots ; InfoNCE over the full structure.
"""
import os, sys, time
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # lacot repo root
from lacot.e_target import PerceiverPooler

# 資料位置：預設走官方 OGBENCH_DATA_DIR，沒設才用本機 archive
OGB_DATA = os.environ.get("OGBENCH_DATA_DIR", "/archive/cymaxwelllee/data/ogbench")

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device, flush=True)

DATA = f"{OGB_DATA}/pointmaze-medium-navigate-v0.npz"
d = np.load(DATA)
OBS = np.asarray(d["observations"], np.float32)
TERM = np.asarray(d["terminals"], bool)
N = OBS.shape[0]
ends = np.flatnonzero(TERM); starts = np.concatenate([[0], ends[:-1] + 1])
traj_end = np.empty(N, np.int64)
for s0, e0 in zip(starts, ends):
    traj_end[s0:e0 + 1] = e0
mu, sd = OBS.mean(0), OBS.std(0) + 1e-6

B, T_CAP, D_MODEL, K, GEOM_P = 64, 16, 256, 64, 0.02

def make_batch(rng):
    rows, goals = [], []
    while len(rows) < B:
        r = int(rng.integers(0, N)); te = int(traj_end[r])
        if te - r < 2:
            continue
        gr = min(r + int(rng.geometric(GEOM_P)), te)
        if gr <= r:
            continue
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

def sota_mlp(inp, hidden, out, nlayers=2):
    layers, prev = [], inp
    for _ in range(nlayers):
        lin = nn.Linear(prev, hidden); nn.init.xavier_uniform_(lin.weight); nn.init.zeros_(lin.bias)
        layers += [lin, nn.GELU(), nn.LayerNorm(hidden)]; prev = hidden
    lin = nn.Linear(prev, out); nn.init.xavier_uniform_(lin.weight); nn.init.zeros_(lin.bias)
    layers.append(lin)
    return nn.Sequential(*layers)

# e_target side: encode each traj position -> K slots
traj_enc = sota_mlp(2, 512, 512, 2).to(device)
e_pooler = PerceiverPooler(d_in=512, d_model=D_MODEL, k=K, num_layers=2, num_heads=4).to(device)
# query side (s,g) -> K slots  (symmetric; this is the u-shaped conditioning)
sg_enc = sota_mlp(2, 512, 512, 2).to(device)
q_pooler = PerceiverPooler(d_in=512, d_model=D_MODEL, k=K, num_layers=2, num_heads=4).to(device)
mods = [traj_enc, e_pooler, sg_enc, q_pooler]
params = [p for m in mods for p in m.parameters()]
print(f"params {sum(p.numel() for p in params)/1e6:.2f}M | B={B} K={K} d_model={D_MODEL} (NO pool; compare full K*d)", flush=True)
opt = torch.optim.Adam(params, lr=1e-3)
TEMP = 0.1
eye = torch.eye(B, dtype=bool, device=device)

def run_step(rng):
    traj, mask, s, g = make_batch(rng)
    Bc, Tc, _ = traj.shape
    tf = traj_enc(traj.reshape(Bc * Tc, 2)).reshape(Bc, Tc, 512)
    et = e_pooler(tf, key_padding_mask=mask)          # [B,K,D_MODEL]   e_target (NOT pooled)
    sgf = torch.stack([sg_enc(s), sg_enc(g)], dim=1)  # [B,2,512]  s,g as two tokens
    q = q_pooler(sgf)                                 # [B,K,D_MODEL]   query slots (u-shaped)
    ef = F.normalize(et.reshape(Bc, K * D_MODEL), dim=1)   # DIRECT: flatten all K slots
    qf = F.normalize(q.reshape(Bc, K * D_MODEL), dim=1)
    logits = (qf @ ef.t()) / TEMP
    labels = torch.arange(Bc, device=device)
    loss = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))
    acc = (logits.argmax(1) == labels).float().mean().item()
    off = (ef @ ef.t())[~eye].mean().item()           # e_target diversity (low = distinct)
    return loss, acc, off

rng = np.random.default_rng(0)
print(f"chance match-acc = 1/{B} = {1/B:.3f}", flush=True)
t0 = time.time()
for stp in range(1000):
    loss, acc, off = run_step(rng)
    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    if (stp + 1) % 50 == 0:
        print(f"step {stp+1:4d}  loss {loss.item():.4f}  match-acc {acc:.3f}  etarget-offdiag-sim {off:+.3f}  ({time.time()-t0:.0f}s)", flush=True)

for m in mods:
    m.eval()
accs, offs = [], []
with torch.no_grad():
    for j in range(10):
        _, a, o = run_step(np.random.default_rng(1000 + j)); accs.append(a); offs.append(o)
print(f"\nFINAL (fresh): match-acc {np.mean(accs):.3f} (chance {1/B:.3f})  etarget-offdiag-sim {np.mean(offs):+.3f}", flush=True)
print("PASS if match-acc >> chance AND offdiag low => full K-slot e_target encodes distinct trajectories, no collapse, no pooling info loss.", flush=True)
