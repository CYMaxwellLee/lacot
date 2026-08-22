"""3-seed sweep: is slot-wise cosine (2) really worse than flatten+cosine (1), or noise?
Both = direct K-slot comparison (no pool). Same eval batches for fairness. Reports mean +/- std.
"""
import os
import sys, numpy as np, torch
from torch import nn
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # m4 repo root
from lacot.e_target import PerceiverPooler

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
B, T_CAP, D_MODEL, K, GEOM_P, TEMP = 64, 16, 256, 64, 0.02, 0.1

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

def sim_fn(method, q, et):
    if method == "flatten":  # (1)
        return F.normalize(q.reshape(q.shape[0], -1), dim=1) @ F.normalize(et.reshape(et.shape[0], -1), dim=1).t()
    An, Bn = F.normalize(q, dim=-1), F.normalize(et, dim=-1)   # (2) slot-wise
    return torch.einsum("ikd,jkd->ij", An, Bn) / K

# fixed eval batches (same for every method/seed -> fair)
EVAL = [make_batch(np.random.default_rng(5000 + j)) for j in range(10)]

def train_eval(method, seed):
    torch.manual_seed(seed)
    te_, ep, sg, qp = sota_mlp(2, 512, 512).to(device), PerceiverPooler(512, D_MODEL, K, 2, 4).to(device), sota_mlp(2, 512, 512).to(device), PerceiverPooler(512, D_MODEL, K, 2, 4).to(device)
    mods = [te_, ep, sg, qp]
    opt = torch.optim.Adam([p for m in mods for p in m.parameters()], lr=1e-3)
    rng = np.random.default_rng(seed)
    lab = torch.arange(B, device=device)
    def fwd(traj, mask, s, g):
        Bc, Tc, _ = traj.shape
        et = ep(te_(traj.reshape(Bc * Tc, 2)).reshape(Bc, Tc, 512), key_padding_mask=mask)
        q = qp(torch.stack([sg(s), sg(g)], 1))
        return q, et
    for _ in range(1000):
        q, et = fwd(*make_batch(rng))
        logits = sim_fn(method, q, et) / TEMP
        loss = 0.5 * (F.cross_entropy(logits, lab) + F.cross_entropy(logits.t(), lab))
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    for m in mods:
        m.eval()
    accs = []
    with torch.no_grad():
        for eb in EVAL:
            q, et = fwd(*eb)
            logits = sim_fn(method, q, et) / TEMP
            accs.append((logits.argmax(1) == lab).float().mean().item())
    return float(np.mean(accs))

print(f"PerceiverPooler signature note: (d_in,d_model,k,num_layers,num_heads)", flush=True)
for method in ["flatten", "slotwise"]:
    accs = [train_eval(method, s) for s in (0, 1, 2)]
    print(f"{method:9s} (①flatten / ②slotwise): seeds {[round(a,3) for a in accs]}  mean {np.mean(accs):.3f} ± {np.std(accs):.3f}", flush=True)
