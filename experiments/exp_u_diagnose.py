"""U-DIAGNOSIS (per 主人: why is our u not trained well?). Clean PyTorch, ported M4
model (m4model/, no wpm), official raw .npz. oracle=100% proves head+true-et works,
so the ONLY gap is flow-sampled u != true e_target. Pinpoint WHICH failure:
  (1) flow distribution bad  -> true_et has bad NLL / conditioning doesn't work
  (2) high-dim sampling spread-> a sample lands in the typical SET, not the mode
       (cosine(sample,true_et) low at T=1, RISES as temperature drops)
  (3) refine doesn't pull u back to true_et
  (4) head too sensitive to u error -> tiny u error -> big action error
"""
import os
import sys, numpy as np, torch
from torch import nn
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # m4 repo root
from lacot.nf_head import Flow
from lacot.e_target import PerceiverPooler
from lacot.model import RefineOperator

# 資料位置：預設走官方 OGBENCH_DATA_DIR，沒設才用本機 archive
OGB_DATA = os.environ.get("OGBENCH_DATA_DIR", "/archive/cymaxwelllee/data/ogbench")

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device, flush=True)
ENV = "pointmaze-medium-navigate-v0"
d = np.load(f"{OGB_DATA}/{ENV}.npz")  # official raw data
OBS = np.asarray(d["observations"], np.float32); ACT = np.asarray(d["actions"], np.float32); TERM = np.asarray(d["terminals"], bool)
N = OBS.shape[0]; ends = np.flatnonzero(TERM); starts = np.concatenate([[0], ends[:-1] + 1])
traj_end = np.empty(N, np.int64)
for s0, e0 in zip(starts, ends):
    traj_end[s0:e0 + 1] = e0
mu, sd = OBS.mean(0), OBS.std(0) + 1e-6
B, T_CAP, D_MODEL, K, GEOM_P, TEMP, COND, CHUNK, ADIM = 64, 16, 256, 64, 0.02, 0.1, 256, 4, 2
DIM = K * D_MODEL
WANDER_MAX = 3.0  # my own clean relabel (no wpm): geometric goal, drop wandering paths

def make_batch(rng):
    rows, goals = [], []
    while len(rows) < B:
        r = int(rng.integers(0, N)); te = int(traj_end[r])
        if te - r < CHUNK:
            continue
        gr = min(r + int(rng.geometric(GEOM_P)), te)
        if gr - r < CHUNK:
            continue
        path = OBS[r:gr + 1]
        if np.linalg.norm(np.diff(path, axis=0), axis=1).sum() / (np.linalg.norm(path[-1] - path[0]) + 1e-6) > WANDER_MAX:
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
        super().__init__(); self.net = sota_mlp(COND + DIM, 512, CHUNK * ADIM, n=3)
    def forward(self, cond, u):
        return self.net(torch.cat([cond, u.reshape(u.shape[0], -1)], -1)).reshape(-1, CHUNK, ADIM)

def flow_sample_T(flow, cond, temp):
    z = temp * torch.randn(cond.shape[0], flow.seq_len, flow.token_dim, device=device)
    u = z
    for i in reversed(range(len(flow.blocks))):
        if i < len(flow.blocks) - 1:
            u = flow.perm.inverse(u)
        u = flow.blocks[i].inverse(u, cond)
    return u

def slotcos(a, b):  # mean per-slot cosine (K slots), the alignment metric
    return F.cosine_similarity(a, b, dim=-1).mean(dim=-1).mean().item()

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
print(f"  match-acc {(logits.argmax(1)==lab).float().mean().item():.3f}", flush=True)

cond_enc = sota_mlp(2, 512, 512).to(device); cond_head = sota_mlp(1024, 512, COND).to(device)
flow = Flow(token_dim=D_MODEL, seq_len=K, n_blocks=6, cond_dim=COND).to(device)
refine = RefineOperator(COND, K, D_MODEL, hidden=256).to(device)
ahead = ActionMLP().to(device)
f_mods = [cond_enc, cond_head, flow, refine, ahead]
opt2 = torch.optim.Adam([p for m in f_mods for p in m.parameters()], lr=5e-4)
def condvec(s, g):
    return cond_head(torch.cat([cond_enc(s), cond_enc(g)], 1))
mse = lambda p, a: (p - a).pow(2).mean()
print("stage 2 flow+refine+action ...", flush=True)
for stp in range(2500):
    traj, mask, s, g, act = make_batch(rng)
    with torch.no_grad():
        et = etarget(traj, mask)
    cond = condvec(s, g)
    l_nf = flow.nll(et, cond) / DIM
    l_anchor = mse(ahead(cond, et), act)
    u = flow.sample(B, cond).detach(); us = [u]
    for _ in range(3):
        u = refine(cond, u); us.append(u)
    l_refine = sum(mse(ahead(cond, us[r + 1]), act) for r in range(3)) / 3
    l_cons = sum((us[r] - us[r + 1].detach()).pow(2).mean() for r in range(3)) / 3
    total = l_nf + l_anchor + l_refine + 0.5 * l_cons
    opt2.zero_grad(set_to_none=True); total.backward()
    torch.nn.utils.clip_grad_norm_([p for m in f_mods for p in m.parameters()], 1.0); opt2.step()
    if (stp + 1) % 2000 == 0:
        print(f"  step {stp+1}  l_nf/dim {l_nf.item():.3f} l_anchor {l_anchor.item():.4f}", flush=True)
for m in f_mods:
    m.eval()

print("\n==== U-DIAGNOSIS ====", flush=True)
with torch.no_grad():
    traj, mask, s, g, act = make_batch(np.random.default_rng(9999))
    et = etarget(traj, mask); cond = condvec(s, g)
    # wrong cond (shuffle) for conditioning test
    perm = torch.randperm(B, device=device); cond_wrong = cond[perm]
    # (1) flow distribution: NLL of true et under correct vs wrong cond; NLL of a sample
    nll_true = (flow.nll(et, cond) / DIM).item()
    nll_wrong = (flow.nll(et, cond_wrong) / DIM).item()
    samp = flow.sample(B, cond)
    nll_samp = (flow.nll(samp, cond) / DIM).item()
    print(f"(1) flow density  NLL/dim: true_et|cond {nll_true:.3f}   true_et|WRONG-cond {nll_wrong:.3f}   sample|cond {nll_samp:.3f}", flush=True)
    print(f"    => conditioning works if true|cond < true|wrong. true_et higher-density (lower NLL) than sample? {nll_true < nll_samp}", flush=True)
    # (2) cosine(sample, true_et) vs temperature  + random floor
    rnd = torch.randn_like(et)
    print(f"(2) slot-cosine(u, true_et):  random-floor {slotcos(rnd, et):.3f}", flush=True)
    for tmp in (1.0, 0.7, 0.5, 0.3, 0.1):
        print(f"      temp {tmp}: {slotcos(flow_sample_T(flow, cond, tmp), et):.3f}", flush=True)
    # (3) refine PER ROUND: action-MSE (the REAL metric) + raw vs CENTERED cosine.
    #  raw cosine is confounded by refine's LayerNorm removing the per-slot mean;
    #  centered-cos removes that mean from BOTH -> fair alignment measure.
    def ccos(a, b):
        return F.cosine_similarity(a - a.mean(-1, keepdim=True), b - b.mean(-1, keepdim=True), dim=-1).mean().item()
    et_c_ref = et  # true et for comparison
    u = flow.sample(B, cond)
    print(f"(3) refine/round:  R0 act-MSE {mse(ahead(cond,u),act).item():.4f}  raw-cos {slotcos(u,et_c_ref):.3f}  centered-cos {ccos(u,et_c_ref):.3f}", flush=True)
    for r in range(1, 6):
        u = refine(cond, u)
        print(f"      R{r} act-MSE {mse(ahead(cond,u),act).item():.4f}  raw-cos {slotcos(u,et_c_ref):.3f}  centered-cos {ccos(u,et_c_ref):.3f}", flush=True)
    # (4) head sensitivity: action MSE from true_et vs sampled u vs refined u vs et+noise
    base = (act.mean(0, keepdim=True) - act).pow(2).mean().item()
    err_true = mse(ahead(cond, et), act).item()
    u1 = flow.sample(B, cond); err_s = mse(ahead(cond, u1), act).item()
    ulow = flow_sample_T(flow, cond, 0.3); err_low = mse(ahead(cond, ulow), act).item()
    ur = u1.clone()
    for _ in range(3):
        ur = refine(cond, ur)
    err_ref = mse(ahead(cond, ur), act).item()
    for nz in (0.1, 0.3, 0.5):
        en = mse(ahead(cond, et + nz * torch.randn_like(et)), act).item()
        print(f"(4) head sensitivity: action MSE  true_et+noise({nz}) {en:.4f}", flush=True)
    print(f"(4) action MSE:  base(mean) {base:.4f}  true_et {err_true:.4f}  sample {err_s:.4f}  low-temp {err_low:.4f}  refined {err_ref:.4f}", flush=True)
print("=> reading: temp-cosine RISES = typical-set spread (fix: low temp/refine). NLL bad = flow undertrained.", flush=True)
