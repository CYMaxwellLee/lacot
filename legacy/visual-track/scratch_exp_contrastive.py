"""M4: does a CONTRASTIVE objective (SOTA/CRL-style, no reconstruction) make e_target
ENCODE the trajectory without collapsing? (s,g) query <-> future-trajectory e_target key,
InfoNCE with in-batch negatives. Same real data as the reconstruction run, for contrast.

Metrics:
  match-acc     : can (s,g) pick its own future out of the batch? (chance = 1/B) -> e_target encodes
  key-offdiag   : mean cosine sim between different trajectories' e_targets -> LOW = distinct = NOT collapsed
Run on jasmine GPU:
  MUJOCO_GL=egl OGBENCH_DATA_DIR=... WPM_CACHE_DIR=... .venv/bin/python -u exp_etarget_contrastive.py
"""
import os, sys, time
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

sys.path.insert(0, "/archive/cymaxwelllee/fpo")
import wpm.data.pipeline as P
from wpm.models.e_target import ETargetGenerator
from wpm.train_value_official import ImpalaSmall

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device, "| torch", torch.__version__, flush=True)

DS = "visual-pointmaze-medium-navigate-v0"
cfg = P.PipelineConfig(dataset_name=DS)
pipe = P.WpmPipeline(cfg, split="val")
OBS = pipe.arrays["observations"]

T_CAP, B, D, D_MODEL, K = 16, 32, 128, 256, 64

def make_batch(rng):
    b = pipe.sample_batch(rng, B)
    trajs, lens = [], []
    for i in range(B):
        r0, r1 = int(b.row[i]), int(b.goal_row[i])
        L = r1 - r0 + 1
        idx = np.unique(np.linspace(r0, r1, min(T_CAP, L)).round().astype(int))
        trajs.append(np.asarray(OBS[idx])); lens.append(len(idx))
    Tmax = max(lens)
    fut = np.zeros((B, Tmax, 64, 64, 3), np.float32)
    mask = np.ones((B, Tmax), bool)
    for i, t in enumerate(trajs):
        Li = t.shape[0]; fut[i, :Li] = t.astype(np.float32) / 255.0; mask[i, :Li] = False
    fut = np.transpose(fut, (0, 1, 4, 2, 3))
    s = np.transpose(np.asarray(b.current).astype(np.float32) / 255.0, (0, 3, 1, 2))
    g = np.transpose(np.asarray(b.goal).astype(np.float32) / 255.0, (0, 3, 1, 2))
    return (torch.from_numpy(fut).to(device), torch.from_numpy(mask).to(device),
            torch.from_numpy(s).to(device), torch.from_numpy(g).to(device), lens)

sg_enc = ImpalaSmall(in_ch=3, out_dim=512).to(device)
gen = ETargetGenerator(ImpalaSmall(in_ch=3, out_dim=512), encoder_out=512,
                       d_model=D_MODEL, k=K, num_layers=2, num_heads=4).to(device)
q_head = nn.Sequential(nn.Linear(1024, 512), nn.GELU(), nn.Linear(512, D)).to(device)
k_head = nn.Sequential(nn.Linear(D_MODEL, D), nn.GELU(), nn.Linear(D, D)).to(device)
params = (list(sg_enc.parameters()) + list(gen.parameters())
          + list(q_head.parameters()) + list(k_head.parameters()))
print(f"params {sum(p.numel() for p in params)/1e6:.2f}M | B={B} K={K} d_model={D_MODEL} T_CAP={T_CAP}", flush=True)
opt = torch.optim.Adam(params, lr=1e-3)
TEMP = 0.1
eye = torch.eye(B, dtype=bool, device=device)

def run_step(rng):
    fut, mask, s, g, lens = make_batch(rng)
    q = q_head(torch.cat([sg_enc(s), sg_enc(g)], dim=1))
    et = gen(fut, key_padding_mask=mask)      # [B,K,D_MODEL]  COMPRESS
    k = k_head(et.mean(dim=1))                # [B,D]
    q = F.normalize(q, dim=1); k = F.normalize(k, dim=1)
    logits = (q @ k.t()) / TEMP
    labels = torch.arange(B, device=device)
    loss = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))
    acc = (logits.argmax(1) == labels).float().mean().item()
    offdiag = (k @ k.t())[~eye].mean().item()
    return loss, acc, offdiag

rng = np.random.default_rng(0)
print(f"chance match-acc = 1/{B} = {1/B:.3f}", flush=True)
t0 = time.time()
for stp in range(1000):
    loss, acc, off = run_step(rng)
    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    if (stp + 1) % 50 == 0:
        print(f"step {stp+1:4d}  loss {loss.item():.4f}  match-acc {acc:.3f}  key-offdiag-sim {off:+.3f}  ({time.time()-t0:.0f}s)", flush=True)

# final eval on a fresh batch
sg_enc.eval(); gen.eval(); q_head.eval(); k_head.eval()
accs, offs = [], []
with torch.no_grad():
    for _ in range(10):
        _, a, o = run_step(np.random.default_rng(1000 + _))
        accs.append(a); offs.append(o)
print(f"\nFINAL (fresh batches): match-acc {np.mean(accs):.3f} (chance {1/B:.3f})  |  key-offdiag-sim {np.mean(offs):+.3f}", flush=True)
print("interpret: match-acc >> chance AND key-offdiag low => e_target ENCODES distinct trajectories, NO collapse (unlike reconstruction).", flush=True)
