"""M4 feasibility: can we COMPRESS a real variable-length s->g trajectory into
e_target [B,K,D] and DECODE it back? Reconstruction pretraining on REAL rendered
visual-pointmaze trajectories (different step counts, padded + masked).

Uses the existing blocks (ETargetGenerator + FrameDecoder), not a rebuild.
Run on jasmine GPU:
  MUJOCO_GL=egl OGBENCH_DATA_DIR=... WPM_CACHE_DIR=... .venv/bin/python -u exp_etarget_recon_real.py
"""
import os, sys, time, math
import numpy as np
import torch

sys.path.insert(0, "/archive/cymaxwelllee/fpo")
import wpm.data.pipeline as P
from wpm.models.e_target import ETargetGenerator, FrameDecoder
from wpm.train_value_official import ImpalaSmall

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device, "| torch", torch.__version__, flush=True)

DS = "visual-pointmaze-medium-navigate-v0"
cfg = P.PipelineConfig(dataset_name=DS)
pipe = P.WpmPipeline(cfg, split="val")
OBS = pipe.arrays["observations"]  # (N,64,64,3) uint8
print("obs array:", OBS.shape, OBS.dtype, flush=True)

T_CAP = 16   # subsample each trajectory to <= this many frames (keeps variable length)
B = 16

def make_real_batch(rng):
    b = pipe.sample_batch(rng, B)
    trajs, lens = [], []
    for i in range(B):
        r0, r1 = int(b.row[i]), int(b.goal_row[i])
        L = r1 - r0 + 1
        idx = np.unique(np.linspace(r0, r1, min(T_CAP, L)).round().astype(int))
        trajs.append(np.asarray(OBS[idx]))     # (Li,64,64,3) uint8
        lens.append(len(idx))
    Tmax = max(lens)
    frames = np.zeros((B, Tmax, 64, 64, 3), np.float32)
    mask = np.ones((B, Tmax), dtype=bool)      # True == PAD
    for i, t in enumerate(trajs):
        Li = t.shape[0]
        frames[i, :Li] = t.astype(np.float32) / 255.0
        mask[i, :Li] = False
    frames = np.transpose(frames, (0, 1, 4, 2, 3))  # -> (B,Tmax,3,64,64)
    return (torch.from_numpy(frames).to(device),
            torch.from_numpy(mask).to(device), lens)

def masked_mse(recon, frames, mask):
    valid = (~mask).float()[:, :, None, None, None]  # [B,T,1,1,1]
    se = ((recon - frames) ** 2) * valid
    return se.sum() / (valid.sum() * frames.shape[2] * frames.shape[3] * frames.shape[4] + 1e-8)

D_MODEL, K = 256, 64
gen = ETargetGenerator(ImpalaSmall(in_ch=3, out_dim=512), encoder_out=512,
                       d_model=D_MODEL, k=K, num_layers=2, num_heads=4).to(device)
dec = FrameDecoder(d_model=D_MODEL, out_ch=3, img_size=64, num_layers=2, num_heads=4).to(device)
nparam = sum(p.numel() for p in list(gen.parameters()) + list(dec.parameters()))
print(f"model params: {nparam/1e6:.2f}M | K={K} d_model={D_MODEL} T_CAP={T_CAP} B={B}", flush=True)

rng = np.random.default_rng(0)
# --- shape round-trip on a real variable-length batch ---
frames, mask, lens = make_real_batch(rng)
Tmax = frames.shape[1]
with torch.no_grad():
    e = gen(frames, key_padding_mask=mask)
    recon = dec(e, num_frames=Tmax)
    init_loss = masked_mse(recon, frames, mask).item()
    mean_frame = frames.mean(dim=(0, 1), keepdim=True)
    baseline = masked_mse(mean_frame.expand_as(frames), frames, mask).item()
print(f"COMPRESS ok: e_target {tuple(e.shape)} (want [{B},{K},{D_MODEL}]) | trajectory lens {lens}", flush=True)
print(f"DECODE ok:   recon {tuple(recon.shape)} == frames {tuple(frames.shape)}", flush=True)
print(f"init masked-MSE {init_loss:.5f} | mean-frame baseline {baseline:.5f}", flush=True)

opt = torch.optim.Adam(list(gen.parameters()) + list(dec.parameters()), lr=2e-3)
STEPS = 800
t0 = time.time()
final = init_loss
for step in range(STEPS):
    frames, mask, lens = make_real_batch(rng)
    Tmax = frames.shape[1]
    e = gen(frames, key_padding_mask=mask)      # COMPRESS
    recon = dec(e, num_frames=Tmax)             # DECODE
    loss = masked_mse(recon, frames, mask)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    final = loss.item()
    if not math.isfinite(final):
        print("NaN/Inf -> abort", flush=True); sys.exit(1)
    if (step + 1) % 50 == 0:
        print(f"step {step+1:4d}  masked-MSE {final:.5f}  ({(time.time()-t0):.0f}s)", flush=True)

print(f"\nDONE: masked-MSE init {init_loss:.5f} -> final {final:.5f} "
      f"(mean-frame baseline {baseline:.5f}); dropped {init_loss/max(final,1e-9):.1f}x", flush=True)

# --- save input vs reconstruction for a few samples ---
from PIL import Image
gen.eval(); dec.eval()
frames, mask, lens = make_real_batch(np.random.default_rng(7))
Tmax = frames.shape[1]
with torch.no_grad():
    recon = dec(gen(frames, key_padding_mask=mask), num_frames=Tmax).clamp(0, 1)
fr = (frames.cpu().numpy().transpose(0, 1, 3, 4, 2) * 255).astype(np.uint8)   # B,T,H,W,3
rc = (recon.cpu().numpy().transpose(0, 1, 3, 4, 2) * 255).astype(np.uint8)
K_UP = 3
pad = 6
outdir = "/archive/cymaxwelllee/fpo/outputs/sg_demo"
os.makedirs(outdir, exist_ok=True)
rows = []
for i in range(min(3, B)):
    Li = lens[i]
    ncol = min(Li, 8)
    idxs = np.unique(np.linspace(0, Li - 1, ncol).round().astype(int))
    def strip(arr):
        imgs = [np.kron(arr[i, j], np.ones((K_UP, K_UP, 1), np.uint8)) for j in idxs]
        H, W = imgs[0].shape[:2]
        canvas = np.full((H, len(imgs) * W + (len(imgs) - 1) * pad, 3), 245, np.uint8)
        x = 0
        for im in imgs:
            canvas[:, x:x + W] = im; x += W + pad
        return canvas
    top, bot = strip(fr), strip(rc)
    gap = np.full((pad, top.shape[1], 3), 245, np.uint8)
    rows.append(np.vstack([top, gap, bot]))
    rows.append(np.full((pad * 2, top.shape[1], 3), 255, np.uint8))
maxw = max(r.shape[1] for r in rows)
rows = [np.pad(r, ((0, 0), (0, maxw - r.shape[1]), (0, 0)), constant_values=255) for r in rows]
Image.fromarray(np.vstack(rows)).save(f"{outdir}/recon_real.png")
print(f"SAVED {outdir}/recon_real.png (each pair: TOP=original trajectory, BOTTOM=reconstruction)", flush=True)
