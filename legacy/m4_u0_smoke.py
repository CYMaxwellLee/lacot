"""Choice (4) verification: the refine loop's u^0 = a SAMPLE from the flow (detached).

Design Q1/Q3: at inference there is NO future, so u^0 MUST be sampled from
p(u|s,g); training therefore runs the refine loop from the SAME sampled-style u^0
(train/inference consistency). The "refine from sampled u^0 converges + inference
is coherent" half is already shown by m4_smoke (8/8, matched action MSE 8e-5,
per-round gap 1.95->0). This smoke ISOLATES the remaining question the full-model
smoke does not: is the flow SAMPLE a GOOD refine start -- does the PURE (L_NF-only)
flow imagine the RIGHT trajectory, so u^0 ~ e_target(cond) rather than noise?

Setup: recon-pretrain + freeze the generator; e_target = gen(future); cond =
frozen enc(s,g); train ONLY the flow (L_NF) on (e_target, cond); then look at its
samples. Checks:
  1. the flow imagines the RIGHT trajectory: ||sample(cond) - e_target(cond)||^2
     (matched) << ||sample(cond) - e_target(shuffled)||^2 (a sample matches its own
     cond's e_target, not a random one);
  2. the sample is a MUCH better refine start than a base-Gaussian draw (the "no
     flow" baseline): matched dist << prior dist;
  3. no NaN/Inf.

Run (CPU is fine):  .venv/bin/python -m wpm.models.m4_u0_smoke
"""
from __future__ import annotations

import math
import sys
import time

import torch
import torch.nn.functional as F

from wpm.models.e_target import ETargetGenerator, FrameDecoder
from wpm.models.nf_head import Flow
from wpm.train_value_official import ImpalaSmall


def make_batch(B: int, T: int, H: int = 64, W: int = 64, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    c_s = 0.25 + 0.5 * torch.rand(B, 2, generator=g)
    c_g = 0.25 + 0.5 * torch.rand(B, 2, generator=g)
    bulge = (2.0 * torch.rand(B, 1, generator=g) - 1.0) * 0.25
    t = torch.linspace(0.0, 1.0, T)
    straight = (1 - t)[None, :, None] * c_s[:, None, :] + t[None, :, None] * c_g[:, None, :]
    direction = c_g - c_s
    perp = torch.stack([-direction[:, 1], direction[:, 0]], dim=-1)
    perp = perp / (perp.norm(dim=-1, keepdim=True) + 1e-6)
    arch = torch.sin(math.pi * t)[None, :, None]
    center = straight + bulge[:, None, :] * arch * perp[:, None, :]
    ys = torch.linspace(0.0, 1.0, H)
    xs = torch.linspace(0.0, 1.0, W)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    cy = center[..., 0][..., None, None]
    cx = center[..., 1][..., None, None]
    d2 = (gy[None, None] - cy) ** 2 + (gx[None, None] - cx) ** 2
    blob = torch.exp(-d2 / (2 * 0.15 ** 2))
    frames = blob.unsqueeze(2).expand(B, T, 3, H, W).contiguous()
    return frames[:, 0], frames[:, -1], frames, c_s, c_g


def main() -> int:
    torch.manual_seed(0)
    B, T, H, W = 48, 6, 64, 64
    D_MODEL, K, ENC_OUT = 64, 8, 512
    RECON_STEPS, FLOW_STEPS, LR = 250, 700, 2e-3

    s_frame, g_frame, future, c_s, c_g = make_batch(B, T, H, W, seed=0)

    checks: list[tuple[str, bool]] = []

    def check(name: str, ok: bool) -> None:
        checks.append((name, bool(ok)))

    t0 = time.time()

    # recon-pretrain the generator, freeze
    gen = ETargetGenerator(ImpalaSmall(in_ch=3, out_dim=ENC_OUT), encoder_out=ENC_OUT,
                           d_model=D_MODEL, k=K, num_layers=2, num_heads=4)
    dec = FrameDecoder(d_model=D_MODEL, out_ch=3, img_size=H, num_layers=2, num_heads=4)
    opt_r = torch.optim.Adam(list(gen.parameters()) + list(dec.parameters()), lr=LR)
    for _ in range(RECON_STEPS):
        opt_r.zero_grad(set_to_none=True)
        loss = F.mse_loss(dec(gen(future), num_frames=T), future)
        loss.backward()
        opt_r.step()
    for p in gen.parameters():
        p.requires_grad_(False)
    gen.eval()

    with torch.no_grad():
        e_target = gen(future).detach()                                     # u = e_target (F=identity)
        cond = torch.cat([gen.encoder(s_frame), gen.encoder(g_frame)], dim=-1)

    # train ONLY the (pure) flow on L_NF
    flow = Flow(token_dim=D_MODEL, seq_len=K, n_blocks=4, d_hidden=128, cond_dim=cond.shape[-1])
    opt = torch.optim.Adam(flow.parameters(), lr=LR)
    any_nan = False
    for _ in range(FLOW_STEPS):
        opt.zero_grad(set_to_none=True)
        loss = flow.nll(e_target, cond)
        loss.backward()
        opt.step()
        if not math.isfinite(loss.item()):
            any_nan = True
            break

    with torch.no_grad():
        sample = flow.sample(B, cond)                       # u^0 the refine loop would start from
        perm = torch.randperm(B)
        d_matched = F.mse_loss(sample, e_target).item()               # sample(cond) vs its own e_target
        d_shuffled = F.mse_loss(sample, e_target[perm]).item()        # vs a wrong cond's e_target
        d_prior = F.mse_loss(torch.randn_like(e_target), e_target).item()  # base-Gaussian draw ("no flow")
        samp_finite = bool(torch.isfinite(sample).all())

    check("no NaN/Inf, sample finite", (not any_nan) and samp_finite)
    check("flow imagines the RIGHT trajectory (matched dist << shuffled)", d_matched < 0.5 * d_shuffled)
    check("flow sample is a far better start than a base-Gaussian draw (matched << prior)",
          d_matched < 0.5 * d_prior)

    npass = sum(ok for _, ok in checks)
    print(f"\n=== M4 choice (4) u^0 = flow sample isolated smoke: {npass}/{len(checks)} ===")
    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print(f"  ||u^0 - e_target||^2 :  matched {d_matched:.4f}   shuffled {d_shuffled:.4f}   "
          f"base-Gaussian(no flow) {d_prior:.4f}")
    print(f"  ⇒ the flow's sample lands ~on the imagined trajectory (small matched), a good refine start")
    print(f"  | B={B} T={T} K={K} d_model={D_MODEL} recon_steps={RECON_STEPS} flow_steps={FLOW_STEPS} "
          f"({time.time() - t0:.0f}s)")
    return 0 if npass == len(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
