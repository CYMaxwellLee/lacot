"""Choice (2) isolated verification: cond = the FROZEN encoder's (s,g) features.

M4Model conditions the density on the reconstruction-pretrained, FROZEN encoder's
(s,g) features (design S11: the deployed encoder is the frozen one). m4_smoke
exercised this BUNDLED with everything else; here we ISOLATE it: only the density
+ F=identity, and we ask two focused questions.

  Q1: is the frozen-encoder cond USABLE -- does the density learn p(u|cond) and
      actually USE it (log p higher for the MATCHED (s,g) than a shuffled one)?
  Q2: does routing (s,g) through the FROZEN encoder LOSE conditioning power vs the
      ground-truth raw (s,g) centres? (train an identical density on each; compare
      the matched-vs-mismatched gap.)

Controlled synthetic: a blob travels from start centre c_s to goal centre c_g with
a mid-path bulge (the conditional variation NOT in (s,g)); s_frame/g_frame are the
blob at c_s/c_g. cond_enc = concat(enc(s_frame), enc(g_frame)) [frozen encoder];
cond_raw = concat(c_s, c_g) [ground-truth]. Checks:
  1. frozen-encoder cond: L_NF drops a lot (the density learns);
  2. frozen-encoder cond: the density USES it (matched log p > mismatched);
  3. frozen-encoder cond: sampling works (right shape, finite); no NaN/Inf;
  4. raw (s,g) cond also conditions (baseline sanity);
  5. the frozen-encoder cond does NOT lose most of the conditioning power -- its
     matched-vs-mismatched gap is a meaningful fraction of the raw cond's gap.

Run (CPU is fine):  .venv/bin/python -m wpm.models.m4_cond_smoke
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
    return frames[:, 0], frames[:, -1], frames, c_s, c_g  # s_frame, g_frame, future, c_s, c_g


def train_density(u_target, cond, k, d, steps, lr, seed=0):
    """Fresh Flow density on (u_target, cond); return (init_nll, final_nll, lp_matched, lp_mismatched, samp_ok, any_nan)."""
    torch.manual_seed(seed)
    b = cond.shape[0]
    flow = Flow(token_dim=d, seq_len=k, n_blocks=4, d_hidden=128, cond_dim=cond.shape[-1])
    with torch.no_grad():
        init_nll = flow.nll(u_target, cond).item()
    opt = torch.optim.Adam(flow.parameters(), lr=lr)
    any_nan = False
    final_nll = init_nll
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        loss = flow.nll(u_target, cond)
        loss.backward()
        opt.step()
        final_nll = loss.item()
        if not math.isfinite(final_nll):
            any_nan = True
            break
    with torch.no_grad():
        lp_matched = flow.log_prob(u_target, cond).mean().item()
        perm = torch.randperm(b)
        lp_mismatched = flow.log_prob(u_target, cond[perm]).mean().item()
        samp = flow.sample(b, cond)
        samp_ok = tuple(samp.shape) == (b, k, d) and bool(torch.isfinite(samp).all())
    return init_nll, final_nll, lp_matched, lp_mismatched, samp_ok, any_nan


def main() -> int:
    torch.manual_seed(0)
    B, T, H, W = 48, 6, 64, 64
    D_MODEL, K, ENC_OUT = 64, 8, 512
    RECON_STEPS, DENS_STEPS, LR = 250, 700, 2e-3

    s_frame, g_frame, future_frames, c_s, c_g = make_batch(B, T, H, W, seed=0)

    checks: list[tuple[str, bool]] = []

    def check(name: str, ok: bool) -> None:
        checks.append((name, bool(ok)))

    t0 = time.time()

    # recon-pretrain the generator (incl. encoder), then FREEZE
    gen = ETargetGenerator(
        ImpalaSmall(in_ch=3, out_dim=ENC_OUT), encoder_out=ENC_OUT,
        d_model=D_MODEL, k=K, num_layers=2, num_heads=4,
    )
    dec = FrameDecoder(d_model=D_MODEL, out_ch=3, img_size=H, num_layers=2, num_heads=4)
    with torch.no_grad():
        baseline_recon = F.mse_loss(
            future_frames.mean(dim=(0, 1), keepdim=True).expand_as(future_frames), future_frames).item()
    opt_r = torch.optim.Adam(list(gen.parameters()) + list(dec.parameters()), lr=LR)
    recon_nan = False
    final_recon = None
    for _ in range(RECON_STEPS):
        opt_r.zero_grad(set_to_none=True)
        loss = F.mse_loss(dec(gen(future_frames), num_frames=T), future_frames)
        loss.backward()
        opt_r.step()
        final_recon = loss.item()
        if not math.isfinite(final_recon):
            recon_nan = True
            break
    gen.eval()
    for p in gen.parameters():
        p.requires_grad_(False)

    with torch.no_grad():
        u_target = gen(future_frames).detach()                       # F=identity: u = e_target
        cond_enc = torch.cat([gen.encoder(s_frame), gen.encoder(g_frame)], dim=-1)  # FROZEN encoder cond
        cond_raw = torch.cat([c_s, c_g], dim=-1)                     # ground-truth (s,g)

    # density with the FROZEN-ENCODER cond
    i_e, f_e, m_e, mm_e, samp_e, nan_e = train_density(u_target, cond_enc, K, D_MODEL, DENS_STEPS, LR, seed=1)
    gap_e = m_e - mm_e
    # density with the RAW (s,g) cond (baseline)
    i_r, f_r, m_r, mm_r, samp_r, nan_r = train_density(u_target, cond_raw, K, D_MODEL, DENS_STEPS, LR, seed=1)
    gap_r = m_r - mm_r

    check("reconstruction honest before freeze (beats mean-frame baseline)",
          (not recon_nan) and final_recon is not None and final_recon < baseline_recon)
    check("frozen-encoder cond: L_NF dropped a lot (final < 0.5 x init)", f_e < 0.5 * i_e)
    check("frozen-encoder cond: density USES it (matched log p > mismatched)", m_e > mm_e + 1.0)
    check("frozen-encoder cond: sampling ok, no NaN/Inf", samp_e and not nan_e)
    check("raw (s,g) cond also conditions (baseline sanity)", (m_r > mm_r + 1.0) and not nan_r)
    check("frozen-encoder cond keeps meaningful conditioning power (gap_enc > 0.3 x gap_raw)",
          gap_e > 0.3 * gap_r and gap_e > 1.0)

    npass = sum(ok for _, ok in checks)
    print(f"\n=== M4 choice (2) cond=frozen-encoder isolated smoke: {npass}/{len(checks)} ===")
    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print(f"  recon final {final_recon:.4f} (baseline {baseline_recon:.4f})")
    print(f"  frozen-encoder cond [dim {cond_enc.shape[-1]}]: L_NF {i_e:.1f}->{f_e:.1f} | "
          f"matched {m_e:.1f} vs mismatched {mm_e:.1f}  (gap {gap_e:.1f})")
    print(f"  raw (s,g) cond      [dim {cond_raw.shape[-1]}]: L_NF {i_r:.1f}->{f_r:.1f} | "
          f"matched {m_r:.1f} vs mismatched {mm_r:.1f}  (gap {gap_r:.1f})")
    print(f"  | B={B} T={T} K={K} d_model={D_MODEL} enc_out={ENC_OUT} "
          f"recon_steps={RECON_STEPS} dens_steps={DENS_STEPS} ({time.time() - t0:.0f}s)")
    return 0 if npass == len(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
