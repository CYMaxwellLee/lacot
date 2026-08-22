"""Full-model wiring smoke (build order S5 step 5): all three M4 losses train
together, gradients route correctly, and the inference path runs coherently.

Composes M4Model (wpm/models/m4.py) from the individually-validated blocks and
checks the JOINT behaviour that the per-block smokes cannot:
  Phase 1  reconstruction-pretrain the e_target generator (+ throwaway decoder),
           then FREEZE it (its encoder is reused, frozen, for the (s,g) cond).
  Phase 2  build M4Model on the frozen generator.
  Phase 3  joint-train L_NF + L_action(anchor + deep-refine) + L_consistency.

Controlled synthetic (s, g, future, actions): a blob travels a smooth path from
start centre c_s (= s frame) to goal centre c_g (= g frame); the future frames
are the whole path; the actions are the per-step blob displacement (the motion).
So (s,g) -> cond, the path -> e_target, and the motion -> actions are all
coherent -- the model can learn (s,g) -> u -> action end to end.

Checks (careful -- 主人 "謹慎的推進"):
  1. reconstruction honest before freeze (MSE beats the mean-frame baseline);
  2. the FULL training_losses(frames...) path runs, is finite, and backprops;
  3. the generator (incl. its encoder) stays FROZEN through joint training;
  4. the trainable params (flow / refine / action head) actually change;
  5. no NaN/Inf during joint training;
  6. all the driven losses drop: L_NF down, and both action terms (anchor from the
     clean target-u, and the deep-supervised refine rounds) down a lot;
  7. INFERENCE path (design Q1: NO future, NO e_target) runs: (s,g) -> sample
     u^0 -> refine -> decode -> action chunk of the right shape, finite;
  8. inference is COHERENT: the decoded action for the matched (s,g) is closer to
     that episode's true action than to a shuffled one (the whole pipeline learned
     a (s,g)->action map, not a constant).

NOTE (scope + FLAGGED design choices): a WIRING smoke on ONE fixed synthetic batch.
The design choices baked into M4Model (shared frozen encoder for cond; F=identity;
sampled u^0; single-stage training; F=identity gradient routing) are documented at
the top of wpm/models/m4.py and are for 主人 to confirm/redirect. Two-stage
training (S7) and real OGBench hindsight data are the next steps.

Run (CPU is fine):  .venv/bin/python -m wpm.models.m4_smoke
"""
from __future__ import annotations

import math
import sys
import time

import torch
import torch.nn.functional as F

from wpm.models.e_target import ETargetGenerator, FrameDecoder
from wpm.models.m4 import M4Model
from wpm.train_value_official import ImpalaSmall


def make_m4_batch(B: int, T: int, H: int = 64, W: int = 64, seed: int = 0):
    """Blob path -> (s_frame, g_frame, future_frames, actions). See module docstring."""
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
    center = straight + bulge[:, None, :] * arch * perp[:, None, :]  # [B,T,2]
    ys = torch.linspace(0.0, 1.0, H)
    xs = torch.linspace(0.0, 1.0, W)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    cy = center[..., 0][..., None, None]
    cx = center[..., 1][..., None, None]
    d2 = (gy[None, None] - cy) ** 2 + (gx[None, None] - cx) ** 2
    blob = torch.exp(-d2 / (2 * 0.15 ** 2))
    frames = blob.unsqueeze(2).expand(B, T, 3, H, W).contiguous()  # [B,T,3,H,W]
    s_frame = frames[:, 0]        # blob at start
    g_frame = frames[:, -1]       # blob at goal
    future_frames = frames        # whole path (e_target compresses it)
    disp = center[:, 1:] - center[:, :-1]                       # [B,T-1,2] per-step motion
    actions = (disp / (disp.abs().amax() + 1e-6)).clamp(-1.0, 1.0)  # scaled to [-1,1]
    return s_frame, g_frame, future_frames, actions


def _flat(params) -> torch.Tensor:
    return torch.cat([p.reshape(-1) for p in params])


def main() -> int:
    torch.manual_seed(0)
    B, T, H, W = 40, 6, 64, 64
    D_MODEL, K, ENC_OUT = 64, 8, 512
    ACTION_DIM, CHUNK_LEN, NUM_BINS = 2, T - 1, 64
    RECON_STEPS, JOINT_STEPS, LR = 250, 400, 2e-3
    ROUNDS, LAM = 3, 0.5

    s_frame, g_frame, future_frames, actions = make_m4_batch(B, T, H, W, seed=0)

    checks: list[tuple[str, bool]] = []

    def check(name: str, ok: bool) -> None:
        checks.append((name, bool(ok)))

    any_nan = False
    t0 = time.time()

    # === Phase 1: reconstruction-pretrain the generator (honest target), then freeze ===
    gen = ETargetGenerator(
        ImpalaSmall(in_ch=3, out_dim=ENC_OUT), encoder_out=ENC_OUT,
        d_model=D_MODEL, k=K, num_layers=2, num_heads=4,
    )
    dec = FrameDecoder(d_model=D_MODEL, out_ch=3, img_size=H, num_layers=2, num_heads=4)
    with torch.no_grad():
        init_recon = F.mse_loss(dec(gen(future_frames), num_frames=T), future_frames).item()
        mean_frame = future_frames.mean(dim=(0, 1), keepdim=True)
        baseline_recon = F.mse_loss(mean_frame.expand_as(future_frames), future_frames).item()
    opt_r = torch.optim.Adam(list(gen.parameters()) + list(dec.parameters()), lr=LR)
    final_recon = init_recon
    for step in range(RECON_STEPS):
        opt_r.zero_grad(set_to_none=True)
        loss = F.mse_loss(dec(gen(future_frames), num_frames=T), future_frames)
        loss.backward()
        opt_r.step()
        final_recon = loss.item()
        if not math.isfinite(final_recon):
            any_nan = True
            break
        if (step + 1) % 50 == 0:
            print(f"  [recon] step {step + 1:4d}  MSE {final_recon:.4f}  (baseline {baseline_recon:.4f})")
    check("reconstruction honest before freeze (beats mean-frame baseline)",
          math.isfinite(final_recon) and final_recon < baseline_recon)
    gen.eval()
    for p in gen.parameters():
        p.requires_grad_(False)

    # === Phase 2: build the full model ===
    model = M4Model(gen, encoder_out=ENC_OUT, d_model=D_MODEL, k=K,
                    action_dim=ACTION_DIM, chunk_len=CHUNK_LEN, num_bins=NUM_BINS,
                    n_flow_blocks=4, refine_hidden=128)

    # verify the FULL raw-frames path runs + backprops + is finite
    total0, comps0 = model.training_losses(s_frame, g_frame, future_frames, actions,
                                           rounds=ROUNDS, lam_cons=LAM)
    total0.backward()
    check("full training_losses(frames) runs, finite, backprops",
          math.isfinite(total0.item()) and all(math.isfinite(v) for v in comps0.values()))
    model.zero_grad(set_to_none=True)

    trainable = (list(model.flow.parameters()) + list(model.refine.parameters())
                 + list(model.action_head.parameters()))
    gen_snapshot = _flat(model.generator.parameters()).clone()
    trainable_snapshot = _flat(trainable).clone()

    # === Phase 3: joint-train (cache the frozen front-end for speed) ===
    with torch.no_grad():
        cond = model.encode_cond(s_frame, g_frame).detach()      # frozen encoder
        u_target = model.e_target(future_frames).detach()        # frozen; F=identity
    opt = torch.optim.Adam(trainable, lr=LR)
    init_comps = final_comps = None
    for step in range(JOINT_STEPS):
        opt.zero_grad(set_to_none=True)
        total, comps = model.losses_given(cond, u_target, actions, rounds=ROUNDS, lam_cons=LAM)
        total.backward()
        opt.step()
        if init_comps is None:
            init_comps = comps
        final_comps = comps
        if not math.isfinite(total.item()):
            any_nan = True
            break
        if (step + 1) % 100 == 0:
            print(f"  [joint] step {step + 1:4d}  L_NF {comps['l_nf']:.1f}  "
                  f"L_act(anchor {comps['l_act_anchor']:.3f} / refine {comps['l_act_refine']:.3f})  "
                  f"L_cons {comps['l_cons']:.4f}")

    gen_now = _flat(model.generator.parameters())
    trainable_now = _flat(trainable)

    with torch.no_grad():
        act_infer = model.infer_action(s_frame, g_frame, rounds=ROUNDS, mode="greedy")
        infer_mse = F.mse_loss(act_infer, actions).item()
        perm = torch.randperm(B)
        infer_mse_shuffled = F.mse_loss(act_infer, actions[perm]).item()
        # diagnostic: per-round refine gap (is a high L_cons the u^0->u^1 first jump, or genuine drift?)
        u0_d = model.flow.sample(B, cond)
        us_d = model.refine_rounds(cond, u0_d, 6)
        per_round_gap = [(us_d[r + 1] - us_d[r]).pow(2).mean().item() for r in range(6)]

    check("generator (incl. encoder) stayed frozen", torch.equal(gen_now, gen_snapshot))
    check("trainable params (flow/refine/head) actually changed",
          not torch.equal(trainable_now, trainable_snapshot))
    check("no NaN/Inf during joint training", not any_nan)
    check("all driven losses dropped (L_NF down; both action terms < 0.5x init)",
          final_comps["l_nf"] < init_comps["l_nf"]
          and final_comps["l_act_anchor"] < 0.5 * init_comps["l_act_anchor"]
          and final_comps["l_act_refine"] < 0.5 * init_comps["l_act_refine"])
    check("inference path runs: action chunk shape + finite",
          tuple(act_infer.shape) == (B, CHUNK_LEN, ACTION_DIM) and bool(torch.isfinite(act_infer).all()))
    check("inference is coherent (matched action MSE < shuffled)",
          infer_mse < infer_mse_shuffled)

    npass = sum(ok for _, ok in checks)
    print(f"\n=== M4 full-model wiring smoke: {npass}/{len(checks)} ===")
    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print(f"  recon MSE: init {init_recon:.4f} -> final {final_recon:.4f}  (baseline {baseline_recon:.4f})")
    print(f"  L_NF:          init {init_comps['l_nf']:.1f} -> final {final_comps['l_nf']:.1f}")
    print(f"  L_act anchor:  init {init_comps['l_act_anchor']:.3f} -> final {final_comps['l_act_anchor']:.3f}")
    print(f"  L_act refine:  init {init_comps['l_act_refine']:.3f} -> final {final_comps['l_act_refine']:.3f}")
    print(f"  L_cons:        init {init_comps['l_cons']:.4f} -> final {final_comps['l_cons']:.4f}")
    print(f"  inference action MSE: matched {infer_mse:.5f}  shuffled {infer_mse_shuffled:.5f}")
    print(f"  per-round refine gap (u^0..u^6): {['%.4f' % x for x in per_round_gap]}  "
          f"(big first + small rest = converges after the sample->u^1 jump)")
    print(f"  | B={B} T={T} K={K} d_model={D_MODEL} action={ACTION_DIM}x{CHUNK_LEN} bins={NUM_BINS} "
          f"rounds={ROUNDS} recon_steps={RECON_STEPS} joint_steps={JOINT_STEPS} ({time.time() - t0:.0f}s)")
    return 0 if npass == len(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
