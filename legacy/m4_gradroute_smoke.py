"""Choice (3) verification: gradient routing under F=identity.

Verifies that with F=identity + a FROZEN generator, EACH loss's gradient reaches
exactly the intended parameter groups and NOWHERE else (design Q2: "freeze the
target side, train the flow/policy side", specialized to F=identity):

  L_NF                              -> flow ONLY (density trained by NLL on e_target)
  L_action (anchor, frozen u_target) -> action head ONLY
  L_action (refine, deep supervision) -> refine op + action head (u^0 is a NO-GRAD
                                         sample, so the gradient does NOT reach the flow)
  L_consistency                     -> refine op ONLY
  the generator (incl. its encoder) -> NEVER receives gradient (frozen)

STRUCTURAL test: it inspects WHICH params get gradients, which is independent of
weight quality, so the generator is frozen FRESH (no reconstruction pretraining)
for speed. It asserts routing, not loss values.

Run (CPU is fine):  .venv/bin/python -m wpm.models.m4_gradroute_smoke
"""
from __future__ import annotations

import sys

import torch

from wpm.models.e_target import ETargetGenerator
from wpm.models.m4 import M4Model
from wpm.train_value_official import ImpalaSmall


def group_grad_norms(model: M4Model) -> dict[str, float]:
    def gn(params) -> float:
        total = 0.0
        for p in params:
            if p.grad is not None:
                total += float(p.grad.norm().item()) ** 2
        return total ** 0.5
    return {
        "gen": gn(model.generator.parameters()),
        "flow": gn(model.flow.parameters()),
        "refine": gn(model.refine.parameters()),
        "head": gn(model.action_head.parameters()),
    }


def routed(model: M4Model, loss: torch.Tensor) -> dict[str, bool]:
    model.zero_grad(set_to_none=True)
    loss.backward()
    return {k: v > 1e-9 for k, v in group_grad_norms(model).items()}


def main() -> int:
    torch.manual_seed(0)
    B, T, H, W = 8, 4, 64, 64
    D_MODEL, K, ENC_OUT = 32, 4, 128
    ACTION_DIM, CHUNK_LEN, NUM_BINS = 2, 3, 32
    ROUNDS = 3

    frames = torch.rand(B, T, 3, H, W)          # random frames are fine (structural test)
    s_frame, g_frame, future = frames[:, 0], frames[:, -1], frames
    actions = torch.rand(B, CHUNK_LEN, ACTION_DIM) * 2 - 1

    gen = ETargetGenerator(ImpalaSmall(in_ch=3, out_dim=ENC_OUT), encoder_out=ENC_OUT,
                           d_model=D_MODEL, k=K, num_layers=2, num_heads=2)
    for p in gen.parameters():
        p.requires_grad_(False)                  # FROZEN (fresh; routing independent of training)
    model = M4Model(gen, encoder_out=ENC_OUT, d_model=D_MODEL, k=K,
                    action_dim=ACTION_DIM, chunk_len=CHUNK_LEN, num_bins=NUM_BINS,
                    n_flow_blocks=3, refine_hidden=64)

    cond = model.encode_cond(s_frame, g_frame)   # frozen encoder
    u_target = model.e_target(future)            # frozen; F=identity -> u=e_target
    b = B

    # each loss in isolation
    r_nf = routed(model, model.flow.nll(u_target, cond))
    r_anchor = routed(model, model.action_head.nll(
        model.action_head(u_target.reshape(b, -1)), actions).mean())

    u0 = model.flow.sample(b, cond).detach()
    us = model.refine_rounds(cond, u0, ROUNDS)
    l_ref = us[0].new_zeros(())
    for r in range(ROUNDS):
        l_ref = l_ref + model.action_head.nll(
            model.action_head(us[r + 1].reshape(b, -1)), actions).mean()
    r_refine = routed(model, l_ref)

    u0c = model.flow.sample(b, cond).detach()
    usc = model.refine_rounds(cond, u0c, ROUNDS)
    l_cons = usc[0].new_zeros(())
    for r in range(ROUNDS):
        l_cons = l_cons + (usc[r] - usc[r + 1].detach()).pow(2).mean()
    r_cons = routed(model, l_cons)

    checks: list[tuple[str, bool]] = []

    def check(name: str, ok: bool) -> None:
        checks.append((name, bool(ok)))

    check("L_NF routes to flow ONLY", r_nf == {"gen": False, "flow": True, "refine": False, "head": False})
    check("L_action(anchor) routes to action head ONLY",
          r_anchor == {"gen": False, "flow": False, "refine": False, "head": True})
    check("L_action(refine) routes to refine + head ONLY (NOT flow)",
          r_refine == {"gen": False, "flow": False, "refine": True, "head": True})
    check("L_consistency routes to refine ONLY",
          r_cons == {"gen": False, "flow": False, "refine": True, "head": False})
    check("generator NEVER receives gradient (frozen) in any loss",
          not any(r["gen"] for r in (r_nf, r_anchor, r_refine, r_cons)))

    npass = sum(ok for _, ok in checks)
    print(f"\n=== M4 choice (3) gradient-routing (F=identity) smoke: {npass}/{len(checks)} ===")
    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print("  routing table (True = gradient reaches this group):")
    print(f"    L_NF            {r_nf}")
    print(f"    L_action anchor {r_anchor}")
    print(f"    L_action refine {r_refine}")
    print(f"    L_consistency   {r_cons}")
    return 0 if npass == len(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
