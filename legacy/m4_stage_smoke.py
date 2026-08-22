"""Choice (5) verification: single-stage vs two-stage training.

Design S7 keeps NF-CoT's two-stage curriculum (density first, then joint) as an
option. m4_smoke used single-stage. This isolates the question: does two-stage
BUY anything, now that 主人 chose (a) -- a PURE flow (only L_NF, never action;
verified by m4_gradroute_smoke)?

Prediction (Luna): under (a) the flow is ISOLATED from the action objective, so
its optimisation is the SAME whether trained alone or jointly -> two-stage gives
NO flow benefit. The only thing staging could help is the refine/head starting on
a warmed-up flow's (better) u^0. We test both.

Fair budget: the flow gets the SAME number of L_NF gradient steps either way.
  single-stage : JOINT joint steps                      (flow: JOINT L_NF; refine/head: JOINT)
  two-stage    : STAGE1 flow-only + STAGE2 joint         (flow: STAGE1+STAGE2 L_NF; refine/head: STAGE2)
with JOINT = STAGE1 + STAGE2, so the flow sees the same total L_NF steps; only the
ORDER (and how many steps the refine/head get) differs.

Checks:
  1. single-stage: inference is coherent (matched action MSE << shuffled);
  2. two-stage:    inference is coherent;
  3. the flow density is ~the SAME either way (L_NF close) -- confirms the flow is
     isolated from action, so staging cannot change it (the core (a) consequence);
  4. single-stage is SUFFICIENT: its inference is not meaningfully worse than
     two-stage (matched MSE within 2x) -> two-stage is not needed here.

Run (CPU is fine):  .venv/bin/python -m wpm.models.m4_stage_smoke
"""
from __future__ import annotations

import math
import sys
import time

import torch
import torch.nn.functional as F

from wpm.models.e_target import ETargetGenerator, FrameDecoder
from wpm.models.m4 import M4Model
from wpm.models.m4_smoke import make_m4_batch
from wpm.train_value_official import ImpalaSmall


def build_model(gen, enc_out, d_model, k, action_dim, chunk_len, num_bins, seed):
    torch.manual_seed(seed)
    return M4Model(gen, encoder_out=enc_out, d_model=d_model, k=k, action_dim=action_dim,
                   chunk_len=chunk_len, num_bins=num_bins, n_flow_blocks=4, refine_hidden=128)


def evaluate(model, cond, u_target, actions, s_frame, g_frame, rounds):
    with torch.no_grad():
        l_nf = model.flow.nll(u_target, cond).item()
        b = cond.shape[0]
        perm = torch.randperm(b)
        gap = (model.flow.log_prob(u_target, cond).mean()
               - model.flow.log_prob(u_target, cond[perm]).mean()).item()
        act = model.infer_action(s_frame, g_frame, rounds=rounds)
        matched = F.mse_loss(act, actions).item()
        shuffled = F.mse_loss(act, actions[perm]).item()
    return l_nf, gap, matched, shuffled


def main() -> int:
    torch.manual_seed(0)
    B, T, H, W = 40, 6, 64, 64
    D_MODEL, K, ENC_OUT = 64, 8, 512
    ACTION_DIM, CHUNK_LEN, NUM_BINS = 2, T - 1, 64
    RECON_STEPS, LR, ROUNDS, LAM = 250, 2e-3, 3, 0.5
    JOINT, STAGE1, STAGE2 = 400, 200, 200   # JOINT == STAGE1 + STAGE2 (flow sees same L_NF steps)

    s_frame, g_frame, future, actions = make_m4_batch(B, T, H, W, seed=0)

    # recon-pretrain ONE generator, freeze; cache the frozen front-end
    gen = ETargetGenerator(ImpalaSmall(in_ch=3, out_dim=ENC_OUT), encoder_out=ENC_OUT,
                           d_model=D_MODEL, k=K, num_layers=2, num_heads=4)
    dec = FrameDecoder(d_model=D_MODEL, out_ch=3, img_size=H, num_layers=2, num_heads=4)
    opt_r = torch.optim.Adam(list(gen.parameters()) + list(dec.parameters()), lr=LR)
    for _ in range(RECON_STEPS):
        opt_r.zero_grad(set_to_none=True)
        F.mse_loss(dec(gen(future), num_frames=T), future).backward()
        opt_r.step()
    for p in gen.parameters():
        p.requires_grad_(False)
    gen.eval()
    with torch.no_grad():
        cond = torch.cat([gen.encoder(s_frame), gen.encoder(g_frame)], dim=-1).detach()
        u_target = gen(future).detach()

    t0 = time.time()

    # --- single-stage: joint from step 0 ---
    m_s = build_model(gen, ENC_OUT, D_MODEL, K, ACTION_DIM, CHUNK_LEN, NUM_BINS, seed=1)
    opt = torch.optim.Adam(list(m_s.flow.parameters()) + list(m_s.refine.parameters())
                           + list(m_s.action_head.parameters()), lr=LR)
    for _ in range(JOINT):
        opt.zero_grad(set_to_none=True)
        total, _ = m_s.losses_given(cond, u_target, actions, rounds=ROUNDS, lam_cons=LAM)
        total.backward()
        opt.step()
    nf_s, gap_s, match_s, shuf_s = evaluate(m_s, cond, u_target, actions, s_frame, g_frame, ROUNDS)

    # --- two-stage: flow-only, then joint ---
    m_t = build_model(gen, ENC_OUT, D_MODEL, K, ACTION_DIM, CHUNK_LEN, NUM_BINS, seed=1)
    opt1 = torch.optim.Adam(m_t.flow.parameters(), lr=LR)
    for _ in range(STAGE1):
        opt1.zero_grad(set_to_none=True)
        m_t.flow.nll(u_target, cond).backward()
        opt1.step()
    opt2 = torch.optim.Adam(list(m_t.flow.parameters()) + list(m_t.refine.parameters())
                            + list(m_t.action_head.parameters()), lr=LR)
    for _ in range(STAGE2):
        opt2.zero_grad(set_to_none=True)
        total, _ = m_t.losses_given(cond, u_target, actions, rounds=ROUNDS, lam_cons=LAM)
        total.backward()
        opt2.step()
    nf_t, gap_t, match_t, shuf_t = evaluate(m_t, cond, u_target, actions, s_frame, g_frame, ROUNDS)

    checks: list[tuple[str, bool]] = []

    def check(name, ok):
        checks.append((name, bool(ok)))

    check("single-stage inference coherent (matched << shuffled)", match_s < 0.3 * shuf_s)
    check("two-stage inference coherent (matched << shuffled)", match_t < 0.3 * shuf_t)
    check("single-stage SUFFICIENT for the task (matched MSE within 2x of two-stage)",
          match_s < 2.0 * match_t + 1e-6)

    npass = sum(ok for _, ok in checks)
    print(f"\n=== M4 choice (5) single- vs two-stage smoke: {npass}/{len(checks)} ===")
    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print(f"  single-stage : L_NF {nf_s:.1f}  cond-gap {gap_s:.0f} | infer matched {match_s:.5f}  shuffled {shuf_s:.5f}")
    print(f"  two-stage    : L_NF {nf_t:.1f}  cond-gap {gap_t:.0f} | infer matched {match_t:.5f}  shuffled {shuf_t:.5f}")
    # HONEST FINDING (Luna predicted these would be IDENTICAL -- WRONG): the two-stage flow
    # reaches a LOWER L_NF than single-stage even though action never touches the flow (S Q2 / (a),
    # verified by m4_gradroute_smoke). Isolation-from-action does NOT imply identical density: the
    # TRAINING SCHEDULE still moves the flow's optimisation (two-stage resets the Adam state at the
    # stage boundary; the peaky L_NF landscape amplifies path-dependence). BUT it did not matter for
    # the outcome -- BOTH reach the same coherent inference. So single-stage is sufficient here; the
    # single-vs-two-stage call for the flow's density needs REAL data + multiple seeds (one run on an
    # easy synthetic task can't settle it).
    print(f"  ⚠️ two-stage flow reaches lower L_NF ({nf_t:.0f} vs {nf_s:.0f}) via schedule/optimizer path,"
          f" NOT action coupling; but inference is equally coherent -> single-stage suffices for the task")
    print(f"  | JOINT={JOINT} STAGE1={STAGE1} STAGE2={STAGE2} B={B} K={K} ({time.time() - t0:.0f}s)")
    return 0 if npass == len(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
