# LaCoT — Latent Chain-of-Thought Actor

Goal-conditioned control by **reasoning in a continuous latent space**, then
decoding that reasoning into actions. A normalizing flow generates a latent
"thought" trajectory `u` conditioned on the current state and goal; a refine
operator revises it; an action head decodes it into a control chunk.

Named 2026-08-23 (previously the internal codename "LaCoT").

## Status — what is proven and what is not

| Claim | Evidence | Verdict |
|---|---|---|
| Architecture (head + `e_target`) can solve the task | ORACLE run: head fed the *true* `e_target` reaches **100%** success | ✅ proven — equals the BFS expert ceiling |
| The eval itself is hard (not a trivial task) | a straight-line controller scores only **31%** — 69% of tasks need wall-following | ✅ calibrated |
| The flow *knows* good `u` | true-`e_target` NLL under the conditional (**-3.94**) beats sampled (**-3.34**) | ✅ conditioning is used |
| Sampling reaches good `u` | cosine(sample, true `e_target`) = **0.47** (T=1.0), **0.59** (T=0.1) | ❌ **the bottleneck** — samples land off the typical set |
| Refine improves actions | action error R0 **0.272** → R1 **0.257**, then plateaus | ⚠️ real but weak — one step only, no test-time scaling |
| Refine aligns `u` to true `e_target` | centered-cosine collapses R0 0.403 → R1 **0.006** | ❌ it moves `u` *orthogonal* to the true target |

**Bottom line:** the architecture is proven (oracle = 100%); end-to-end sits at
**R=0 → 79%** against that 100% ceiling. The gap is entirely in the `u` the flow
produces. Refine helps a little but cannot reach the ceiling (refined 0.257 vs
true-`e_target` floor 0.225) and does not scale with more steps.

**Next lead:** `value-directed refine` — a trained GCIVL value
(`checkpoints/scratch_value.pt`, hindsight-advantage sign agreement 0.89) gives
refine a *direction* to climb, which is exactly what it currently lacks.

## Ground rules (set by 主人, 2026-08-22)

- **Everything benchmarks against official OGBench** (`seohongpark/ogbench`
  `impls/`: GCDataset, GCBC, eval). Raw data + eval protocol are the fairness
  requirement; relabel / chunking / normalization / network / loss are the
  method's own — every SOTA method relabels differently.
- ⛔ **No `wpm`, no `fpo`.** Not as an import, not as a path, not as a shell.
  Verified by `smoke_lacot.py` check ④ (asserts nothing named `wpm`/`fpo`
  reaches `sys.modules` or `sys.path`).
- Target **`stitch`**, not `navigate`: official GCBC already scores **99%** on
  `pointmaze-medium-navigate` (continuous expert trajectories — BC just follows)
  but **23%** on `stitch`, where trajectories must be composed. Reasoning has
  room to matter only in the latter.

## Layout

```
lacot/           the model — backbone (CausalTransformer), e_target
                 (PerceiverPooler / ETargetGenerator / FrameDecoder),
                 nf_head (Flow, ARFlowBlock), heads, model (LaCoTActor,
                 RefineOperator).  No wpm imports.
experiments/     18 clean experiment scripts (2026-08-22 research day):
                 scratch_lacot_oracle.py   — the ORACLE=100% run
                 scratch_lacot_rollout.py  — success-rate eval (real rollouts)
                 scratch_value.py       — GCIVL value training
                 exp_u_diagnose.py      — the u-sampling diagnosis
                 scratch_wiring_check.py— plumbing check, no training needed
docs/            LaCoT-NF-latent-planning-design.md (the design, born 2026-08-17)
checkpoints/     scratch_value.pt (trained GCIVL value)
legacy/          quarantined, NOT part of LaCoT:
                 wpm_toplevel_model.py  — WPM's image-based top-level model
                 lacot_*.py                — old smokes that import wpm
                 visual-track/          — 8 scripts still needing
                                          wpm.data.pipeline / ImpalaSmall /
                                          build_pointmaze_pixel_env; the visual
                                          track was retired in favour of state.
```

## Running

Data and GPUs live on the compute nodes, not on zeldajr.

```bash
# on jasmine (3x RTX 4060 Ti)
cd /archive/cymaxwelllee/lacot
OGBENCH_DATA_DIR=/archive/cymaxwelllee/data/ogbench \
  .venv/bin/python -u smoke_lacot.py             # independence + forward check
OGBENCH_DATA_DIR=/archive/cymaxwelllee/data/ogbench MUJOCO_GL=osmesa \
  .venv/bin/python -u experiments/scratch_wiring_check.py
```

- `OGBENCH_DATA_DIR` — official env var; defaults to
  `/archive/cymaxwelllee/data/ogbench` (12G, 16 files). Nothing hardcodes a path
  any more.
- `LACOT_CKPT_DIR` — overrides where `checkpoints/` is written.
- Canonical source lives in `~/Projects/lacot/` on zeldajr (deliberately OUTSIDE the shared maid workspace repo — 主人 2026-08-23: LaCoT gets its own repo); jasmine holds a synced copy
  plus its own `.venv` (torch 2.6.0+cu124). Heavy artefacts stay on `/archive`,
  never in the repo.
- ⚠️ Use Slurm `sbatch` for real runs — ad-hoc `ssh` bypasses the scheduler.

## Open threads

1. Reconcile the refine/thinking route with **NF-CoT** (arXiv 2606.06447), whose
   core is supervised distillation + single-shot autoregressive sampling — it has
   **no refine loop**. Ours does, and it measurably (if weakly) works.
2. Wire `value-directed refine` (the value is trained and waiting).
3. Rebuild on the official OGBench framework: official raw data + eval, own
   in-segment relabel, then re-confirm oracle ≈ 100% and debug `u`.
4. Reimplement segment-boundary detection independently — the oracle run used
   fpo's `derive_subtrajectory_boundaries` (maze_arrival, 7.73 segments/episode).
   That dependency must be paid off in our own code.
5. `experiments/` are research scratch scripts, not a library. They earned their
   results; they still need to be folded into a proper training entry point.
