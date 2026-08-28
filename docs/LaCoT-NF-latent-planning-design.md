# LaCoT — NF Latent Reasoning for Trajectory Planning (design)

_Owner: Chun-Yi (主人). Drafted with Luna, 2026-08-17. Status: DESIGN (implementation-first, build block-by-block)._
_This document is the durable record of the LaCoT idea born on 2026-08-17. Build from this._

---

## 0. One-line

A compact **continuous-latent reasoning** step: a normalizing-flow "thought" `u`, iteratively refined to convergence, then decoded into an **action chunk**. Test-time scaling = number of refinement rounds.

---

## 1. Motivation

- **M3 (self-generated futures + recursion) does not beat the GCBC floor.** The explicit-future-imagination mechanism does not add value. Concretely its known pains:
  - generating good future **images** is hard and expensive;
  - **self-copy**: a self-generated future contains only the proposal that conditioned it, so the future-conditioned readout collapses into copying its own proposal (design doc §5.6);
  - the value/progress machinery is fragile (collapses; needs warm-start).
- **Hypothesis (主人)**: reasoning in a **compact continuous latent space** (à la NF-CoT) instead of high-bandwidth explicit futures can realise the "planning helps" thesis where explicit futures failed. **Feels like it can beat M3.**

## 2. The pivot experiment (decides whether LaCoT is worth it)

**Does M2 (oracle, TRUE futures) beat the floor?**
- **oracle > floor but M3 ≈ floor** → the ceiling exists; the problem is *generating* good futures → **LaCoT (latent, no image generation, no self-copy) is the right fix.** ✅
- **oracle ≈ floor** → the "future context helps" premise itself is weak → LaCoT (same premise, different substrate) may not help; rethink.
- ⇒ The M2 oracle run we are building (see `docs/` M2 chain) is exactly this experiment. Also ask 主人 what the students' oracle-vs-floor result was.

## 3. Design (build spec, 主人 2026-08-17)

Data = the trajectories M3 already collects for training → `(state, goal, action_chunk)` (+ the future trajectory for the flow target).

```
(current state, goal)  --encoder-->  e            # compressed conditioning embedding
                        --project-->  u_0          # initial latent "thought"
u  --AR NF, iterate a few rounds-->  u*            # latent reasoning / planning; converges
u* --decode (action head)-->  action / action_chunk
```

**Three losses:**
1. **NF loss** `L_NF = -log p(u | state, goal)`, where `u` is trained to align with **e_target = the compressed FUTURE TRAJECTORY** (主人's pick, 2026-08-17). i.e. the trajectory plays the role of NF-CoT's "CoT" — the reasoning trace, VAE/encoder-compressed into the flow's target. At **inference** we do NOT need e_target — sample `u | (state,goal)` from the flow; the trajectory knowledge is baked into the flow.
2. **action loss** — decode the action chunk from the converged `u` (behaviour cloning on the trajectory data). This **end-to-end supervises the converged `u*`** (at the refine loop, NOT the flow density — the flow stays pure, see Q2) — so, unlike NF-CoT, we need no separate CoT-distillation teacher.
3. **consistency loss** (主人's addition) `L_cons = || u_t - sg(u_{t+1}) ||²` (consistency-model style, stop-grad) — forces the iterative refinement to **converge to a stable `u`** and not drift. This convergence guarantee is exactly what a naive iterate-and-refine loop lacks.

**Why the core is clean:** reasoning lives in the latent `u` — no explicit future-image generation, no self-copy from label readback, no fragile value/progress token. `u` is supervised end-to-end by the action loss + regularised by the flow + stabilised by consistency.

## 4. Implementation path (recipe + engine, both known)

- **Recipe = NF-CoT** ("Latent Reasoning with Normalizing Flows", arXiv:2606.06447, 2026-06). **No code released** — build from the paper (method in §7).
- **Engine = TARFlow** ("Normalizing Flows are Capable Generative Models", Apple, ICML 2025, arXiv:2412.06329). **Official code: github.com/apple/ml-tarflow** (also STARFlow: github.com/apple/ml-starflow). Key file `transformer_flow.py` (`Flow` class). Loss = NLL; forward = likelihood (log-det Jacobian), inverse = sample.
  - ⭐ **TARFlow operates on a sequence of token vectors, not only image pixels** — so it drops in as a density/reasoning head over the model's latent tokens. This is why it fits.
- **Components, all matched-capacity to the SOTA baselines (HIQL/CRL), not inherited from any prior in-house model**: 👁 image encoder = the SOTA visual baseline's small CNN (deployed); 🧠 a conditioning backbone (matched-capacity transformer over the latent tokens); 👄 action head (chunk_len 8 × action_dim 5 × 256 bins). **LaCoT's core new block = the "generate u" NF head** (TARFlow engine) + the training-only flow-target trajectory encoder.

## 5. Methodology (主人 2026-08-17): block-by-block

Build one block, **validate it (can sample / compute likelihood / is stable) before moving up**. Do not stack everything then debug. Measure-first at every step.

**Build order:**
1. Minimal "generate `u`" block using the TARFlow `Flow` class — validate: samples a latent sequence, computes exact log-likelihood, is stable.
2. Add the flow-target: encode the future trajectory → e_target; wire `L_NF` (u aligned to e_target). Validate.
3. Add the action head decode from `u` + `L_action`. Validate the end-to-end BC works.
4. Add the iterative refinement (a few rounds) + `L_consistency`. Validate convergence/stability.
5. Wire the full model together (encoder → NF head → backbone → action head). Two-stage training (§7).
6. Eval: does it beat the floor? (the LaCoT test).

## 6. References to explore (主人 2026-08-17)

- **World Value Model (WVM)** — arXiv:2606.24742, "World Value Models for Robotic
  Manipulation" (Wang et al., 2026-06); READ 2026-08-19. NOT a latent->action policy
  (the 2026-08-17 note was imprecise): it is a **value model** that marries a **world
  model with value estimation** for deep temporal understanding, claims SOTA
  Value-Order-Correlation, and stays reliable on **suboptimal / mixed-quality data**
  (its own Suboptimal-Value-Bench), sim + real manipulation. **Relevance to LaCoT**: it
  targets the exact pain point that pushed us value-free -- value estimation being
  fragile on mixed-quality data. **Role for LaCoT (主人 2026-08-20): WVM is a BASELINE to
  beat -- add it to the baseline set.** LaCoT (value-free latent planning) aims to beat WVM
  (world-model + value) on the shared eval. (Enter it via its own policy/value readout;
  pin the exact comparable form when baselines are built.) It also stays a candidate
  critic if LaCoT later adds the optional AWR (§7), but its primary role is now a baseline. *(abstract-level read; full method / VOC definition
  / training not yet read.)*
- **Swappable NF engine**: the "generate `u`" block is **modular**. TARFlow is one engine; could swap for a different normalizing flow, or **Kaiming He's drift / flow-matching family** (often simpler/faster). Plan: build+validate with TARFlow first (has code), then try swapping the engine. *(find the exact Kaiming He "drifting model" ref tomorrow.)*

## 7. NF-CoT precise method (implementation reference, from arXiv:2606.06447)

- **Compression**: frozen VAE encoder turns explicit CoT `d_{1:L}` → continuous codes `e_{1:K} ∈ ℝ^{K×D}` (K=64 slots, posterior mean as target). Shallow flow blocks `F_θ` (5 MetaBlocks, alternating Identity/Flip, near-identity init, triangular Jacobian): `u_{1:K} = F_θ(e_{1:K}; q)`. VAE frozen; shallow blocks + projections trainable.
- **NF head (causal Gaussian AR density)**: `p_θ(u_{1:K}|q) = ∏_i N(u_i; μ_θ(q,u_{<i}), diag(σ²_θ(q,u_{<i})))`. Backbone + NF head outputs μ, σ². Sampled left-to-right.
- **Exact likelihood**: `log p_θ(e_{1:K}|q) = log p_θ(u_{1:K}|q) + log|det J_{F_θ}(e;q)|`.
- **Objective**: `L_sup = λ_flow · L_flow + λ_text · L_text`, with `L_flow = -log p_θ(e|q)`, `L_text = -Σ_j log p_θ(x_j|q,u,x_{<j})`. Small Gaussian noise on `e` before the flow loss (robustness). *(For LaCoT: L_text → L_action; the CoT target → the trajectory target; add L_consistency + iterative rounds.)*
- **Two-stage**: Stage 1 freeze backbone, train shallow blocks + projections; Stage 2 unfreeze all. (Removing Stage 1 hurts.)
- **Inference**: no VAE/shallow blocks at test time; sample `u|q` (temp 0.9) → decode (temp 0.6), reuse KV-cache. 64 latent tokens.
- **Latent-space RL** (optional later): GRPO, `log π_θ(u,x|q) = log p_θ(u|q) + log p_θ(x|q,u)`, execution reward, group-normalised advantage, shallow blocks frozen. Latent-space RL preserves pass@k diversity where token-space GRPO concentrates.

## 8. Open questions / decisions

- ✅ **flow target = compressed future trajectory** (主人 2026-08-17) — not action-chunk.
- How exactly to "compress the trajectory into e_target" — encoder over the trajectory frames/states (analogue of NF-CoT's VAE over CoT tokens). Decide the trajectory horizon / representation.
- Number of refinement rounds (test-time-scaling axis); the exact consistency-loss form (stop-grad target).
- Whether to keep NF-CoT's two-stage curriculum or train end-to-end for the robot setting.
- **K (number of latent tokens in u) — a hyperparameter, sweep it** (主人 2026-08-19).
  NF-CoT uses K=64 to compress a full CoT; our u compresses a TRAJECTORY (different
  information density), so treat 64 as a reference anchor only and sweep K
  empirically. NF-CoT keeps alternating Identity/Flip permutations even at K=64,
  which supports our reverse-flip default for the (likely smaller) LaCoT K; revisit a
  learned 1x1-conv permutation only if a large K + deep stack shows order-bias.

- ✅ **F (the e_target -> u shallow-flow warp): START AS IDENTITY** -- u = e_target,
  log|det J_F| = 0, so L_NF reduces to training component 1's density directly on
  e_target (measure-first, fewest moving parts; 主人 2026-08-20). ⏸ **DO-NOT-FORGET
  TODO**: once the density is wired on e_target, if component 1 cannot fit e_target
  well (underfits -- e_target's distribution too warped for the AR-Gaussian), SWAP F
  for a SHALLOW ARFlowBlock stack (a few layers, near-identity init, cond = (s,g);
  reuse `nf_head.ARFlowBlock`, whose triangular Jacobian gives
  log|det J_F| = -sum(alpha)). This warp is the expressiveness lever if the identity
  density underfits -- 主人 explicitly asked to not forget it.

## 9. Status

- 2026-08-17: idea born + design drafted (this doc). NF-CoT read; TARFlow code path found. Not yet coded.
- 2026-08-18: design session with 主人 (§10). Component 1 (the flow) coded in
  `wpm/models/nf_head.py` — NOT yet validated. Walked the NF fundamentals +
  consistency loss + EMA + the open-design menu, one step at a time.
- Next: validate component 1 (block-1 smoke), then resolve the refine-operator
  wiring (Q1/Q2 in §10) before coding the refinement.

## 10. Design session 2026-08-18 (with 主人) — consistency loss, refinement, open choices

✅ = decided · 🔶 = current lean (revisit) · ⏳ = still open / being designed.

### Two distinct "iterations" (do NOT conflate — same letter u, different level)

- **token level**: u_0, u_1, ..., u_{K-1} — the K tokens of ONE thought,
  generated left-to-right by the autoregressive flow (component 1).
- **round level**: u^(0) -> u^(1) -> ... -> u* — the WHOLE thought refined over
  rounds; THIS is "planning = test-time scaling". The consistency loss and the
  refinement operator live at the round level.

### Consistency loss (主人's addition) — purpose and the collapse trap

- Purpose: make the round-refinement CONVERGE to a stable fixed point u*, not
  drift/oscillate. Skeleton form: `L_cons = || u^(r) - sg(u^(r+1)) ||^2`.
- ⚠️ **Consistency ALONE has a trivial useless solution: collapse to a
  constant** (every round, every input -> the same c; perfectly consistent,
  perfectly useless). So it is NEVER used alone.
- ✅ **"Good" is defined by the OTHER two losses, not consistency**:
  - action loss (BC): u* must decode to the correct action chunk — anchors
    WHERE the fixed point should be (what "good" means).
  - flow loss (L_NF): u aligned to the compressed future trajectory — gives u
    meaningful structure.
  - Division of labour: action + flow decide "what a good u* is"; consistency
    makes the refinement stably reach it. This patches M3's failure directly
    (its recursion depth did not converge to anything better).

### EMA target — mechanism and the stop-grad clarification

- Two networks: student θ (trained) + teacher θ_ema (NOT gradient-trained;
  `θ_ema <- decay·θ_ema + (1-decay)·θ`, decay ~0.999 ≈ avg of last ~1000 steps).
  Consistency target is produced by the teacher.
- Benefits: (1) stable, non-moving target (vs a live self-target that jumps each
  step); (2) extra insurance against the constant-collapse; (3) an averaged
  teacher is usually a bit better than any single snapshot (Polyak/SWA), so the
  student chases a slightly-stronger self.
- ⚠️ **stop-grad and EMA are ORTHOGONAL, not alternatives** (主人 caught this):
  stop-grad = "don't backprop into the target" (always on); EMA = "the target's
  weights are a slow average". The real fork is target-from-LIVE-net (sg) vs
  target-from-EMA-net (also sg). Both are stop-grad.
- Already used in this repo (train_gciql target critics, ema.decay 0.995/0.999).

### Open-design menu + current leans

1. 🔶/⏳ **refine operator** (u^(r+1) from u^(r)) — lean: the conditioning backbone reads
   (state, goal, u^(r)) -> NF head emits u^(r+1) (append the previous latent thought
   back into context and re-reason, entirely in latent space). Being finalized against
   NF-CoT's exact wiring (Q1) + the gradient-routing question (Q2).
2. 🔶 **action loss placement** — lean: every round (deep supervision), so every
   depth is decodable -> supports test-time scaling + resists collapse.
3. 🔶 **consistency target** — lean: start with live stop-grad; add the EMA
   teacher only if we see instability/collapse (measure-first; the action loss
   is the primary anti-collapse anchor).
4. 🔶 **which round pairs** — lean: adjacent (u^(r) vs u^(r+1)); if the operator
   contracts, adjacent consistency propagates to a global fixed point.
5. ✅ **training depth** — random depth per example, R ~ U{0..3} (like M3), so
   every depth is in-distribution and a test-time depth sweep is valid.
6. 🔶 **loss weights** λ_flow / λ_action / λ_cons — lean: action primary (1),
   flow + cons as regularizers. Tune LATER with Optuna (TPE + ASHA pruning) or
   Ray Tune (parallel across the 4 GPUs), and only after the mechanism works
   (do NOT tune before the core signal is verified; CLAUDE.md).

### Q1 resolved — the flow <-> u <-> decode wiring (NF-CoT structure)

Two distinct latents; keeping them apart is the whole thing:
- **e_target** = the TARGET, compressed: our compressed FUTURE TRAJECTORY
  (NF-CoT's analogue is the VAE-compressed CoT). Exists only at training.
- **u** = the flow-space "thought"; the density p(u | s, g) lives here.
- The shallow flow F maps between them at training: `u = F(e_target; s, g)`.

Two losses: L_NF makes the real e_target high-probability
(`log p(e|s,g) = log p(u|s,g) + log|det J_F|`); L_action decodes the action
chunk from u (BC).

⭐ **Train-vs-inference asymmetry** (the key):
- TRAIN: we HAVE the future -> e_target -> u = F(e_target; s,g). u is computed
  from the real future.
- INFER: NO future. Sample u ~ p(u|s,g) directly (the component-1 AR flow,
  left-to-right, conditioned ONLY on s,g). No VAE, no e.
- **The density p(u|s,g) never takes e as input** — e is only the training
  target. Future-knowledge is baked into the density's weights at training and
  the future vanishes at inference (exactly like an LM: real text trains it,
  inference samples p(next | context)).

### Q2 resolved — gradient routing: PURE flow (imagine trajectories), policy-awareness at the refine loop

The tension: L_NF wants u to be a faithful density of e_target; L_action wants u
to decode to the right action.

⭐ **Decision (主人 2026-08-21): keep the flow PURE.** The flow's one job is to
IMAGINE the trajectory — learn p(u|s,g) matching e_target. The action loss does
NOT shape the flow. Rationale (主人, verbatim): "flow 應該專心練出想像的
trajectories，不應該被 action 影響".

Why this is NOT "policy-blind" (the 2026-08-18 worry, now resolved):
- **The action info is ALREADY IN THE TRAJECTORY.** e_target = the compressed
  future trajectory; the executed motion (hence the action) is derivable from it.
  So a flow that imagines the trajectory well already yields a u the action head
  can decode — the flow need not be distorted toward actions. *(Evidence: the
  full-model wiring smoke's inference is coherent — matched action MSE 8e-5 — with
  a PURE, L_NF-only flow. `wpm/models/lacot_smoke.py`.)*
- **Policy-awareness enters at the REFINE loop, not the flow.** The refine
  operator reads (s,g,u) and is deep-supervised by L_action, so it POLISHES the
  flow's imagined u^0 into an action-decodable u*. Clean division of labour:
  **flow imagines the trajectory (pure generative); refine + action polish it to
  be executable.** u is not policy-blind — the polishing just happens one layer up.

Concretely under **F=identity** (current build): the action gradient does not
reach the flow at all (u_target = e_target is frozen; the refine u^0 is a no-grad
sample). VERIFIED by `wpm/models/lacot_gradroute_smoke.py` (5/5): L_NF → flow only;
L_action → head + refine; L_cons → refine; the generator is frozen throughout.

- ✅ **DO freeze the target-encoder** (the trajectory → e_target compressor),
  pretrained by RECONSTRUCTION. e_target's job is to faithfully represent the real
  future — it is the label. If any downstream loss can reshape it, the model
  cheats by making the target trivial → collapse. This is NF-CoT's frozen-VAE
  choice; 主人's intuition.
- **One line: keep the flow pure (imagine trajectories, L_NF only), freeze the
  target (keep the label honest), and let the refine loop carry policy-awareness.**

⛔ **SUPERSEDES the 2026-08-18 note** "Do NOT block action → flow / blocking makes
u policy-blind". That framing assumed u's only policy signal was the action
gradient into the flow; the refine loop (added in the §10 session) is where
policy-awareness now enters, so the flow can stay pure. ⏸ If a future variant
makes F a real warp (§8), revisit whether the action should shape F (not the flow
density itself).

### Q3 resolved — which u to decode from at training (主人 2026-08-19)

The question: at training, decode the action from the e_target-derived u
(`u = F(e_target)`, future-informed, clean signal but never seen at inference)
or from a SAMPLED `u ~ p(u|s,g)` (matches inference, but noise while the flow is
still untrained)? This is the classic train/inference mismatch (exposure bias,
= teacher-forcing in LMs).

- **Reframe (the key):** the two u's are MEANT to converge. L_NF's whole job is
  to pile p(u|s,g)'s mass where the e_target-derived u lands, so at convergence a
  sampled u is distributed like the target u and the mismatch vanishes. The
  mismatch is therefore a TRANSIENT (early-training) problem, not a fundamental
  one — we do not pick one u forever, we schedule "stable early, inference-aligned
  late."
- **The schedule is a curriculum** (= scheduled sampling): decode from the target
  u early (clean), blend in the sampled u late (closes exposure bias).
- ✅ **Decision — decode from BOTH, and get the curriculum for free from refine:**
  1. **Target-u anchor (A):** an explicit BC on the action decoded from
     `u = F(e_target)`. Always-clean signal; weight it HIGH early. (= stable early)
  2. **Refine-round deep supervision (bridge to B):** run the refine loop from a
     sampled-style `u^(0)` and apply the action loss at EVERY round `u^(0..R)`
     (reinforces open-menu #2). Because the refine operator reads
     `(s,g,u^(r)) -> u^(r+1)` — literally the inference computation — deep
     supervision makes the decoder see the whole rough->refined spectrum =
     inference-aligned, with no hand-tuned scheduled-sampling ratio; the flow
     maturing lets this carry the load late. (= inference-aligned late)
  - One line: **A is the north star (stable early), the refine rounds pave the
    bridge to B (inference-aligned late); the refine loop IS the curriculum.**
- ⚠️ **Residual risk (measure-first):** early on, round-0's sampled `u^(0)` is
  noise, so deep-supervising the action from it = learning from noise (can drag,
  or teach the decoder to ignore u). Mitigate: (a) the (A) anchor always dilutes
  it; (b) WARM-UP — enable the refine-round supervision only once L_NF is decent,
  or ramp its weight from 0. The gauge to watch: the exposure-bias gap = action
  decode quality on sampled-u vs target-u. Gap stays large -> tune the mix / add
  explicit scheduled sampling; gap small -> done. Do NOT pre-pay for a fix we may
  not need.

### Training data = hindsight relabeling; value-free by default (主人 2026-08-18)

- **Data engine = hindsight relabeling** (主人's framing, confirmed): the pile of
  trajectories, relabeled by treating a state actually reached later as the goal
  g, yields a huge number of (s, g, the-real-future-between-them) triples. Each
  real future is the e_target that trains the flow p(u|s,g). Self-supervised,
  abundant, no manual labels — and OGBench's pipeline already does hindsight
  relabeling.
- **Value-free by default.** M3's stack — a goal-conditioned value V(s,g) +
  explicit future-conditioning — is fragile when (s,g) vary (measured:
  cube-double value collapse; AWR weighting was a net poison on pointmaze-large;
  the future generator fails on cube-double). LaCoT sidesteps both: the future is
  latent (no pixel generation), and the core losses (flow NLL + action BC +
  consistency) need no value. ✅ Start value-free (pure hindsight BC).
  - ⚠️ Honest caveat: pure BC clones average behavior; if the data has bad paths
    it can learn mediocrity. The §7 objective keeps an OPTIONAL AWR weight
    (`w_t = clip(exp((V(s_{t+H}) - V(s_t)) / β))`) to upweight better-than-average
    actions. Add it only if pure BC underperforms (measure-first) — do NOT carry
    M3's value burden by default.

### Fairness of the comparison, and the engine-swap extension (主人 2026-08-18)

- **Is value-free LaCoT vs value-based baselines unfair? No — the opposite.** LaCoT
  forgoes the value function (a source of better-than-BC improvement), so it
  fights HIQL/GCIQL with one hand tied; winning value-free is a STRONGER claim,
  not an unfair advantage. Benchmarks compare whole methods (HIQL = hierarchy +
  value, CRL = contrastive, GCBC = pure BC already differ wildly and are compared
  directly). The fairness that matters is same-data / same-eval / same-tuning-
  budget (already required by the eval rules), not same-components. To be airtight:
  report LaCoT both value-free AND with the optional §7 AWR, plus a GCBC floor on the
  SAME hindsight data (isolates the latent-reasoning mechanism from the data).
- **Engine-swap extension** (主人): the "generate u" block is modular (§6) — swap
  TARFlow for flow matching / rectified flow / Kaiming-He drift models. Often
  simpler + faster to sample than an AR flow, and SoTA-strong. Trade-off: the flow
  loss changes from exact-likelihood NLL to a velocity-field regression (MSE, often
  easier to train); exact likelihood then needs ODE integration (pricier), but LaCoT
  does not require it. Plan: build + validate with TARFlow first (has code, exact
  likelihood, easy to debug), then swap the engine as an ablation — bonus paper
  point: the mechanism holding across engines = robustness.

### Build status / next

- Component 1 (the "generate u" flow) coded: `wpm/models/nf_head.py`
  (ARFlowBlock + Permutation + Flow) and VALIDATED (主人 2026-08-19). Block-1
  smoke `wpm/models/nf_head_smoke.py` (run: `.venv/bin/python -m
  wpm.models.nf_head_smoke`) is 15/15 green: round-trip invertible to ~1e-15;
  analytic `-sum(alpha)` == autograd-Jacobian slogdet to ~1e-16 (the exact
  likelihood is real); mean NLL reaches the analytic differential entropy on a
  known diagonal Gaussian (rel 0.08%) and on a nonlinear AR chain (rel 6%,
  beating the best single Gaussian by 4.6 nats and reproducing the conditional);
  no NaN/Inf across training and sampling.
- New component implied by Q2: a **trajectory encoder (-> e_target), pretrained
  by reconstruction and frozen** (analogue of NF-CoT's frozen VAE).

## 11. Design session 2026-08-19 (with 主人) — e_target (target side) design

Walked the whole NF flow (component 1) block-by-block, then designed the target
side. ✅ = decided-lean · 🔶 = option/ablation · ⚠️ = guard.

### What e_target compresses
- ✅ e_target = the compressed FUTURE STATE trajectory (NO actions). For the
  visual OGBench variants an observation is a 64x64x3 RGB frame, so e_target
  compresses a SEQUENCE OF FUTURE FRAMES.
- ⚠️ Leakage red-line (fpo hard constraint #6): e_target must NOT contain future
  actions, nor near-future frames closer than 2x the chunk length (those pin the
  executed action down via inverse dynamics). e_target compresses [s + 2*chunk ... g].
- ⚠️ Keep TARGET vs CONDITIONING separate: e_target is the future only; (s,g) is
  the flow's cond, never folded into e_target.

### Horizon
- ✅ Because we hindsight-relabel, the "future" is the segment from s to the
  relabeled goal g, so horizon length = how far g is; the horizon question reduces
  to the g-distance sampling distribution.
- 🔶 Lean (主人 agrees "various lengths"): sample g at a NEAR-to-FAR mixture of
  distances (learn short- and long-horizon planning; also feeds the refine /
  test-time-scaling story).
- The Perceiver compressor decouples future length from embedding size, so a long
  horizon is fine.

### Encoder (the compressor) + size
- Trajectory encoder = a per-frame image encoder (SAME / matched-capacity as the
  SOTA baselines' encoder (the OGBench visual small CNN), NOT inherited from WPM or a fancier backbone)
  feeding a Perceiver / learned-query cross-attention that pools the variable-length
  future frame-features into a FIXED K x D e_target (Perceiver Resampler / Set-
  Transformer PMA / DETR-style queries).
- ✅ Reconstruction-pretrain encoder+decoder, then FREEZE the encoder as the
  e_target generator (Q2; NF-CoT's frozen-VAE analogue). The decoder forces
  e_target to represent the future faithfully (no collapse) -- 主人's intuition.
- K x D (e_target size) is a capacity bottleneck: too small loses long/complex
  futures; too big weakens abstraction (drifts toward frame replay) and risks
  detail/leakage. It is about COMPRESSIBILITY, not raw length (goal-directed
  trajectories are redundant). SET IT BY RECONSTRUCTION QUALITY (measure-first:
  sweep K x D, watch decoder reconstruction across the horizon distribution).
  NF-CoT's K=64 is the starting anchor.
- 🔶 Ablation (主人 2026-08-19): add a few GLOBAL/summary query tokens (ViT [CLS]
  analogue) so e_target = [global codes] + [detail-route codes]. Partly redundant
  (learned queries are already global-context-aware), so keep it an ablation; the
  global summary is of the FUTURE trajectory only (not (s,g), not actions).

### Fairness (主人 2026-08-19)
- ⭐ Calibrate to SOTA, not WPM (主人 2026-08-19): LaCoT is its own thesis, compared
  against the real SOTA (HIQL / CRL / WVM — WVM added as a baseline to beat, 主人 2026-08-20). The image encoder should match THEIR encoder
  (OGBench's visual-baseline small CNN), NOT be inherited from WPM just because the
  code already has one. WPM is potentially a separate paper; do not let its internal
  choices steer LaCoT's comparison ("don't get led by WPM").
- The per-frame image encoder is DEPLOYED (runs at inference to encode cond=(s,g)),
  so it must be matched-capacity with the SOTA baselines' encoder -- the real
  fairness lever. Do NOT sneak in a fancier backbone (a big ViT is overkill for
  64x64 and off the baseline norm; if used, every method uses it).
- The Perceiver trajectory-compressor is TRAINING-ONLY (frozen, produces the target;
  never runs at inference, exactly like a baseline's value function), so it does not
  inflate deployed capacity -- fair in the same sense a critic is. Report the GCBC
  floor on the same hindsight data to isolate the latent-planning mechanism.
- Principle (as in S10): same data / same eval / same tuning budget / matched
  DEPLOYED capacity -- NOT same components.

---

## 12. Literature session 2026-08-24 (with 主人) — test-time scaling for latent reasoning

_主人's actual goal (stated 2026-08-24): **find papers where "thinking many times" at test
time actually gets better.** He first sent arXiv:2604.21215 believing it was such a paper.
Recorded at 主人's explicit request ("這幾篇也要記錄進去這個 project doc")._

_Context: this session followed the 2026-08-23 finding that our `u` has collapsed to a
single constant vector (pairwise cos = 1.0000 across 512 different (s,g)), see
`docs/FINDINGS-2026-08-23.md` §1._

### 12.1 The four papers

| # | Paper | Verdict for LaCoT |
|---|-------|----------------|
| a | **Recurrent Transformer** — arXiv:2604.21215 (Oncescu, Morwani, Jelassi, Meterez, Kwun, Kakade; 2026-04-23) | ⛔ **NOT what 主人 wanted.** Its recurrence runs along the *sequence axis* (layer `i` attends to KV computed off its own post-update activation `z_i`, Eq. 1–2). Paper is explicit: depth is baked into the architecture, **inference depth is not variable** — you cannot "think more times" at test time. Still useful for one thing, see 12.4. |
| b | **Huginn / Recurrent Depth** — arXiv:2502.05171 | ⭐ **The paper 主人 was looking for.** 3.5B, latent recurrence, up to 50 loops at test time, reasoning score rises with loop count. |
| c | **STARS** — arXiv:2605.26733, "Stabilizing Recurrent Dynamics for Test-Time Scalable Latent Reasoning in Looped LMs" | ⭐ Directly targets *our* disease: keeps latent states at **per-input** asymptotically stable fixed points via Jacobian-spectral-radius regularization. |
| d | **Recurrent-Depth VLA (RD-VLA)** — arXiv:2602.07845 | ⭐ Same domain as us (latent iterative reasoning → **action**). Prelude / weight-tied recurrent core / coda. |

### 12.2 🚨 The structural finding — why ours collapses and theirs does not

**All three working papers (b, c, d) put the latent on the ONLY path to the output.**

- Huginn: `s_i = R(e, s_{i-1})`, then `p = C(s_r)` — the coda reads **only** `s_r`.
- RD-VLA: coda decodes **only** the converged scratchpad into action space.
- The question is *not* removed from the model — it is **re-injected into the recurrent
  core at every step** (Huginn concatenates `e` each iteration through a learned adapter
  2h→h). So question-information reaches the output **only by passing through the
  recurrent block**.

⇒ **LaCoT's `ahead(cond, u)` reads BOTH.** `cond` alone gets 0.922 on medium-navigate
(the honest BC floor, FINDINGS-2026-08-23 §2), so `u` can be garbage at no cost.

⇒ 🚨 **Reinforcing evidence: none of the three has any latent-level supervision** — the
only training signal is the final output loss (verified per-paper). We *do* have a latent
target (`e_target`) and still collapsed. **So our missing piece is not "give `u` a target";
it is "`u` is not on the critical path".**

_(This is the same 👁 eye = cond / 🧠 brain = u / 👄 mouth = head metaphor used with 主人
on 2026-08-17: their mouth can only listen to the brain; ours can also look at the eyes.)_

### 12.3 Training-time recurrence count — all three SAMPLE it, we fix it at 3

| paper | train-time loop count | backprop |
|-------|----------------------|----------|
| Huginn | log-normal Poisson, mean r̄, σ=1/2, heavy tail | truncated, last **k=8** iterations only |
| RD-VLA | log-normal Poisson, μ_rec = **32** | TBPTT, last **d=8**, earlier steps detached |
| STARS | uniform/log-normal/Poisson over [1,16] or [1,100] | — |
| **LaCoT (ours)** | **fixed 3** | full 3 |

⇒ ⭐ **This alone explains the 2026-08-23 observation that R0…R8 are all identical.**
The model was never asked to be good at round 8. Absence of test-time scaling in LaCoT is
not evidence that test-time scaling doesn't work — **we never entered the game.**

### 12.4 Two routes to stability (they are complements, not rivals)

**Architectural route (Huginn, RD-VLA)**
- *Input injection* every step, through a **learnable** adapter. Huginn's framing:
  like gradient descent, the objective's data must be available at **every** step, not
  only at initialization (§3.1).
- Sandwich RMSNorm (before+after both attn and MLP) — reported **essential at scale** (§3.2/4.3).
- 🚨 **Huginn hit our exact failure**: an early variant without learnable adapter weights
  had **token correlation collapse to 1.0** (their Fig. 5). Same number as our cos = 1.0000.
- Init: σ_h² = 2/(5h); out-proj σ² = 1/(5hl).
- RD-VLA states input injection is *the* mechanism that prevents representational collapse.

**Regularization route (STARS)**
- `L_JSRR = (1/N) Σ ‖J^(t,i) v^(t,i)‖²`, J = Jacobian of the recurrent block at iteration t.
- ρ(J) estimated by **single-step power iteration via JVP** — no explicit Jacobian.
- `L_STARS = E_{t~P}[(1-λ)·L_SFT^(t) + λ·L_JSRR^(t)]`; target ρ(J) < 1.

**From paper (a), the one that "didn't fit":** its Theorem 4.1 gives the no-exploding-
gradient condition `max eig(αV) < 1` under uniform attention, plus RMS normalization
before the persistent K/V. ⇒ Same family of medicine as the LayerNorm fix already on our
carry-over list.

⚠️ **Correction to a mid-session claim (Luna, 2026-08-24):** I first said "contraction ⇒
collapse" (Banach: a contraction has a unique fixed point, so every start converges to it).
That is **half wrong**. STARS *deliberately* makes the map contractive; contraction is fine
as long as the output-side loss carries the content, because then the fixed point is
**per-input**. Our problem is not contractivity, it is that our output side isn't carrying
anything (the head has a bypass). ⛔ Do not "fix" LaCoT by trying to avoid contraction.

### 12.5 The honest bad news — test-time scaling is not free

```
STARS   Ouro-1.4B-STARS   4 steps 74.18%  <- peak
                          8 steps 65.55%  (-8.63; SFT baseline drops -17.49)
RD-VLA  LIBERO             1 iter   8.4%
                           2 iters 40.5%
                           4 iters 84.1%
                           8 iters 92.6%
                          12 iters 93.1%  <- peak
                          16+      slight degradation
Huginn  easy (OpenBookQA)  saturates ~8 recurrences
        hard  (GSM8K)      still improving at 32-50; 47.23% at r=64 w/ weight averaging
```

- STARS states plainly (Appendix A): *"performance does not always improve monotonically
  with more steps."*
- Precise shape: STARS and RD-VLA **peak then decline**; Huginn **saturates** rather than
  reversing, and hard questions keep gaining to ~50.
- ⭐ Huginn §5.3/Fig. 9: **the saturation point moves with problem difficulty** (ARC-Challenge
  saturation correlates with few-shot context size). That is exactly the property LaCoT wants:
  extra thinking pays off only where it's needed.
- ⇒ RD-VLA is the strongest single piece of evidence for 主人's question: **in an action
  domain, thinking more times really does get better** (8.4% → 93.1%).

### 12.6 ⇒ What LaCoT is missing, in dependency order

```
① u must be on the critical path      (head stops eating cond)
② train-time round count must be sampled, not fixed at 3
③ stability machinery                 (injection adapter / spectral radius / RMSNorm)
```

⛔ **Order matters**: with ① undone, ② and ③ change nothing — there is no pressure holding
content in `u`, so a deeper or better-conditioned recurrence just reaches the same constant.

**Proposed verification order (Luna 2026-08-24, awaiting 主人):**
- **Step 0** (seconds, existing checkpoint, no training): measure head sensitivity
  `∂ahead/∂u` vs `∂ahead/∂cond`. Directly measures how wide the bypass is, and — important —
  **stays valid even though `u` is already constant**, because gradients are local and do
  not depend on `u`'s distribution. **Falsifiable**: if the head turns out to be sensitive
  to `u`, the whole diagnosis above is wrong and Step 1 is cancelled.
- **Step 1** (one training run): head no longer eats `cond`; does `u` still collapse?
- ⚠️ **Rejected experiment, recorded so it isn't re-proposed**: "shuffle `u` and see how
  much the score drops." Since `u` is already a constant, shuffling it is a no-op, so that
  test *must* return "no difference" — it looks like evidence and carries none.

### 12.7 Follow-ups not yet read

- `Encode, Think, Decode` — arXiv:2510.07358 (latent encoder → recursive thinking block →
  decoder; structurally closest to ours).
- `Parallel Test-Time Scaling for Latent Reasoning` — arXiv:2510.07745 (**latent reward
  model** picks trajectories ⇒ same idea as 主人's value-directed refine).
- `IterRef` — arXiv:2511.05562 (reward-guided noise↔denoise refinement for discrete
  diffusion ⇒ denoising-refine + value guidance combined).
- `A Survey on Latent Reasoning` — arXiv:2507.06203 (map of the field).
- `LaDiR` (already on the carry-over list) — latent diffusion reasoning, 30 denoise steps.

### 12.8 Step 0 RESULT (2026-08-24) — the bypass hypothesis was half wrong

`experiments/head_bypass.py` (CPU, seconds, reads checkpoints only — no data, no training).
Splits `ahead.net.0.weight` (shape 512 × (COND+DIM)) into its `cond` columns and its `u`
columns and reports per-dimension RMS, so the 256 vs 1024 width difference cancels.

```
                            cond/dim   u/dim   cond:u
[control] random init        0.01613  0.01614   0.999   <- control behaves correctly
12000 steps large-navigate   0.05906  0.04683   1.261
12000 steps large-stitch     0.05317  0.03496   1.521
24000 steps medium-navigate  0.08007  0.06642   1.206
                                       median   1.206  (n=21 checkpoints, range 1.01-1.52)
```

⛔ **The head did NOT learn to ignore `u`.** Per-dimension weight on `u` is only ~20% below
`cond`, and since `u` is 1024-wide against `cond`'s 256, the **total** Frobenius mass on the
`u` block is ~2.5× that of `cond` (2.25 vs 0.89 at large-navigate/12000).

⇒ **Retraction:** the 12.2 phrasing "the head can look at the eyes so it ignores the brain"
is wrong as stated. Keeping the original wording above deliberately — the corrected
mechanism only makes sense against it.

⭐ **Corrected mechanism.** Because `u` is constant, `W_u · u` is a *fixed vector* — it acts
as a **learned bias**. So there is no reason for the head to zero those weights; that block
is doing something useful, just not "reading `u`'s content".

⇒ The disease is not in the **weights**, it is in the **gradient**:
`∂l_refine/∂(refine params) = ∂l_refine/∂u · ∂u/∂(refine params)`. The head already drives
`l_refine` low using `cond`, so `∂l_refine/∂u` is small, so `refine` never feels pressure to
carry content and drifts to the cheapest solution (a constant).

⇒ **The action item is unchanged (① still first), but its justification changed**: not "the
mouth won't listen to the brain" but "**the brain is never forced to speak**".

⚠️ **Limitation of this probe, now known to be the important one**: first-layer weight mass
cannot distinguish "reads `u`'s content" from "uses `W_u·u` as a bias". Any follow-up must
measure *sensitivity to variation in `u`*, not weight magnitude.

**Next measurement (proposed, cost ≈ 0):** the gap between `l_anchor` (head fed the true
`e_target`) and `l_refine` (head fed the collapsed `u`) — that gap *is* "how much `u` adds
beyond what `cond` already supplies". Both are already printed every 1000 steps by
`scratch_lacot_rollout.py`, so it can be recovered from the 2026-08-23 `slurm_outputs/` logs
without re-running anything.

---

## 13. Official alignment + length audit (2026-08-24, 主人: 「缺口改掉，下次不要重新再被影響到」)

### 13.1 What OGBench actually prescribes (first-hand)

Source: OGBench repo `impls/agents/gcbc.py::get_config()`, `impls/utils/datasets.py::GCDataset.sample_goals()`,
and `impls/hyperparameters.sh` (the file that reproduces the paper's main table).

```
GCBC defaults:  actor_p_curgoal=0.0  actor_p_trajgoal=1.0  actor_p_randomgoal=0.0
                actor_geom_sample=False        <- UNIFORM, not geometric
uniform branch: d = rand();  goal = round(min(i+1, final)*d + final*(1-d))
                ⇒ uniform over [next step … end of this trajectory]

hyperparameters.sh, pointmaze — EVERY line (navigate and stitch, all six agents):
                --eval_episodes=50
  large-stitch: GCBC has NO override (plain defaults);
                GCIVL / GCIQL / QRL / CRL / HIQL add
                --actor_p_randomgoal=0.5 --actor_p_trajgoal=0.5
```

⭐ **The split matters**: only *value-bearing* methods use random goals on stitch. Plain GCBC does not —
a random goal has no path in the data, so its action labels are noise unless a value function can
judge whether that goal is worth pursuing. LaCoT's head is currently BC-shaped (no value in the loss),
⇒ **LaCoT aligns to the GCBC row**: uniform trajgoal, no random goals.

⚠️ **Correction**: `scratch_lacot_rollout.py` carried a comment claiming "官方 eval_episodes=20". That is
`impls/main.py`'s *flag default*; pointmaze's actual commands all override it to 50.

### 13.2 Length audit — `experiments/measure_lengths.py` (3 s, run 19723)

```
                        medium-navigate   large-stitch
dataset trajectories    1000 x 1001 步    5000 x 201 步
goal dist, ours geom     47.6 步 (mean)    37.8 步
goal dist, official uni 249.9 步          50.4 步        <- 5.3x vs 1.3x
oracle BFS path length   126 步            255 步
truncated at horizon=300   0%              20%   🚨
T_CAP=16 ⇒ train        每 3.0 步一點     每 2.4 步一點
T_CAP=16 ⇒ eval         每 7.9 步一點     每 15.9 步一點  <- 2.6x vs 6.6x
```

🚨 **Three findings, in order of severity:**

1. **20% of large-stitch's oracle answers were half a path.** The BFS walker hit its 300-step cap
   before reaching the goal, so for one題 in five the "correct answer" fed to the head was *wrong*.
   ⛔ Not a tuning issue — a bug. Very plausibly a major part of why large-stitch's ceiling sat at 0.343.
2. **T_CAP's real problem is sampling DENSITY, not size.** Train paths get a point every 2.4 steps;
   eval oracle paths every 15.9 ⇒ the encoder has never seen a path that sparse. ⇒ *This* is the
   OOD source, not "16 is small". ⭐ Note T_CAP resamples the whole path (endpoints always kept) —
   it never truncates.
3. **Switching to uniform goals helps the two envs very differently.** large-stitch's dataset is
   5000 *short fragments* (201 steps), so uniform's ceiling is right there (1.3x). medium-navigate
   is the big one (5.3x) — and decisive: our 47.6-step training goals never covered the 126 steps
   eval actually demands.

### 13.3 What changed (both `exp_etarget_ceiling.py` and `scratch_lacot_rollout.py`)

```
① goal sampling  geometric(0.02) → official uniform
② T_CAP          16 → 256   (LACOT_TCAP)
③ eval episodes  20 → 50    (official)
④ oracle horizon 300 → 800  (LACOT_ORACLE_HORIZON)
⑤ NEW: oracle self-reports its truncation rate at the end of every run
```

⛔ **Numbers produced after 2026-08-24 are NOT directly comparable to anything from 08-23 or earlier** —
three of the knobs above moved.

⭐ Per 主人's "下次不要重新再被影響到": each value is now the *code default* with the first-hand source
cited inline (which official file, which line) plus an explicit ⛔-don't-revert note — so the next
person to touch it reads the provenance before changing it. ⑤ exists because a half-path answer is
numerically plausible and invisible; the 20% was only found by measuring, and that shouldn't depend
on luck twice.

---

## 14. Value-guided refine — the update rule (2026-08-25, with 主人)

`[主人]`「Best of N不太好看，要讓u 的V本身就學會」
⇒ 裁示：**走梯度上升，不走 best-of-N。** best-of-N 只是「多抽幾張」，⛔ 不會「越想越好」。

### 14.1 The update rule (主人 2026-08-25 說這個形式比較好懂 ⇒ 之後一律用這個寫法)

```
u ← u + η · [ ∇u V(u) + λ · ∇u log p(u | s,g) ]

u                  現在手上這條計畫（想出來的軌跡的 latent）
η                  步長
∇u V(u)            羅盤 —— 往 return 高的方向
∇u log p(u|s,g)    結界 —— 往「更像真軌跡」的方向（flow 給的密度）
λ                  韁繩鬆緊
```

⚠️ 寫法備註（主人 8/25 兩次踩到）：
1. ⛔ 別寫成 `∇u [ V(u) + λ log p ]` —— 數學等價，但主人反映「中括號裡沒有微分項」，B 形式好讀。
2. ⛔ `∇u` 的 u 是**下標**，不是乘法。在純文字裡寫成 `(∂/∂u)[...]` 或 `grad_u` 較不會被讀成 `u × [...]`。

**為什麼兩項相加**：兩個都是「越大越好」⇒ 一起最大化 ⇒ 找一條*又好、又還像真的*路。
λ 太小 ⇒ 衝出 flow 沒看過的區域，u 解出來是垃圾；λ 太大 ⇒ 待在資料裡、不會比資料好。

⇒ **「多想幾輪」＝這個式子多跑幾次** ⇒ test-time scaling 從這裡長出來，⭐ 是結構給的，不是期待來的。

### 14.2 ⛔ 今天被推翻的兩條（ルナ提出、主人當場問掉）

1. **「量 V(s,g) 在牆附近準不準」** —— `[主人]`「他們的V不是我們的V」
   ⇒ M3 那顆 V(s,g) 走監督式路線時根本不在鏈路上，量它沒意義。
   （唯一剩的用途：當第二把尺驗 V(u)，尺不準就別用。）

2. **「讓 V 的梯度流回 encoder，把 latent 變 value-aligned」** —— `[主人]`「感覺怪怪的，encoder這邊指的是什麼…
   應該是在 value guided thinking過程中，自然就converge 出軌跡，這跟encoder什麼關系」
   ⇒ ① 那等於要解凍 `traj_enc` + `e_pooler`（主人 8/18 自己定的凍結）。
   ⇒ ② 🚨 更根本：**前提就不成立**。flow 學的是 p(e_target|s,g)，而 e_target 全部來自真的走過的軌跡
      ⇒ 撞牆的路不在資料裡 ⇒ flow 不會生它 ⇒「撞牆 vs 穿門在 latent 上相鄰」那個斷崖情境，
      在 flow 的支撐集內不會發生。
   ⇒ ⭐ 真正的風險不是斷崖，是**爬出去**（爬到 flow 沒生過的區域）⇒ 解法就是 14.1 的 λ·∇log p 那一項，
      ⭐ 也就是 Diffuser 的 classifier guidance 在做的事，**不用動 encoder**。

### 14.3 零件盤點 — 只缺一顆

```
log p(u|s,g)   ✅ 有 —— flow 本人（flow.nll ＝ −log p）
head           ✅ 有 —— ahead(cond,u) → 動作
兩個 ∇         ✅ 都是 network，autograd 直接給
V(u)           ⛔ 沒有 —— 要訓
```

### 14.4 ⚠️ 訓 V(u) 的未解設計問題：壞路從哪來

給定 (s,g)，資料集裡只有【一條】真實的路 ⇒ V 看不到「同一題的好答案 vs 壞答案」
⇒ 🚨 它可能只學會「這條路有多長」＝距離，⛔ 不是「這條路好不好」。

候選（⏳ 待主人裁示）：
- (a) 造負例：對真 e_target 加噪／換成別條軌跡的
- (b) ⭐ 用 stitch 那些隨機亂走的片段當天然壞例 —— 正是 `[主人]`「把stitch這些破碎的給用來協助reasoning」
- (c) 改成 V(u,s,g) 條件化 —— ⚠️ 但同一個 (s,g) 仍只有一條 u，對照還是沒有

### 14.5 實驗骨架

```
① 訓 V(u)，先驗排序對不對（pairwise ranking acc）
② 驗爬坡真的讓動作變好 —— 關鍵在對照組
③ 驗爬越多輪越好 ← 主人要的 test-time scaling
```

⭐ ②③ **完全在 inference 做、不用重訓** ⇒ 同一個 checkpoint 上比不同爬坡步數 ⇒ **配對比較**
⇒ 🚨 §① 記的那個 ±0.144 seed 噪聲在這裡咬不到我們。

### 14.6 Step 1 RESULT (2026-08-25) — (b) 的前提成立：stitch 天生就有「同一題多解」

`experiments/measure_stitch_multipath.py`（純 numpy，jasmine 上跑，30 萬個 (s,g) 對）
⭐ 分組用 `experiments/_verify_multipath_groups.py` **另一種寫法**（tuple key + dict，不共用 int64 編碼）
　 獨立重算過，組數／跨軌跡數／median std 完全對得上 ⇒ 不是編碼 bug。

**最有說服力的單一實例**（large-stitch, ε=0.5）：

```
同一題  s=(2.1,15.1) → g=(4.0,15.9)
8 條【不同軌跡】走過它，路長：61 62 66 92 113 135 140 142
⇒ 最長是最短的 2.3 倍 ⇒ 好路 vs 壞路的對照是【資料天生就有的】，不用造
```

**全體統計**（濾掉路長 <20 之後）：

```
large-stitch  ε=0.5
  跨軌跡多解組 11810，覆蓋 93.3%
  每組中位 5 條不同軌跡
  去重後路長 std 30.3（全體 42.6）  ← 降 29%：分組有效，但對照還在
  最長/最短 中位 2.05
  組內 (s,g) 散布 0.217 << ε        ← 防作弊檢查：同一題是真的
medium-stitch 同形狀
```

⇒ ⭐ **判讀的雙邊界**（⛔ 只看單邊會誤判）：
　 `std ≈ 全體` ⇒ 分組白做（那些「同一題」其實不是同一題）；
　 `std ≈ 0` ⇒ 同一題只有一種走法 ⇒ (b) 死。**兩個都要排除才算成立。**

**兩種要過濾的東西（主人 8/25「s,g 重複太高，可能還是要過濾」）**：
1. 同一條軌跡貢獻多個 (i,j) 到同一組 —— 那是同一條路的片段，不是多個解
   ⇒ 已在腳本內按 traj_id 去重；不去重會稀釋變異（ε=0.5 時 30.3 → 31.7 的差）。
2. s≈g 的「原地不動」題（路長 1~5 步）⇒ 用 `LACOT_MINLEN=20` 濾掉。
   ⇒ 濾掉後 ε=0.5 的變異從 28.5 升到 30.3（對照變乾淨）。

⚠️ **⛔ 一條被實測推翻的解釋（ルナ 8/25 提出、當天就自己驗掉）**：
　 ルナ本來把「ε=1.0 時 median std 塌到 3.9」歸因於上面第 2 點（原地題灌進來）。
　 ⇒ 🚨 **錯的**。濾掉之後只從 3.9 升到 5.5（medium 5.1），沒有回到 30 這個量級。
　 ⇒ ⏳ **ε=1.0 的塌陷至今沒有解釋**，而且兩個環境數值一致到不像噪聲（55% 的組 std<5）。
　 ⇒ ★ 它不擋路（我們用的是 ε=0.5），但⛔ 別把它當成已解釋的東西帶走。

---

## 15. Step 1 實測 + 前人 survey（2026-08-26，with 主人）

_★ 今天一整天在問同一件事：**把軌跡壓成 u 之後，還讀得出「這條路好不好」嗎？**_
_★ 答案是「讀得出一點，但只有該有的兩成」，而且我們把原因一格一格排除到只剩一個。_

### 15.1 主人今天的三個設計輸入

1. `[主人]`「不用絕對長度也沒差」⇒ **實測成立**：`w_mse=1` 0.569 vs `w_mse=0` 0.575，CI 重疊。
   ⭐ 而且 `w_mse=0` 還**更乾淨**：只有它的真標籤顯著贏過打亂標籤（0.557 > 0.539），
   `w_mse=1` 兩者 CI 重疊。⇒ 理由：MSE 的打亂只洗牌 batch 內配對，V 仍能從長度分布撈到東西。
2. `[主人]`「0.5 到 0.8 區間太小，不太適合實際使用」⇒ 已加**有物理單位**的指標：
   拿 V 去選路，比隨機挑少走幾步、離最好的還差幾步。⇒ 單位是「步」。
3. `[主人]`「我比較想的是 contrastive 把負樣本推遠 embedding」
   ⇒ 動的是 **u 的空間本身**，⛔ 不是在 V 外面掛判斷。
   🚨 為什麼更根本：refine 就是在 u 空間爬坡 ⇒ 空間的幾何本身該有方向。

### 15.2 🚨 兩個「修好了其實沒修好」的洩漏（今天最重要的工程教訓）

```
8/25  padding mask 上寫著「這條路有幾個點」      ⇒ 改成固定 T_FIX 個點
8/26  🚨 但 linspace().round() 在 L+1 < T_FIX 時索引【重複】
      ⇒ 不重複的點數 ＝ min(L+1, T_FIX) ＝ 長度本身
      ⇒ [實測] 光數點數：T=64 拿 0.898、T=128 拿 0.993、T=201 拿 1.000
```

⇒ 修法：`linspace` 之後改**插值**不取整 ⇒ 對照⑤ 從 0.993 掉到 0.499。
⇒ ⭐ 並把「數不重複的點」做成**常設對照⑤**（直接數 `make_segments` 真的吐出來的點，
　 ⛔ 不用公式 `min(L+1,T)` 代替 —— 那只對取整版成立，換個取樣法就失效）。
⇒ ⭐ 舊取樣法留成 `LACOT_SAMPLE=round`，⛔ 不讓它變成消失的歷史。

### 15.3 一格一格排除（⇒ 只剩 encoder 的訓練目標）

```
✗ 題目本身難      尺②（幾何長度，不用學）T=201 拿 0.978
✗ 訓練量不夠      encoder/V 各加 10 倍 ⇒ 0.611 → 0.629（+0.018），C 組還開始過擬合
✗ 容量不夠        K=1/2/4/16 ⇒ 0.561/0.581/0.596/0.570，加大反而變差
✗ 讀法不對        pairwise ranking loss 加了，主指標沒有跳
✗ V 看不到 (s,g)  加了 0.596 vs 0.591，在噪聲內
⇒ 只剩：encoder 的【訓練目標】從頭到尾只要求它「對得上 s,g」
```

⭐ **關鍵佐證（⇒ 這格最有說服力）**：`T_FIX=32` 時上限只有 0.532、而 V(u) 拿 0.524
　 ⇒ **走完全程 77%**。資訊少的時候 encoder 幾乎把能讀的都讀走了；
　 資訊一多（T=128 上限 0.973）就只讀走 19%。⇒ 不是「讀不到」，是「讀不完」。

⭐ **u 塌掉的直接證據**：有效維度（participation ratio）只有 0.092
　 ⇒ 1024 維裡實際只用約 6 個方向（沒訓練時 3 個）。

### 15.4 兩個失敗的嘗試（⛔ 別重做，但理由要帶著）

1. **「別題的路」當 V 的負樣本** ⇒ 主指標 0.596 → 0.563，「省幾步」4.5 → 1.1。
   原因：負樣本太好認（辨識率 0.994）⇒ V 把容量花在簡單任務上，排擠掉難的。
   ⚠️ 而且對照組 D（有負樣本、不給 s,g）拿到 0.800 —— 它**應該**失敗卻沒有
   ⇒ `[推論 未驗]` 正樣本從同題多解組抽、負樣本從全部的路抽 ⇒ **兩個池子不同**
   ⇒ V 可以靠「你從哪個池子來」分辨。★ 這正是放一個「應該失敗的對照」的價值。
2. **(s,g) 當 query（主人提案）** ⇒ 主指標 0.596 → 0.555（CI 不重疊），「省幾步」4.5 → 0.2。
   `[推論 未驗]` Perceiver 每層是殘差的（`q ← q + attn(q, ctx)`）⇒ query 帶著 (s,g)
   會一路殘留到輸出 ⇒ u 對 (s,g) 的 R² 從 0.920 升到 **0.999** ⇒ 容量反而被佔得更死。
   ⏳ **未試的修法**：輸出時把 query 本身扣掉（`u = out − query_in`）。
   ⚠️ `lacot/e_target.py` 的 `PerceiverPooler.forward` 已加選填 `queries` 參數（純增量，不傳＝行為不變）。

### 15.5 ⚠️ 探針自己踩到的兩個坑（⇒ 指標本身也要被檢驗）

1. **欠定發散**：u 有 1024~4096 維、訓練樣本只有 3308 筆 ⇒ lstsq 的 held-out R² 跑到 −14872。
   ⭐ 好在爛得夠明顯，**要是吐出 0.6 這種合理的數字就騙過去了**。
   ⇒ 修法：先用**訓練集**的 PCA 降到樣本數的 1/8 以下。⛔ 投影矩陣不能用全體算，那是洩漏。
2. 🚨 **「L 的 R² 變高」是假綠燈**：它是**全體**的可讀性，而 u 含有 (s,g)、
   (s,g) 的直線距離本來就跟步數相關 ⇒ 提升可能全來自 (s,g)。
   ⏳ **待改**：改成 partial —— 先用 (s,g) 預測 L 取殘差，再問 u 能不能預測那個殘差。

### 15.6 前人 survey（2026-08-26）

| 問題 | 答案 | 出處 |
|---|---|---|
| 有人在 latent 上做品質排序嗎 | 有。latent reward model（2 層 Transformer 分類器，標籤只是「答案對不對」）ROC-AUC 0.88~0.99 | LTO, arXiv 2509.26314 (ICLR 2026) |
| encoder 該用什麼目標 | JEPA/預測 + **防塌正則**；⛔ 明確反對重建 | PLDM, arXiv 2502.14819；LeWorldModel, arXiv 2605.08732 |
| 有人做 latent 的 value 爬坡嗎 | 有，`z ← z + η∇_z J(z)`（REINFORCE），但**沒有密度項**、承認會漂到不可讀 | LatentSeek, arXiv 2505.13308 |
| Diffuser 那批怎麼對軌跡打分 | 獨立訓一個 reward model 預測軌跡累積 reward，梯度注入 reverse sampling ⇒ **在原始軌跡空間，不是 latent** | Diffuser / Decision Diffuser |

⭐⭐ **我們的定位（⇒ 這段是貢獻論述的骨架）**：

```
LatentSeek   有羅盤、沒結界 ⇒ 論文自承會漂到語意不通
LTO          有 KL 結界、但用拒絕採樣 ⇒ 不會「越想越好」（＝主人不要的 best-of-N）
Diffuser 那批 value guidance 做在【原始軌跡空間】
⇒ 我們        latent 空間 ＋ 梯度爬坡 ＋ flow 給的密度結界
            主人的 λ·∇log p 正好是他們各自缺的那一項
```

🚨 **PLDM 的消融（⇒ 這是「防塌正則不是裝飾」的證據）**：完整 98.0% / 拿掉 variance 13.4% / 拿掉 covariance 29.2%。

⚠️ **一個要記住的警訊**（LeWorldModel）：pairwise IDM 在 oracle 上 R²=0.993 卻只有 34% 成功；
　 GC-IDM 訓練 R² 只有 0.20 卻 100% 成功。⇒ **代理指標會騙人**，而我們一直在追排序準確率。

### 15.7 ⏳ 待辦（按優先序）

1. **補防塌正則**（variance + covariance，VICReg 那套）⇒ 最便宜、最有文獻支持。
2. **探針改 partial R²**（先扣掉 (s,g) 能解釋的部分）。
3. **(s,g) 當 query ＋ 輸出扣掉 query** ⇒ 看主指標會不會回來。
4. 難負樣本（起點接近、終點不同）＋ 正負樣本同池抽樣 ⇒ 修 15.4-1 的兩個洞。
5. ⏳ 第二輪 survey：VICReg 在軌跡表示上的具體用法；GCRL/OGBench 上有沒有人做 test-time refinement。

_⛔ 已撤回：「用重建目標逼 u 保留形狀」—— PLDM 附錄 G 有實證說重建特徵在規劃上更差。_
