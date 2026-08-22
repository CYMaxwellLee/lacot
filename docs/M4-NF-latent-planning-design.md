# M4 — NF Latent Reasoning for Trajectory Planning (design)

_Owner: Chun-Yi (主人). Drafted with Luna, 2026-08-17. Status: DESIGN (implementation-first, build block-by-block)._
_This document is the durable record of the M4 idea born on 2026-08-17. Build from this._

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

## 2. The pivot experiment (decides whether M4 is worth it)

**Does M2 (oracle, TRUE futures) beat the floor?**
- **oracle > floor but M3 ≈ floor** → the ceiling exists; the problem is *generating* good futures → **M4 (latent, no image generation, no self-copy) is the right fix.** ✅
- **oracle ≈ floor** → the "future context helps" premise itself is weak → M4 (same premise, different substrate) may not help; rethink.
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
- **Components, all matched-capacity to the SOTA baselines (HIQL/CRL), not inherited from any prior in-house model**: 👁 image encoder = the SOTA visual baseline's small CNN (deployed); 🧠 a conditioning backbone (matched-capacity transformer over the latent tokens); 👄 action head (chunk_len 8 × action_dim 5 × 256 bins). **M4's core new block = the "generate u" NF head** (TARFlow engine) + the training-only flow-target trajectory encoder.

## 5. Methodology (主人 2026-08-17): block-by-block

Build one block, **validate it (can sample / compute likelihood / is stable) before moving up**. Do not stack everything then debug. Measure-first at every step.

**Build order:**
1. Minimal "generate `u`" block using the TARFlow `Flow` class — validate: samples a latent sequence, computes exact log-likelihood, is stable.
2. Add the flow-target: encode the future trajectory → e_target; wire `L_NF` (u aligned to e_target). Validate.
3. Add the action head decode from `u` + `L_action`. Validate the end-to-end BC works.
4. Add the iterative refinement (a few rounds) + `L_consistency`. Validate convergence/stability.
5. Wire the full model together (encoder → NF head → backbone → action head). Two-stage training (§7).
6. Eval: does it beat the floor? (the M4 test).

## 6. References to explore (主人 2026-08-17)

- **World Value Model (WVM)** — arXiv:2606.24742, "World Value Models for Robotic
  Manipulation" (Wang et al., 2026-06); READ 2026-08-19. NOT a latent->action policy
  (the 2026-08-17 note was imprecise): it is a **value model** that marries a **world
  model with value estimation** for deep temporal understanding, claims SOTA
  Value-Order-Correlation, and stays reliable on **suboptimal / mixed-quality data**
  (its own Suboptimal-Value-Bench), sim + real manipulation. **Relevance to M4**: it
  targets the exact pain point that pushed us value-free -- value estimation being
  fragile on mixed-quality data. **Role for M4 (主人 2026-08-20): WVM is a BASELINE to
  beat -- add it to the baseline set.** M4 (value-free latent planning) aims to beat WVM
  (world-model + value) on the shared eval. (Enter it via its own policy/value readout;
  pin the exact comparable form when baselines are built.) It also stays a candidate
  critic if M4 later adds the optional AWR (§7), but its primary role is now a baseline. *(abstract-level read; full method / VOC definition
  / training not yet read.)*
- **Swappable NF engine**: the "generate `u`" block is **modular**. TARFlow is one engine; could swap for a different normalizing flow, or **Kaiming He's drift / flow-matching family** (often simpler/faster). Plan: build+validate with TARFlow first (has code), then try swapping the engine. *(find the exact Kaiming He "drifting model" ref tomorrow.)*

## 7. NF-CoT precise method (implementation reference, from arXiv:2606.06447)

- **Compression**: frozen VAE encoder turns explicit CoT `d_{1:L}` → continuous codes `e_{1:K} ∈ ℝ^{K×D}` (K=64 slots, posterior mean as target). Shallow flow blocks `F_θ` (5 MetaBlocks, alternating Identity/Flip, near-identity init, triangular Jacobian): `u_{1:K} = F_θ(e_{1:K}; q)`. VAE frozen; shallow blocks + projections trainable.
- **NF head (causal Gaussian AR density)**: `p_θ(u_{1:K}|q) = ∏_i N(u_i; μ_θ(q,u_{<i}), diag(σ²_θ(q,u_{<i})))`. Backbone + NF head outputs μ, σ². Sampled left-to-right.
- **Exact likelihood**: `log p_θ(e_{1:K}|q) = log p_θ(u_{1:K}|q) + log|det J_{F_θ}(e;q)|`.
- **Objective**: `L_sup = λ_flow · L_flow + λ_text · L_text`, with `L_flow = -log p_θ(e|q)`, `L_text = -Σ_j log p_θ(x_j|q,u,x_{<j})`. Small Gaussian noise on `e` before the flow loss (robustness). *(For M4: L_text → L_action; the CoT target → the trajectory target; add L_consistency + iterative rounds.)*
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
  which supports our reverse-flip default for the (likely smaller) M4 K; revisit a
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
  a PURE, L_NF-only flow. `wpm/models/m4_smoke.py`.)*
- **Policy-awareness enters at the REFINE loop, not the flow.** The refine
  operator reads (s,g,u) and is deep-supervised by L_action, so it POLISHES the
  flow's imagined u^0 into an action-decodable u*. Clean division of labour:
  **flow imagines the trajectory (pure generative); refine + action polish it to
  be executable.** u is not policy-blind — the polishing just happens one layer up.

Concretely under **F=identity** (current build): the action gradient does not
reach the flow at all (u_target = e_target is frozen; the refine u^0 is a no-grad
sample). VERIFIED by `wpm/models/m4_gradroute_smoke.py` (5/5): L_NF → flow only;
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
  the future generator fails on cube-double). M4 sidesteps both: the future is
  latent (no pixel generation), and the core losses (flow NLL + action BC +
  consistency) need no value. ✅ Start value-free (pure hindsight BC).
  - ⚠️ Honest caveat: pure BC clones average behavior; if the data has bad paths
    it can learn mediocrity. The §7 objective keeps an OPTIONAL AWR weight
    (`w_t = clip(exp((V(s_{t+H}) - V(s_t)) / β))`) to upweight better-than-average
    actions. Add it only if pure BC underperforms (measure-first) — do NOT carry
    M3's value burden by default.

### Fairness of the comparison, and the engine-swap extension (主人 2026-08-18)

- **Is value-free M4 vs value-based baselines unfair? No — the opposite.** M4
  forgoes the value function (a source of better-than-BC improvement), so it
  fights HIQL/GCIQL with one hand tied; winning value-free is a STRONGER claim,
  not an unfair advantage. Benchmarks compare whole methods (HIQL = hierarchy +
  value, CRL = contrastive, GCBC = pure BC already differ wildly and are compared
  directly). The fairness that matters is same-data / same-eval / same-tuning-
  budget (already required by the eval rules), not same-components. To be airtight:
  report M4 both value-free AND with the optional §7 AWR, plus a GCBC floor on the
  SAME hindsight data (isolates the latent-reasoning mechanism from the data).
- **Engine-swap extension** (主人): the "generate u" block is modular (§6) — swap
  TARFlow for flow matching / rectified flow / Kaiming-He drift models. Often
  simpler + faster to sample than an AR flow, and SoTA-strong. Trade-off: the flow
  loss changes from exact-likelihood NLL to a velocity-field regression (MSE, often
  easier to train); exact likelihood then needs ODE integration (pricier), but M4
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
- ⭐ Calibrate to SOTA, not WPM (主人 2026-08-19): M4 is its own thesis, compared
  against the real SOTA (HIQL / CRL / WVM — WVM added as a baseline to beat, 主人 2026-08-20). The image encoder should match THEIR encoder
  (OGBench's visual-baseline small CNN), NOT be inherited from WPM just because the
  code already has one. WPM is potentially a separate paper; do not let its internal
  choices steer M4's comparison ("don't get led by WPM").
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
