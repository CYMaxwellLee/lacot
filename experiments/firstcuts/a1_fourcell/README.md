**在證明什麼、判準是什麼**：證明「E(τ) 教師碼帶路線資訊、且 action head 會跟隨」——
判準（預註冊）：定向差 S_c 在配對 bootstrap 95% CI 下 > 0〔拍〕（head 動作與碼所屬
路線的合法決策一致率，swap 後跟碼走而非跟條件走）；第 0 步關卡：合格分岔對 n ≥ 30
〔拍〕，不足即停下回報改題（⛔ 不硬跑小樣本）。

# A1：教師編碼四格 swap

This directory implements WORK-A1-fourcell. The first operation is always the
step-zero gate. If it finds fewer than 30 qualified route pairs, `run_a1.py`
prints `STOP AND REPORT` and does not load a model or run four-cell/read-head
forwards. There is no small-sample override.

## Fixed gate and pairing

`step0_count.py` calls OGBench's `make_env_and_datasets` and reads the official
validation split. The maze itself comes from `env.unwrapped.maze_map`; passable
maze cells and their XY centers come from that official map and
`env.unwrapped.ij_to_xy`. BFS uses the shared `lacot.subgoal.grid_bfs`.

A candidate is an observed validation-trajectory segment whose consecutive-
deduplicated maze-cell route is a BFS shortest route. Its ordered start/goal
cell pair must have at least two equally short BFS next steps, and the recorded
route must start with one of them. A matched pair has the same ordered start and
goal cells, different first shortest-path choices, different source episodes,
and start and goal XY endpoints each at most 2.0 environment units apart. A
source episode can be selected only once. The route marker is the ordered maze
corridor cell-ID sequence `row * maze_width + column`; continuous XY samples do
not define route identity. Consecutive duplicate cells are removed with
`lacot.intent.traj_to_cells`.

The condition-radius, shortest-route requirement, and episode-disjoint matching
are fixed in `step0_count.py`. If the gate is below 30, the approved next
decisions are to broaden the condition scale or use synthetic branch starts;
the script does neither by itself.

### First local gate run (2026-09-27)

The official `pointmaze-large-stitch-v0` validation split has 500 episodes and
100,000 rows. Its OGBench maze is 9×12 with 46 passable cells. The route
diversity distribution is `{1: 8}`: eight endpoint-cell groups had multiple
BFS-shortest next steps, but each had only one observed shortest route in val.
The strict preregistered count is `n_qualified_pairs=0`, so the implementation
stopped before any four-cell or read-head forward. The wiring command reports
`STOP_SKIP` for the same reason. No small-sample result was run or treated as a
PASS.

## Model interface and weights

The local default checkpoint is:

`results/ckpt_large-stitch_self_K8_c256_ch4_st8000_T128_ep2_gu_eorecon_ictr_tch0.5_emw0.999_norf_cd0.1_bci_s0.pt`

It is a local `recon_ictr` checkpoint with K=8, D_MODEL=256, COND=256,
CHUNK=4, T_CAP=128, no intent adapter, and no VQ. The scripts accept
`--ckpt PATH` for another compatible checkpoint. The default loads the saved
raw weights; `--ema` explicitly selects saved EMA weights for the condition
encoder, flow, and action heads while keeping the frozen saved trajectory
encoder.

The interfaces were read from the requested repo sources:

- `experiments/scratch_lacot_rollout.py`: `etarget(traj, mask)` returns
  `[B,K,D_MODEL]`; `condvec(s,g)` applies `cond_enc` to s and g, concatenates,
  then applies `cond_head`; `ahead(cond,u)` returns `[B,CHUNK,ADIM]`;
  `bc_head(cond)` returns `[B,CHUNK,ADIM]`.
- `lacot/e_target.py`: `PerceiverPooler(frame_feats, key_padding_mask=...)`
  returns frozen e tokens. A true route is time-resampled to the checkpoint's
  fixed T_CAP and normalized with the training split's XY mean/std before
  `traj_enc` and `e_pooler` run.
- `lacot/nf_head.py`: `Flow.sample(n, cond, generator=...)` draws a standard
  normal base and applies the reverse blocks. The comparator `F(cond,0)` uses
  that same reverse path with an all-zero base input. It is named **base zero
  point inverse image** in this code and output.
- `lacot/intent.py` and `lacot/subgoal.py`: adjacent duplicate route cells and
  the shared BFS implementation are reused.

All forwards use trained checkpoint weights. No optimizer or training loop is
constructed.

## Four cells, scoring, and read heads

For each matched pair, the four cells are `H(c_A,e_A)`, `H(c_A,e_B)`,
`H(c_B,e_A)`, and `H(c_B,e_B)`. The `cond` inputs stay in their original
positions; only the two frozen E(τ) tensors are exchanged. The head's first
predicted XY action is mapped to the closest cardinal maze direction using the
official `ij_to_xy` transform.

The recorded route supplies its own first branch choice. BFS supplies the full
legal-next-step set: every neighbor whose distance to the goal is one less is
legal, so all equally short choices count and no deterministic BFS tie-break is
used. Each cell reports both exact e-route match and membership in this legal
set.

The paired directional score is fixed as

```text
S_c(pair) = 1/2 * [match(H(c_A,e_B), route_B) - match(H(c_A,e_B), route_A)
                 + match(H(c_B,e_A), route_A) - match(H(c_B,e_A), route_B)]
S_c = mean over qualified pairs of S_c(pair)
```

This checks the swapped outputs against the route carried by e and against the
route associated with the held-fixed c. The confidence interval is a paired
percentile bootstrap over pairs, 10,000 draws, seed 20260927. The
**PREREGISTERED** decision is `n_pairs >= 30` and the 95% CI lower bound for
`S_c` greater than zero.

The same pair set is evaluated with four already-trained read heads:
`bc_head(cond)`, `ahead(cond,E(τ))`,
`ahead(cond,F(cond,0))` where `F(cond,0)` is the **base zero point inverse
image**, and `ahead(cond,F(cond,ξ))` with a seeded standard-normal ξ.

## Run

Use the repository virtual environment:

```bash
.venv/bin/python experiments/firstcuts/a1_fourcell/step0_count.py
.venv/bin/python experiments/firstcuts/a1_fourcell/run_a1.py
.venv/bin/python experiments/firstcuts/a1_fourcell/test_a1_wiring.py
```

After the gate passes, `run_a1.py` defaults to a smoke of at most eight pairs.
`--full` runs the full gated table and is intended for the later machine run.
The wiring script first applies an e-ignoring mutation and requires the
two-direction assertions to fail on it, then checks the trained forward path:
`H(c_A,e_A)` / `H(c_B,e_B)` route match must exceed route-choice random, while
replacing e with all-zero vectors must bring route match within 0.25 of random.
If the gate is below 30, the wiring script explicitly reports `STOP_SKIP`, not
PASS.

An optional `--dataset-dir DIR` selects the OGBench dataset directory. The
scripts also honor `OGBENCH_DATA_DIR` and discover the common local data paths.
`--out FILE` writes the run output as JSON.
