import os
"""Probe the pointmaze .npz structure to rule out data-handling bugs (per 主人).
Key question: is a TERM-split 'trajectory' one clean s->goal segment, or a 1001-step
episode holding many sub-goals? And do hindsight (s,g) paths stay direct or wander?"""
import numpy as np

# 資料位置：預設走官方 OGBENCH_DATA_DIR，沒設才用本機 archive
OGB_DATA = os.environ.get("OGBENCH_DATA_DIR", "/archive/cymaxwelllee/data/ogbench")
ENV = "pointmaze-medium-navigate-v0"
d = np.load(f"{OGB_DATA}/{ENV}.npz")
print("keys:", list(d.keys()))
OBS = np.asarray(d["observations"], np.float32); TERM = np.asarray(d["terminals"], bool)
N = OBS.shape[0]; ends = np.flatnonzero(TERM); starts = np.concatenate([[0], ends[:-1] + 1])
lens = ends - starts + 1
print(f"N={N}  n_episodes(TERM-split)={len(ends)}  ep-len min/med/max={lens.min()}/{int(np.median(lens))}/{lens.max()}")
if "masks" in d.keys():
    print("has 'masks' (OGBench success flags):", np.asarray(d["masks"]).shape, "sum", int(np.asarray(d["masks"]).sum()))
# characterise hindsight paths: for geometric goals, is the recorded s->g path direct?
rng = np.random.default_rng(0)
traj_end = np.empty(N, np.int64)
for s0, e0 in zip(starts, ends):
    traj_end[s0:e0 + 1] = e0
ratios = []  # path length / straight-line dist ; >>1 = wandering (crosses sub-goals)
overshoots = []  # does the path pass much closer to g before the end? (sign of sub-goal cross)
for _ in range(3000):
    r = int(rng.integers(0, N)); te = int(traj_end[r])
    if te - r < 20:
        continue
    gr = min(r + int(rng.geometric(0.02)), te)
    if gr - r < 10:
        continue
    path = OBS[r:gr + 1]
    straight = np.linalg.norm(path[-1] - path[0]) + 1e-6
    plen = np.linalg.norm(np.diff(path, axis=0), axis=1).sum()
    ratios.append(plen / straight)
    dists = np.linalg.norm(path - path[-1], axis=1)
    overshoots.append(dists.min() < 0.5 and np.argmin(dists) < len(dists) - 5)  # got to g early then left
ratios = np.array(ratios)
print(f"hindsight s->g (geom p=0.02): path/straight ratio med {np.median(ratios):.2f} p90 {np.percentile(ratios,90):.2f}  (1=direct, big=wander)")
print(f"  frac where path reaches g early then leaves (sub-goal cross): {np.mean(overshoots):.2f}")
# goal distance distribution (how far are geometric goals in steps)
gaps = []
for _ in range(3000):
    r = int(rng.integers(0, N)); te = int(traj_end[r])
    if te - r < 5:
        continue
    gaps.append(min(int(rng.geometric(0.02)), te - r))
print(f"hindsight goal step-gap: med {int(np.median(gaps))} p90 {int(np.percentile(gaps,90))} (vs ep-len {int(np.median(lens))})")
