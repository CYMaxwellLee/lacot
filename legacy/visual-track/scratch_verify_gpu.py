"""VERIFY the GPU (EGL) render non-determinism is ONLY tiny (<=1 level) edge jitter.
Renders a large + scattered sample twice; the verdict is the GLOBAL max abs diff.
If max abs diff > 1  -> it is NOT just tiny; do NOT loosen the check.
"""
import os, sys
import numpy as np

sys.path.insert(0, "/archive/cymaxwelllee/fpo")
os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco
from wpm.eval_envs import build_pointmaze_pixel_env

# 資料位置：預設走官方 OGBENCH_DATA_DIR，沒設才用本機 archive
OGB_DATA = os.environ.get("OGBENCH_DATA_DIR", "/archive/cymaxwelllee/data/ogbench")


def render_row(env, qpos_row, qvel_row):
    u = env.unwrapped
    data = u._data if hasattr(u, "_data") else u.data
    model = u._model if hasattr(u, "_model") else u.model
    data.qpos[:] = qpos_row
    data.qvel[:] = qvel_row
    mujoco.mj_forward(model, data)
    return np.asarray(u.get_ob(), dtype=np.uint8)


DATA = f"{OGB_DATA}/pointmaze-medium-navigate-v0-val.npz"
with np.load(DATA) as d:
    qpos = np.asarray(d["qpos"], dtype=np.float64)
    qvel = np.asarray(d["qvel"], dtype=np.float64)
n_rows = qpos.shape[0]

# (a) contiguous first 1500 rows (covers traj-0 that failed + into traj-1)
# (b) scattered across the whole val set
contiguous = list(range(min(1500, n_rows)))
scattered = list(range(0, n_rows, max(1, n_rows // 200)))
rows = sorted(set(contiguous) | set(scattered))
print(f"total val rows: {n_rows}; verifying {len(rows)} rows (contig 1500 + scattered)")

env = build_pointmaze_pixel_env("medium")
A = np.stack([render_row(env, qpos[t], qvel[t]) for t in rows])
B = np.stack([render_row(env, qpos[t], qvel[t]) for t in rows])
env.close()

diff = np.abs(A.astype(int) - B.astype(int))
maxd = int(diff.max())
changed_px_per_row = (diff.max(axis=-1) > 0).reshape(len(rows), -1).sum(axis=1)
n_bad_rows = int((changed_px_per_row > 0).sum())
worst_row_px = int(changed_px_per_row.max()) if len(rows) else 0
total_changed_px = int((diff.max(axis=-1) > 0).sum())

print(f"differing rows: {n_bad_rows}/{len(rows)}")
print(f"worst row: {worst_row_px} px changed (out of 4096)")
print(f"total changed pixels across sample: {total_changed_px}")
print(f"GLOBAL MAX ABS DIFF = {maxd}")
if maxd <= 1:
    print("VERDICT: ONLY <=1-level edge jitter across the whole sample. Safe to loosen to tol=1.")
else:
    print(f"VERDICT: found a {maxd}-level difference -> NOT just tiny. Do NOT loosen.")
