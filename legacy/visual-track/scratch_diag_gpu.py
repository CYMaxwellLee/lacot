"""Diagnose GPU (EGL) render non-determinism for visual-pointmaze.
Renders the same rows twice, reports which rows/pixels differ and by how much.
Env-driven: set MUJOCO_GL / CUDA_VISIBLE_DEVICES / MUJOCO_EGL_DEVICE_ID before running.
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

print("MUJOCO_GL=", os.environ.get("MUJOCO_GL"),
      "| CUDA_VISIBLE_DEVICES=", os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
      "| MUJOCO_EGL_DEVICE_ID=", os.environ.get("MUJOCO_EGL_DEVICE_ID", "<unset>"))

env = build_pointmaze_pixel_env("medium")
N = 120
A = np.stack([render_row(env, qpos[t], qvel[t]) for t in range(N)])
B = np.stack([render_row(env, qpos[t], qvel[t]) for t in range(N)])
env.close()

diff = np.abs(A.astype(int) - B.astype(int))
per_row_px = (diff.max(axis=-1) > 0).reshape(N, -1).sum(axis=1)  # differing pixels per row
bad = np.flatnonzero(per_row_px > 0)
print(f"rendered {N} rows twice | image shape {A.shape[1:]}")
print(f"differing rows: {len(bad)}/{N}  -> {bad.tolist()[:20]}")
if len(bad):
    print(f"  max abs pixel diff (all rows): {int(diff.max())}")
    nz = diff[diff > 0]
    print(f"  diff magnitude on changed values: min {int(nz.min())}, mean {nz.mean():.2f}, max {int(nz.max())}")
    print(f"  per differing row, # pixels changed: {per_row_px[bad].tolist()[:20]}")
    r = int(bad[0])
    ys, xs = np.where(diff[r].max(axis=-1) > 0)
    print(f"  row {r}: {len(ys)} px differ; sample coords {list(zip(ys.tolist()[:6], xs.tolist()[:6]))}")
else:
    print("  >>> FULLY DETERMINISTIC on this config <<<")
