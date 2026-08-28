"""量「資料步數跨度」跟「BFS 任務難度」的關係 —— span probe 的判讀前提。

🚨 為什麼要量：MINSPAN 探針把「跨度 ≥64 資料步」當分布外，但 stitch 軌跡是亂走的，
   88 步的軌跡起終點可能就在隔壁 ⇒ 時間跨度分布外 ≠ 空間難度分布外。
   這支直接量三個分布的 BFS 格距：訓練抽法 / MINSPAN 抽法 / dev 尺各 tier 的題目。
輸出：console 表（等寬）。CPU、~30 秒。
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lacot import dev_eval as DE
import ogbench

OGB_DATA = os.environ.get("OGBENCH_DATA_DIR", "/archive/cymaxwelllee/data/ogbench")
ENV_NAME = os.environ.get("LACOT_ENV", "pointmaze-medium-stitch-v0")
MINSPAN = int(os.environ.get("LACOT_PROBE_MINSPAN", 64))
NPAIR = int(os.environ.get("LACOT_NPAIR", 2000))
rng = np.random.default_rng(0)

d = np.load(f"{OGB_DATA}/{ENV_NAME}.npz")
OBS = np.asarray(d["observations"], np.float32)
TERM = np.asarray(d["terminals"], bool)
N = OBS.shape[0]; ends = np.flatnonzero(TERM); starts = np.concatenate([[0], ends[:-1] + 1])
traj_end = np.empty(N, np.int64)
for s0, e0 in zip(starts, ends):
    traj_end[s0:e0 + 1] = e0

env, _, _ = ogbench.make_env_and_datasets(ENV_NAME, dataset_dir=OGB_DATA)
cells = DE._passable_cells(env)
cell_xy = np.array([env.unwrapped.ij_to_xy(c) for c in cells], np.float64)


def xy_to_cell(xy):
    return cells[int(np.argmin(((cell_xy - np.asarray(xy[:2])) ** 2).sum(1)))]


# 每個格出發的 BFS 表做快取（迷宮才幾百格，全算也行）
_bfs_cache = {}


def bfs_d(a, b):
    a, b = tuple(a), tuple(b)
    if a not in _bfs_cache:
        _bfs_cache[a] = DE._bfs_from(env, a)
    return _bfs_cache[a].get(b, -1)


def sample_pairs(minspan):
    ds, spans = [], []
    while len(ds) < NPAIR:
        r = int(rng.integers(0, N)); te = int(traj_end[r])
        if te - r < max(8, minspan):
            continue
        if minspan > 0:
            gr = int(rng.integers(r + minspan, te + 1))
        else:
            _d = rng.random()
            gr = int(round(min(r + 1, te) * _d + te * (1 - _d)))
            gr = max(gr, min(r + 8, te))
        dd = bfs_d(xy_to_cell(OBS[r]), xy_to_cell(OBS[gr]))
        if dd >= 0:
            ds.append(dd); spans.append(gr - r)
    return np.array(ds), np.array(spans)


def rows(tag, ds, spans=None):
    q = lambda a, p: float(np.percentile(a, p))
    sp = f"  步數 p50 {int(np.median(spans)):3d}" if spans is not None else ""
    print(f"  {tag:<18s} BFS p50 {q(ds,50):5.1f}  p90 {q(ds,90):5.1f}  max {ds.max():5.0f}{sp}",
          flush=True)


print(f"env={ENV_NAME}  cell_w={DE.cell_width(env):.2f}", flush=True)
d_tr, sp_tr = sample_pairs(0)
d_ms, sp_ms = sample_pairs(MINSPAN)
print("抽法：", flush=True)
rows("訓練（官方）", d_tr, sp_tr)
rows(f"MINSPAN={MINSPAN}", d_ms, sp_ms)
print("dev 尺題目：", flush=True)
tasks = DE.build_dev_tasks(env, n_per_tier=80, n_tiers=3, seed=0)
for t in range(3):
    td = np.array([tk["bfs_dist"] for tk in tasks if tk["tier"] == t])
    rows(f"tier{t}", td)
