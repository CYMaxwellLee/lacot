"""把 dev 尺的題目（含 BFS 參考路徑）dump 成 npz，給探針用 —— 探針維持不碰 env。

🚨 為什麼要這支：exp_span_difficulty 實測「資料步數跨度 ≠ BFS 難度」——
   訓練抽法 BFS p50=0、MINSPAN=64 也才 p50=1，而 dev tier2 是 7~11 格。
   真正的 Q2 探針必須用 tier2 的 (s,g)，而那種配對資料裡沒有真軌跡
   ⇒ 參考路用 BFS 回溯的格心序列（這同時直接測主人 8/24 問的「拿 BFS 生答案」可行性）。
用法（jasmine）：python -u experiments/exp_dump_dev_tasks.py  → results/devtasks_{env}.npz
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lacot import dev_eval as DE
import ogbench

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OGB_DATA = os.environ.get("OGBENCH_DATA_DIR", "/archive/cymaxwelllee/data/ogbench")
ENV_NAME = os.environ.get("LACOT_ENV", "pointmaze-medium-stitch-v0")

env, _, _ = ogbench.make_env_and_datasets(ENV_NAME, dataset_dir=OGB_DATA)
# ⚠️ 參數跟主線 dev 尺一致（n_per_tier=80, n_tiers=3, seed=0）⇒ 題目跟 G2 那批同源
tasks = DE.build_dev_tasks(env, n_per_tier=80, n_tiers=3, seed=0)


def bfs_path(src, dst):
    """從 dist 表回溯 src→dst 的格路徑（每步 dist 嚴格 −1 ⇒ 必達、無環）。"""
    dist = DE._bfs_from(env, src)
    assert tuple(dst) in dist, f"⛔ {dst} 從 {src} 不可達，而它是 dev 題 ⇒ maze_map 讀錯了"
    path, cur = [tuple(dst)], tuple(dst)
    while cur != tuple(src):
        d = dist[cur]
        nxt = [(cur[0] + di, cur[1] + dj) for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1))]
        cur = next(c for c in nxt if dist.get(c, 10 ** 9) == d - 1)
        path.append(cur)
    return path[::-1]


init_xy, goal_xy, bfs_d, tier, paths = [], [], [], [], []
for tk in tasks:
    p = bfs_path(tk["init_ij"], tk["goal_ij"])
    assert len(p) == tk["bfs_dist"] + 1
    paths.append(np.array([env.unwrapped.ij_to_xy(c) for c in p], np.float64))
    init_xy.append(tk["init_xy"]); goal_xy.append(tk["goal_xy"])
    bfs_d.append(tk["bfs_dist"]); tier.append(tk["tier"])

L = max(len(p) for p in paths)
pad = np.zeros((len(paths), L, 2)); plen = np.array([len(p) for p in paths])
for i, p in enumerate(paths):
    pad[i, :len(p)] = p
out = os.path.join(ROOT, "results", f"devtasks_{ENV_NAME}.npz")
np.savez(out, init_xy=np.array(init_xy), goal_xy=np.array(goal_xy),
         bfs_dist=np.array(bfs_d), tier=np.array(tier), path_xy=pad, path_len=plen)
n2 = int((np.array(tier) == 2).sum())
print(f"寫出 {out}  題數={len(paths)}（tier2 {n2} 題）"
      f"  BFS 距離範圍 {min(bfs_d)}~{max(bfs_d)}", flush=True)
