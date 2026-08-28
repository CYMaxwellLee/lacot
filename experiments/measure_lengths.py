"""量三種長度，決定 T_CAP 要放多大、以及 goal 抽法差多少。主人 2026-08-24：「軌跡長度去量量」。

量什麼：
  ① 資料集每條 episode 的長度分布
  ② goal 距離分布：我們的 geometric(0.02) vs 官方的均勻抽（同一批起點，兩種抽法直接對照）
  ③ eval 時官方 task 的 oracle BFS 路徑長度分布 + 有多少比例撞到 horizon 上限（＝答案是半條）

⭐ 為什麼要一起量：T_CAP（Perceiver 一次讀幾個點）要多大，取決於實際路徑多長；
   而訓練時看到的路徑長度由 goal 抽法決定 ⇒ 兩件事是同一個問題的兩半。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ogbench

OGB = os.environ.get("OGBENCH_DATA_DIR", "/archive/cymaxwelllee/data/ogbench")
GEOM_P = 0.02
GAIN = 5.0
# 🚨 F4（subagent 稽核）：300 那次量到的 large-stitch「平均 255 步」是 E[min(L,300)]，
#    是【被截斷過的】平均（20% 撞頂），而 T_CAP=256 就是照那個數字定的 ⇒ 可能定小了。
HORIZON = int(os.environ.get("LACOT_ORACLE_HORIZON", 800))
ENVS = os.environ.get("LACOT_ENVS", "pointmaze-medium-navigate-v0,pointmaze-large-stitch-v0").split(",")


def pct(a, ps=(5, 25, 50, 75, 95, 100)):
    return "  ".join(f"p{p}={np.percentile(a, p):.0f}" for p in ps)


for ENV_NAME in ENVS:
    print(f"\n================ {ENV_NAME} ================", flush=True)
    d = np.load(f"{OGB}/{ENV_NAME}.npz")
    OBS = np.asarray(d["observations"], np.float32)
    TERM = np.asarray(d["terminals"], bool)
    N = OBS.shape[0]
    ends = np.flatnonzero(TERM)
    starts = np.concatenate([[0], ends[:-1] + 1])
    traj_end = np.empty(N, np.int64)
    for s0, e0 in zip(starts, ends):
        traj_end[s0:e0 + 1] = e0
    lens = ends - starts + 1

    print(f"① 資料集：{N} 個 transition、{len(lens)} 條軌跡")
    print(f"   每條長度  mean={lens.mean():.0f}  {pct(lens)}")

    # ② 兩種 goal 抽法，同一批起點
    r = np.random.default_rng(0)
    idx = r.integers(0, N, 200000)
    te = traj_end[idx]
    geo = np.minimum(idx + r.geometric(GEOM_P, size=idx.size), te) - idx          # 我們現在的
    dist = r.random(idx.size)                                                      # 官方均勻
    uni = np.round(np.minimum(idx + 1, te) * dist + te * (1 - dist)).astype(np.int64) - idx
    print(f"\n② goal 距離（步數）")
    print(f"   我們 geometric(0.02)  mean={geo.mean():6.1f}  {pct(geo)}")
    print(f"   官方 均勻             mean={uni.mean():6.1f}  {pct(uni)}")
    print(f"   ⇒ 官方的目標平均遠 {uni.mean()/max(geo.mean(),1e-9):.1f} 倍")

    # ③ eval 時官方 task 的 oracle 路徑
    os.environ.setdefault("OGBENCH_DATA_DIR", OGB)
    env, _, _ = ogbench.make_env_and_datasets(ENV_NAME, dataset_dir=OGB)
    n_tasks = len(env.unwrapped.task_infos)
    plens, truncated, sg_dist = [], 0, []
    for task in range(1, n_tasks + 1):
        for sd_ in range(5):
            obs, info = env.reset(seed=1000 * task + sd_, options={"task_id": task, "render_goal": False})
            goal = np.asarray(info["goal"], np.float64)
            xy = np.asarray(obs, np.float64)
            sg_dist.append(float(np.linalg.norm(xy - goal)))
            hit = True
            for step in range(HORIZON):
                subgoal, bfs = env.unwrapped.get_oracle_subgoal(xy, goal)
                here = env.unwrapped.xy_to_ij(xy)
                target = goal if bfs[here[0], here[1]] == 0 else np.asarray(subgoal)
                a = np.clip(GAIN * (target - xy), -1, 1)
                xy = xy + 0.2 * a
                if np.linalg.norm(xy - goal) < 0.5:
                    hit = False
                    break
            plens.append(step + 1)
            truncated += int(hit)
    plens = np.array(plens); untr = plens[:len(plens)]
    print(f"\n③ oracle BFS 路徑（{n_tasks} tasks x 5 seeds = {len(plens)} 條，horizon={HORIZON}）")
    print(f"   路徑長度  mean={plens.mean():.0f}  {pct(plens)}")
    print(f"   撞到 horizon 上限（答案是半條）：{truncated}/{len(plens)} = {truncated/len(plens):.1%}")
    if truncated:
        print(f"   ⛔ 上面那個 mean 是【被截斷過的】平均，真值更高 —— 放大 horizon 再量")
    print(f"   起點到目標的直線距離  mean={np.mean(sg_dist):.1f}")
    print(f"\n⇒ 現在 T_CAP=16：訓練看到的路平均 {geo.mean():.0f} 步 ⇒ 每 {max(geo.mean()/16,0):.1f} 步一個點")
    print(f"                  eval 的 oracle 路平均 {plens.mean():.0f} 步 ⇒ 每 {plens.mean()/16:.1f} 步一個點", flush=True)
