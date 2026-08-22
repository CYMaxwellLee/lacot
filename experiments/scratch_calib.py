"""Calibrate how EASY our eval is (per 主人 ②): roll out trivial controllers on the
same 5 tasks x 20 seeds and see how many they solve. If the straight-line controller
(ignores walls) already scores high, our high policy numbers partly reflect an easy eval.
  * straight-line P-controller: a = clip(5*(goal - pos))  -- ignores maze walls
  * BFS expert:                 env's get_oracle_subgoal   -- respects walls (solvable ceiling)
No training. Same rollout harness (receding-horizon CHUNK=4, MAXH=env 官方 horizon, info['success'])."""
import os, numpy as np, ogbench

# 資料位置：預設走官方 OGBENCH_DATA_DIR，沒設才用本機 archive
OGB_DATA = os.environ.get("OGBENCH_DATA_DIR", "/archive/cymaxwelllee/data/ogbench")
os.environ.setdefault("OGBENCH_DATA_DIR", OGB_DATA)
ENV_NAME = "pointmaze-medium-navigate-v0"
env, _, _ = ogbench.make_env_and_datasets(ENV_NAME)
MAXH = int(os.environ.get("LACOT_EVAL_MAXH", env.spec.max_episode_steps or 1000))  # 官方標準，不自訂難度
N_TASKS = len(env.unwrapped.task_infos); SEEDS = 20; CHUNK = 4; GAIN = 5.0

def straight_action(pos, goal):
    return np.clip(GAIN * (np.asarray(goal) - np.asarray(pos)), -1, 1).astype(np.float32)

def bfs_action(pos, goal):
    xy = np.asarray(pos, np.float64); gg = np.asarray(goal, np.float64)
    subgoal, bfs = env.unwrapped.get_oracle_subgoal(xy, gg)
    here = env.unwrapped.xy_to_ij(xy)
    target = gg if bfs[here[0], here[1]] == 0 else np.asarray(subgoal)
    return np.clip(GAIN * (target - xy), -1, 1).astype(np.float32)

def rollout(policy, tag):
    succ = ep = 0; steps_to_succ = []
    for task in range(1, N_TASKS + 1):
        for sd in range(SEEDS):
            obs, info = env.reset(seed=1000 * task + sd, options={"task_id": task, "render_goal": False})
            goal = info["goal"]; success = False; steps = 0
            while steps < MAXH and not success:
                for _ in range(CHUNK):  # same receding-horizon chunk cadence as our M4 eval
                    a = policy(obs, goal)
                    obs, rew, term, trunc, info = env.step(a)
                    steps += 1
                    if info.get("success"):
                        success = True
                    if success or term or trunc or steps >= MAXH:
                        break
            succ += int(success); ep += 1
            if success:
                steps_to_succ.append(steps)
    m = int(np.median(steps_to_succ)) if steps_to_succ else -1
    print(f"  {tag}: {succ}/{ep} = {succ/ep:.3f}   (median steps-to-goal {m})", flush=True)

print(f"==== EVAL DIFFICULTY CALIBRATION  {N_TASKS} tasks x {SEEDS} seeds, MAXH {MAXH} ====", flush=True)
rollout(straight_action, "straight-line (ignores walls) -- how easy is the eval")
rollout(bfs_action, "BFS expert (respects walls) -- solvable ceiling")
print("=> straight-line high => eval is easy (few walls between start/goal); low => walls matter.", flush=True)
