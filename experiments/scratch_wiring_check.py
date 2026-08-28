"""Wiring check for the LaCoT success-rate eval — verify the pieces line up BEFORE
trusting any success number. Checks the frame/normalization consistency between
the .npz observations (used for mu/sd) and the LIVE env obs + goal, plus that
goal==success-target and actions actually move the agent toward the goal.
No training needed — pure plumbing verification."""
import os, numpy as np, ogbench

# 資料位置：預設走官方 OGBENCH_DATA_DIR，沒設才用本機 archive
OGB_DATA = os.environ.get("OGBENCH_DATA_DIR", "/archive/cymaxwelllee/data/ogbench")
os.environ.setdefault("OGBENCH_DATA_DIR", OGB_DATA)
ENV_NAME = "pointmaze-medium-navigate-v0"
d = np.load(f"{OGB_DATA}/{ENV_NAME}.npz")
OBS = np.asarray(d["observations"], np.float32); ACT = np.asarray(d["actions"], np.float32)
mu, sd = OBS.mean(0), OBS.std(0) + 1e-6
print("=== .npz observations (mu/sd source) ===")
print("  shape", OBS.shape, "min", OBS.min(0), "max", OBS.max(0))
print("  mu", mu, "sd", sd)
print("  actions min/max:", ACT.min(0), ACT.max(0))

env, _, _ = ogbench.make_env_and_datasets(ENV_NAME)
print("\n=== live env obs + goal per task (same frame as .npz?) ===")
for task in (1, 2, 3):
    obs, info = env.reset(seed=task, options={"task_id": task, "render_goal": False})
    goal = np.asarray(info["goal"]); obs = np.asarray(obs)
    cur = np.asarray(env.unwrapped.cur_goal_xy)
    in_range = (OBS.min(0) - 1 <= obs).all() and (obs <= OBS.max(0) + 1).all()
    goal_in = (OBS.min(0) - 1 <= goal).all() and (goal <= OBS.max(0) + 1).all()
    print(f"  task{task}: obs {obs.round(2)} goal {goal.round(2)} | obs_in_npz_range {in_range} goal_in_range {goal_in} | goal==cur_goal_xy {np.allclose(goal, cur)}")
    print(f"    normalized: s={(obs-mu)/sd} g={(goal-mu)/sd}")

print("\n=== do actions move the agent toward the goal? (oracle sanity) ===")
obs, info = env.reset(seed=1, options={"task_id": 1, "render_goal": False})
goal = np.asarray(info["goal"]); obs = np.asarray(obs)
d0 = np.linalg.norm(obs - goal)
suc = False
for t in range(500):
    a = np.clip(5.0 * (goal - np.asarray(obs)), -1, 1).astype(np.float32)  # crude P-controller toward goal
    obs, rew, term, trunc, info = env.step(a)
    if info.get("success"):
        suc = True; break
    if term or trunc:
        break
d1 = np.linalg.norm(np.asarray(obs) - goal)
print(f"  start dist {d0:.2f} -> end dist {d1:.2f} in {t+1} steps, success={suc}  (a crude toward-goal controller should succeed if wiring is sane)")
print("  => if this SUCCEEDS, the env/goal/action/success loop is wired right and the frame is consistent.")
