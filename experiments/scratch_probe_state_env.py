"""Probe the OGBench STATE pointmaze env API: obs shape, goal representation,
success mechanics — so the lightweight state rollout eval can be built right.
Read-only: makes env, resets a task, steps a few random actions, prints keys."""
import numpy as np, ogbench
env, train_ds, val_ds = ogbench.make_env_and_datasets("pointmaze-medium-navigate-v0")
print("env:", type(env).__name__)
print("action_space:", env.action_space, " obs_space:", env.observation_space)
print("num tasks (task_infos):", len(getattr(env.unwrapped, "task_infos", [])))
obs, info = env.reset(options={"task_id": 1, "render_goal": False})
print("reset obs shape:", np.asarray(obs).shape, "dtype", np.asarray(obs).dtype)
print("reset info keys:", sorted(info.keys()))
if "goal" in info:
    g = np.asarray(info["goal"]); print("  info['goal'] shape:", g.shape, "sample:", g.ravel()[:4])
print("cur_goal_xy:", getattr(env.unwrapped, "cur_goal_xy", None))
print("first obs sample (position?):", np.asarray(obs).ravel()[:6])
# step a few, watch success/reward/term
suc = False
for t in range(50):
    a = env.action_space.sample()
    obs, rew, term, trunc, info = env.step(a)
    if info.get("success"):
        suc = True
    if t < 2:
        print(f"  step{t}: rew={rew} term={term} trunc={trunc} success={info.get('success')} obs={np.asarray(obs).ravel()[:4]}")
print("info keys after step:", sorted(info.keys()))
print("saw success in 50 random steps:", suc)
# train dataset keys (what we already load)
print("train_ds keys:", sorted(train_ds.keys()) if hasattr(train_ds, "keys") else type(train_ds))
