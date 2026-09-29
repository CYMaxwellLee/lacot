"""S2: delivered policy_chunk/rollout AST + delivered Collector, REAL ogbench env (CPU).

Only the flow/head are toys. Shows whether per-(task,episode) resets are
actually paired across draws/arms, as README claims.
"""
import ast
import os
import sys
import numpy as np
import torch
import ogbench

WORK = "/home/cymaxwelllee/Projects/lacot/experiments/_workorders/ucontrast1"
MAIN = "/home/cymaxwelllee/Projects/lacot/experiments/scratch_lacot_rollout.py"
sys.path.insert(0, WORK)
from collector import Collector, summarize  # noqa: E402

torch.set_num_threads(1)
np.random.seed(12345)  # a fixed process-level state, like any real run would have
env = ogbench.make_env_and_datasets("pointmaze-large-stitch-v0", env_only=True)
DELIVERED = open(MAIN).read()
RESET = ('            obs, info = env.reset(seed=1000 * task + sd_, options={"task_id": task, "render_goal": False})\n')
assert DELIVERED.count(RESET) == 1
# Scratch-only illustration of the dev_eval.py:125-133 style pin, oracle path only.
FIXED = DELIVERED.replace(RESET, (
    "            if oracle_draw is not None:\n"
    "                np.random.seed(1000 * task + sd_); env.action_space.seed(1000 * task + sd_)\n") + RESET)


def head(cond, u):
    act = torch.tanh(.5 * (cond[:, 2:4] - cond[:, 0:2]) + .3 * u[:, 0])
    return act[:, None].expand(-1, 2, -1)


def harness(collector, source):
    tree = ast.parse(source)
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ("policy_chunk", "rollout")]
    ns = dict(torch=torch, np=np, env=env, N_TASKS=2, SEEDS=2, MAXH=12,
              FINISH_ON=False, FINISH_R=0., U_SOURCE="flow", GRAD_REFINE=False,
              ORACLE_COLLECTOR=collector, DIAG_DUMP=False, DIAG_ROWS=[],
              _intent_anchor_eval=lambda o, g: None,
              normstate=lambda x: torch.as_tensor(np.asarray(x, np.float32))[None],
              _intent_cond_arm=lambda a: None,
              condvec=lambda s, g, i: torch.cat((s[:, :2], g[:, :2]), -1),
              _bon_plan=lambda n, *a: torch.randn(n, 2, 2), _apply_refine=lambda c, u, r: u,
              _q=lambda u: u, ahead=head, _reset_grad_cache=lambda: None, _reseed_shuf=lambda s: None)
    exec(compile(ast.Module(body=funcs, type_ignores=[]), MAIN, "exec"), ns)
    return ns


def run(arm, sigma, source, draws=3):
    c = Collector(arm, sigma, draws)
    h = harness(c, source)
    old = sys.stdout
    sys.stdout = open(os.devnull, "w")
    try:
        for d in range(draws):
            h["rollout"](1, True, f"{arm}{sigma} d{d}", oracle_draw=d)
    finally:
        sys.stdout = old
    return c


for label, src in (("REWORK", DELIVERED),):
    print(f"==== {label}")
    cols = {}
    for arm, sigma in (("B", 0.0), ("A", 0.0), ("B", 0.1)):
        c = run(arm, sigma, src)
        cols[(arm, sigma)] = c
        g = [r for r in c.rows if (r["task"], r["episode"]) == (1, 0)]
        print(f"  {arm} sigma={sigma}: (task1,ep0) initial_sha per draw =",
              [r["initial_sha256"][:10] for r in g], " env_seed field =", [r["env_seed"] for r in g])
        try:
            s = summarize(c.rows, 3, arm, sigma)
            print(f"    summarize OK: oracle_at_k={s['oracle_at_k']} n_draws={s['n_draws']}")
        except ValueError as e:
            print(f"    summarize RAISES: {e}")
    a = [r["initial_sha256"] for r in cols[("A", 0.0)].rows]
    b = [r["initial_sha256"] for r in cols[("B", 0.1)].rows]
    print("  A vs B(0.1) same initial state per (task,ep,draw):", a == b)
    bt = [r["trajectory_sha256"] for r in cols[("B", 0.1)].rows if (r["task"], r["episode"]) == (1, 0)]
    print("  B(0.1) distinct trajectories across draws for (1,0):", len(set(bt)))

c8 = run("B", 0.0, DELIVERED, draws=8)
s8 = summarize(c8.rows, 8, "B", 0.0)
print("B sigma0 eight draws bitwise trajectory identical:", all(len({r["trajectory_sha256"] for r in c8.rows if (r["task"], r["episode"]) == (t["task"], t["episode"])}) == 1 for t in s8["tasks"]))
