#!/usr/bin/env python
"""P1 診斷：真動作開環重放能重現多久？（判定尺自檢的定位工具）

smoke 發現 `true` arm 在 40 步視窗只有 ~54% 限時到達、末端誤差 1.1 —— 先定位
是「判定尺壞」還是「開環重放本身就重現不了」，再決定要不要改尺。

三個變體：
  base        直接 set_state(dataset qpos/qvel float32) 後開環執行真動作
  warmstart0  同上，但每次 set_state 後把 data.qacc_warmstart 清零
  f64roundtrip 先從 dataset 狀態跑 5 步得到 float64 狀態，存下來，再 set_state 回去重放
               （若這個能完美重現 ⇒ 病因是 dataset 只存 float32、不是 solver 狀態）
量：每步的 xy 誤差、qpos 全體最大誤差，對步數的曲線（分位數）。
⛔ CPU-only。
"""
import argparse
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("HIP_VISIBLE_DEVICES", "")
os.environ.setdefault("MUJOCO_GL", "osmesa")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402

import wv_common as wv  # noqa: E402

DD = os.environ.get("OGBENCH_DATA_DIR", "/home/cymaxwelllee/.ogbench/data")
DATASET = "antmaze-medium-stitch-v0"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--horizon", type=int, default=80)
    ap.add_argument("--seed", type=int, default=20260908)
    args = ap.parse_args()

    raw = wv.load_npz(DD, DATASET, "val")
    qpos, qvel, act = raw["qpos"], raw["qvel"], raw["actions"]
    starts, ends = wv.episode_bounds(raw["terminals"])
    rng = np.random.default_rng(args.seed)
    cands = []
    for s0, e0 in zip(starts, ends):
        for off in range(0, int(e0 - s0 + 1) - args.horizon - 1, 4):
            cands.append(int(s0 + off))
    wins = rng.choice(np.array(cands), size=args.n, replace=False)

    env = wv.make_env(DD, DATASET)
    u = env.unwrapped

    xy_err = {v: np.zeros((args.n, args.horizon)) for v in ("base", "warmstart0", "f64roundtrip")}
    qp_err = {v: np.zeros((args.n, args.horizon)) for v in ("base", "warmstart0")}

    for wi, i0 in enumerate(wins):
        i0 = int(i0)
        for variant in ("base", "warmstart0"):
            env.reset()
            u.set_state(qpos[i0].copy(), qvel[i0].copy())
            if variant == "warmstart0":
                u.data.qacc_warmstart[:] = 0.0
            for t in range(args.horizon):
                env.step(act[i0 + t])
                cur = np.asarray(u.data.qpos)[:15]
                xy_err[variant][wi, t] = np.linalg.norm(cur[:2] - qpos[i0 + t + 1, :2])
                qp_err[variant][wi, t] = np.abs(cur - qpos[i0 + t + 1]).max()

        # f64 round-trip：自己產生 float64 參考軌跡，再從同一個 float64 狀態重放
        env.reset()
        u.set_state(qpos[i0].copy(), qvel[i0].copy())
        ref_qpos = []
        qp0 = np.asarray(u.data.qpos).copy()
        qv0 = np.asarray(u.data.qvel).copy()
        for t in range(args.horizon):
            env.step(act[i0 + t])
            ref_qpos.append(np.asarray(u.data.qpos).copy())
        env.reset()
        u.set_state(qp0, qv0)
        for t in range(args.horizon):
            env.step(act[i0 + t])
            xy_err["f64roundtrip"][wi, t] = np.linalg.norm(
                np.asarray(u.data.qpos)[:2] - ref_qpos[t][:2])

    print(f"=== replay fidelity diagnostic  n={args.n} H={args.horizon} ===")
    print(f"{'step':>5} " + " ".join(f"{v:>26}" for v in ("base xy p50/p90", "warmstart0 xy p50/p90",
                                                          "f64roundtrip xy p50/p90")))
    for t in [0, 1, 2, 3, 4, 7, 11, 15, 19, 23, 31, 39, 47, 59, 79]:
        if t >= args.horizon:
            continue
        cells = []
        for v in ("base", "warmstart0", "f64roundtrip"):
            a = xy_err[v][:, t]
            cells.append(f"{np.percentile(a,50):11.2e}/{np.percentile(a,90):.2e}")
        print(f"{t+1:5d} " + " ".join(f"{c:>26}" for c in cells))
    print("\nxy 誤差超過 goal_tol=0.5 的比例（base）：")
    for t in [3, 7, 11, 15, 19, 23, 31, 39, 79]:
        if t >= args.horizon:
            continue
        print(f"  step {t+1:3d}: {(xy_err['base'][:,t] > wv.GOAL_TOL).mean()*100:5.1f}%   "
              f"p50={np.percentile(xy_err['base'][:,t],50):.3f} "
              f"p90={np.percentile(xy_err['base'][:,t],90):.3f}")


if __name__ == "__main__":
    main()
