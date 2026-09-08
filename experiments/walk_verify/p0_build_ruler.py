#!/usr/bin/env python
"""P0 附件：真螞蟻量尺包（P1/P2 共用的判定尺）。

四把尺，全部從 antmaze-medium-stitch-v0 訓練資料量出（不是猜的、不是預設值）：
  (a) distance -> steps 查表：first-passage 步數的分位數（供限時 N）
  (b) 步速分佈：每步 xy 位移
  (c) 軀幹 z 高度分佈：翻倒線 = p1
  (d) 相鄰步動作差 |a_t - a_{t-1}| 分佈（平滑度基準）

⛔ CPU-only。輸出 results/ruler_pack.json ＋ results/ruler_pack.npz（原始樣本供疊圖）。
"""
import argparse
import os
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("HIP_VISIBLE_DEVICES", "")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402

import wv_common as wv  # noqa: E402

DEFAULT_DATA_DIR = os.environ.get("OGBENCH_DATA_DIR", "/home/cymaxwelllee/.ogbench/data")
DATASET = "antmaze-medium-stitch-v0"
HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR)
    ap.add_argument("--out-dir", type=str, default=os.path.join(HERE, "results"))
    ap.add_argument("--ep-subsample", type=int, default=4,
                    help="distance->steps 表用 1/N 的 episode（其他三把尺一律全量）")
    ap.add_argument("--max-horizon", type=int, default=200)
    ap.add_argument("--sample-cap", type=int, default=400000,
                    help="存進 npz 的原始樣本上限（疊圖用，固定 seed 抽樣）")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    t0 = time.time()
    print("=== P0 ruler pack (真螞蟻量尺包) ===")
    raw = wv.load_npz(args.data_dir, DATASET, "train")
    print(f"data: {raw['_path']} steps={len(raw['actions'])}")

    ruler = wv.build_ruler(raw, max_horizon=args.max_horizon, sub=args.ep_subsample, seed=0)
    ruler["built_wall_seconds"] = time.time() - t0

    print("\n(a) distance -> steps（first-passage 分位數；censored=在 episode 內從未達到）")
    print(f"{'d':>6} {'p10':>7} {'p25':>7} {'p50(N)':>8} {'p75':>7} {'p90':>7} {'censored':>9} {'n':>9}")
    for g in ruler["distance_to_steps"]["grid"]:
        e = ruler["distance_to_steps"]["table"][f"{g:g}"]
        q = e["quantiles"]
        print(f"{g:6.1f} {q['10']:7.1f} {q['25']:7.1f} {q['50']:8.1f} {q['75']:7.1f} "
              f"{q['90']:7.1f} {e['censored_frac']*100:8.1f}% {e['n_reached']:9d}")

    ss, tz, ad = ruler["step_speed"], ruler["torso_z"], ruler["action_smoothness"]
    print(f"\n(b) 步速 |dxy|/step: mean={ss['mean']:.4f} p1={ss['quantiles']['1']:.4f} "
          f"p25={ss['quantiles']['25']:.4f} p50={ss['quantiles']['50']:.4f} "
          f"p75={ss['quantiles']['75']:.4f} p99={ss['quantiles']['99']:.4f}")
    print(f"(c) 軀幹 z: mean={tz['mean']:.4f} p1(翻倒線)={tz['fall_line_p1']:.4f} "
          f"p5={tz['quantiles']['5']:.4f} p50={tz['quantiles']['50']:.4f} p95={tz['quantiles']['95']:.4f}")
    print(f"(d) |a_t - a_(t-1)| (L2/8dim): mean={ad['mean']:.4f} p50={ad['quantiles']['50']:.4f} "
          f"p95={ad['quantiles']['95']:.4f} p99={ad['quantiles']['99']:.4f}")
    print(f"goal_tol (ogbench locomaze/maze.py:86) = {ruler['goal_tol']}")

    # 原始樣本（疊圖用）
    starts, ends = wv.episode_bounds(raw["terminals"])
    xy = raw["qpos"][:, :2]
    sd = np.concatenate([np.linalg.norm(np.diff(xy[s:e + 1], axis=0), axis=1)
                         for s, e in zip(starts, ends)])
    adf = np.concatenate([np.linalg.norm(np.diff(raw["actions"][s:e + 1], axis=0), axis=1)
                          for s, e in zip(starts, ends)])
    z = raw["qpos"][:, 2]
    rng = np.random.default_rng(0)

    def cap(a):
        return a if len(a) <= args.sample_cap else a[rng.choice(len(a), args.sample_cap, replace=False)]

    npz = os.path.join(args.out_dir, "ruler_pack.npz")
    np.savez_compressed(npz, step_speed=cap(sd).astype(np.float32),
                        torso_z=cap(z).astype(np.float32),
                        action_diff=cap(adf).astype(np.float32))
    js = os.path.join(args.out_dir, "ruler_pack.json")
    wv.save_json(ruler, js)

    # 三張基準分佈圖
    wv.draw_hist_overlay([dict(label="real ant step speed", values=cap(sd))],
                         os.path.join(args.out_dir, "ruler_step_speed.png"),
                         "ruler (b): real ant per-step xy displacement", "|dxy| per step")
    wv.draw_hist_overlay([dict(label="real ant torso z", values=cap(z))],
                         os.path.join(args.out_dir, "ruler_torso_z.png"),
                         "ruler (c): real ant torso height", "qpos[2] (m)",
                         vlines=[dict(x=tz["fall_line_p1"], label=f"fall line p1={tz['fall_line_p1']:.3f}",
                                      color_idx=7)])
    wv.draw_hist_overlay([dict(label="real ant |a_t - a_(t-1)|", values=cap(adf))],
                         os.path.join(args.out_dir, "ruler_action_diff.png"),
                         "ruler (d): real ant action smoothness", "L2 norm of consecutive action diff")
    print(f"\nsaved: {js}\nsaved: {npz}\nsaved: 3 ruler PNGs in {args.out_dir}")
    print(f"=== done wall={time.time()-t0:.1f}s ===")


if __name__ == "__main__":
    main()
