#!/usr/bin/env python
"""P0：真人形量尺包（humanoidmaze-medium-stitch-v0，任務量尺包步驟）。

搬自 walk_verify/p0_build_ruler.py 的方法論（ant 版），換到 humanoid：
  (a) distance -> steps 查表：first-passage 步數的分位數
  (b) 步速分佈：每步 xy 位移
  (c) 軀幹（root）z 高度分佈：翻倒線 = p1（雙足更容易翻，這把尺對 humanoid 更重要）
  (d) 相鄰步動作差 |a_t - a_(t-1)| 分佈（L2 over 21 維）
  另外直接從資料印出 act_dim / obs_dim（任務要求「應為 21 維、obs 實測為準」）。

⛔ CPU-only。輸出 results/ruler_pack.json ＋ results/ruler_pack.npz（疊圖用原始樣本）
＋ 三張分佈 PNG。
"""
import argparse
import os
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("HIP_VISIBLE_DEVICES", "")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402

import hd_common as hc  # noqa: E402

DEFAULT_DATA_DIR = os.environ.get("OGBENCH_DATA_DIR", "/home/cymaxwelllee/.ogbench/data")
HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR)
    ap.add_argument("--out-dir", type=str, default=os.path.join(HERE, "results"))
    ap.add_argument("--ep-subsample", type=int, default=4,
                    help="distance->steps 表用 1/N 的 episode（其他三把尺一律全量）")
    ap.add_argument("--max-horizon", type=int, default=400,
                    help="humanoid ep_len=401，跟 ant(200) 版本比例一致地放大到 400")
    ap.add_argument("--sample-cap", type=int, default=400000)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    t0 = time.time()
    print("=== P0 ruler pack (真 humanoid 量尺包, humanoidmaze-medium-stitch-v0) ===")
    raw = hc.load_raw(args.data_dir, hc.DATASET_NAME, "train")
    print(f"data: {raw['_path']} steps={len(raw['actions'])}")

    act_dim = raw["actions"].shape[1]
    obs_dim = raw["observations"].shape[1]
    qpos_dim = raw["qpos"].shape[1]
    qvel_dim = raw["qvel"].shape[1]
    print(f"[實測] act_dim={act_dim} (預期 21) | obs_dim={obs_dim} | qpos_dim={qpos_dim} | qvel_dim={qvel_dim}")
    assert act_dim == 21, f"⛔ act_dim 預期 21，實測 {act_dim}"

    ruler = hc.build_ruler(raw, max_horizon=args.max_horizon, sub=args.ep_subsample, seed=0)
    ruler["built_wall_seconds"] = time.time() - t0
    ruler["obs_dim_measured"] = int(obs_dim)
    ruler["qpos_dim_measured"] = int(qpos_dim)
    ruler["qvel_dim_measured"] = int(qvel_dim)

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
    print(f"(c) 軀幹(root) z: mean={tz['mean']:.4f} p1(翻倒線)={tz['fall_line_p1']:.4f} "
          f"p5={tz['quantiles']['5']:.4f} p50={tz['quantiles']['50']:.4f} p95={tz['quantiles']['95']:.4f}")
    print(f"(d) |a_t - a_(t-1)| (L2/{act_dim}dim): mean={ad['mean']:.4f} p50={ad['quantiles']['50']:.4f} "
          f"p95={ad['quantiles']['95']:.4f} p99={ad['quantiles']['99']:.4f}")
    print(f"goal_tol (ogbench locomaze/maze.py:86) = {ruler['goal_tol']}")

    starts, ends = hc.episode_bounds(raw["terminals"])
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
    hc.save_json(ruler, js)

    hc.draw_hist_overlay([dict(label="real humanoid step speed", values=cap(sd))],
                         os.path.join(args.out_dir, "ruler_step_speed.png"),
                         "ruler (b): real humanoid per-step xy displacement", "|dxy| per step")
    hc.draw_hist_overlay([dict(label="real humanoid torso(root) z", values=cap(z))],
                         os.path.join(args.out_dir, "ruler_torso_z.png"),
                         "ruler (c): real humanoid torso(root) height", "qpos[2] (m)",
                         vlines=[dict(x=tz["fall_line_p1"], label=f"fall line p1={tz['fall_line_p1']:.3f}",
                                      color_idx=7)])
    hc.draw_hist_overlay([dict(label="real humanoid |a_t - a_(t-1)|", values=cap(adf))],
                         os.path.join(args.out_dir, "ruler_action_diff.png"),
                         "ruler (d): real humanoid action smoothness", "L2 norm of consecutive action diff")
    print(f"\nsaved: {js}\nsaved: {npz}\nsaved: 3 ruler PNGs in {args.out_dir}")
    print(f"=== done wall={time.time()-t0:.1f}s ===")


if __name__ == "__main__":
    main()
