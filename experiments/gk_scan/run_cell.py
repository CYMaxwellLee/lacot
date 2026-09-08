#!/usr/bin/env python
"""G×K 因子化 VQ 掃描：訓一格（G, K）並量測（重建、每組使用率、因子有效性）。

用法：
  python run_cell.py --g 4 --k 32 --seed 21032 --threads 4

⛔ CPU-only（強制 device='cpu'）。固定 seed 公式：seed = 20000 + G*1000 + K
（24 格互不相同、可重跑），split_seed 全格共用 42（同一份 train/val 切分，
格與格之間才公平比較，沿用 M0 的理由）。

輸出：
  results/G{G}_K{K}.json   ：本格量測（含因子有效性）
  ckpt/G{G}_K{K}.pt         ：模型 state_dict + config（供推薦格 A/B render 用）
"""
import argparse
import os
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("HIP_VISIBLE_DEVICES", "")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch

import gk_common as gc

DEFAULT_DATA_DIR = os.environ.get("OGBENCH_DATA_DIR", "/home/cymaxwelllee/.ogbench/data")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--g", type=int, required=True, choices=[1, 2, 4, 8, 16, 32])
    ap.add_argument("--k", type=int, required=True, choices=[8, 16, 32, 64])
    ap.add_argument("--seg-len", type=int, default=4)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--beta", type=float, default=0.25)
    ap.add_argument("--group-dim", type=int, default=gc.GROUP_DIM)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--decay", type=float, default=0.99)
    ap.add_argument("--dead-steps", type=int, default=500)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--n-factor-samples", type=int, default=256)
    ap.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR)
    ap.add_argument("--out-dir", type=str, default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    if args.seed is None:
        args.seed = 20000 + args.g * 1000 + args.k

    torch.set_num_threads(max(1, args.threads))

    tag = f"G{args.g}_K{args.k}"
    print(f"=== gk_scan run {tag} (seed={args.seed}, split_seed={args.split_seed}) ===")

    data_path = os.path.join(args.data_dir, f"{gc.DATASET_NAME}.npz")
    if not os.path.exists(data_path):
        print(f"⛔ 找不到資料檔：{data_path}")
        sys.exit(1)

    t0 = time.time()
    data = gc.load_segments_with_obs(data_path, args.seg_len, split_seed=args.split_seed, val_frac=args.val_frac)
    print(
        f"segments: total={len(data['seg_starts'])} train={len(data['train_idx'])} "
        f"val={len(data['val_idx'])} seg_dim={data['segs'].shape[1]} obs_dim={data['obs_dim']} "
        f"fallback_segmentation_used={data['fallback_segmentation_used']} "
        f"(extract took {time.time()-t0:.1f}s)"
    )

    model, results, all_codes, extra = gc.train_and_eval(
        data,
        G=args.g,
        K=args.k,
        D=args.group_dim,
        steps=args.steps,
        batch=args.batch,
        lr=args.lr,
        beta=args.beta,
        hidden=args.hidden,
        decay=args.decay,
        dead_steps=args.dead_steps,
        seed=args.seed,
    )
    m = results["metrics"]
    print(
        f"train_mse={m['recon_mse_train']:.5f} val_mse={m['recon_mse_val']:.5f} "
        f"baseline_val={m['baseline_mse_val']:.5f} improve={m['val_improvement_over_baseline_pct']:.1f}% "
        f"any_collapsed={m['any_group_collapsed']} restarts={m['restarts_total']} "
        f"steps_used={m['total_steps_used']} extended={results['plateau']['extended']} "
        f"train_seconds={results['timing']['train_seconds']:.1f}"
    )

    fv = gc.factor_validity(
        model, data, extra["obs_mu"], extra["obs_sd"], args.g, args.k,
        n_samples=args.n_factor_samples, seed=args.seed + 777,
    )
    print(
        f"factor_validity: mean_delta_mse={fv['mean_delta_mse']:.5f} "
        f"min={fv['min_delta_mse']:.5f} max={fv['max_delta_mse']:.5f} "
        f"mean_abs_offdiag_corr={fv['mean_abs_offdiag_corr']} "
        f"mean_top_dim_energy_frac={fv['mean_top_dim_energy_frac']:.3f}"
    )

    results["factor_validity"] = fv
    results["config"] = dict(
        G=args.g, K=args.k, group_dim=args.group_dim, seg_len=args.seg_len,
        steps=args.steps, batch=args.batch, lr=args.lr, beta=args.beta,
        hidden=args.hidden, decay=args.decay, dead_steps=args.dead_steps,
        seed=args.seed, split_seed=args.split_seed, val_frac=args.val_frac,
        n_factor_samples=args.n_factor_samples, dataset=gc.DATASET_NAME,
    )
    results["data"] = dict(
        n_segments_total=int(len(data["seg_starts"])),
        n_train=int(len(data["train_idx"])),
        n_val=int(len(data["val_idx"])),
        act_dim=int(data["act_dim"]),
        obs_dim=int(data["obs_dim"]),
        seg_dim=int(data["segs"].shape[1]),
        n_episodes=int(data["n_episodes"]),
        fallback_segmentation_used=bool(data["fallback_segmentation_used"]),
    )
    results["capacity_bits"] = float(args.g * np.log2(args.k))

    results_dir = os.path.join(args.out_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    out_json = os.path.join(results_dir, f"{tag}.json")
    gc.save_json(results, out_json)
    print(f"saved: {out_json}")

    ckpt_dir = os.path.join(args.out_dir, "ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"{tag}.pt")
    torch.save(
        dict(
            state_dict=model.state_dict(),
            config=results["config"],
            obs_mu=extra["obs_mu"],
            obs_sd=extra["obs_sd"],
        ),
        ckpt_path,
    )
    print(f"saved: {ckpt_path}")
    print(f"=== done {tag} (wall {time.time()-t0:.1f}s) ===")


if __name__ == "__main__":
    main()
