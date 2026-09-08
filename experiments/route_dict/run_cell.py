#!/usr/bin/env python
"""M0R 路線字典：訓一格 K 並量測 + 視覺化（K 個路線字疊圖 + 3 字真實段疊圖）。

用法：
  python run_cell.py --k 16 --threads 4

⛔ CPU-only。固定 seed 公式：seed = 30000 + K；split_seed 三格共用 42。
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

import route_common as rc

DEFAULT_DATA_DIR = os.environ.get("OGBENCH_DATA_DIR", "/home/cymaxwelllee/.ogbench/data")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, required=True, choices=[8, 16, 32])
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--beta", type=float, default=0.25)
    ap.add_argument("--latent-dim", type=int, default=16)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--decay", type=float, default=0.99)
    ap.add_argument("--dead-steps", type=int, default=500)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--arc-length", type=float, default=rc.ARC_LENGTH)
    ap.add_argument("--n-resample", type=int, default=rc.N_RESAMPLE)
    ap.add_argument("--n-cluster-examples", type=int, default=5)
    ap.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR)
    ap.add_argument("--out-dir", type=str, default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    if args.seed is None:
        args.seed = 30000 + args.k

    torch.set_num_threads(max(1, args.threads))

    tag = f"K{args.k}"
    print(f"=== route_dict run {tag} (seed={args.seed}, split_seed={args.split_seed}) ===")

    data_path = os.path.join(args.data_dir, f"{rc.DATASET_NAME}.npz")
    if not os.path.exists(data_path):
        print(f"⛔ 找不到資料檔：{data_path}")
        sys.exit(1)

    t0 = time.time()
    data = rc.load_route_data(
        data_path, split_seed=args.split_seed, val_frac=args.val_frac,
        arc_length=args.arc_length, n_resample=args.n_resample,
    )
    ext = data["extraction"]
    print(
        f"route segments: total={len(data['segs'])} train={len(data['train_idx'])} "
        f"val={len(data['val_idx'])} seg_dim={data['seg_dim']} grid_ok={ext['grid_ok']} "
        f"n_degenerate_rotation={ext['n_degenerate_rotation']} "
        f"segs_per_ep(mean/min/max)={ext['n_segments_per_episode_mean']:.2f}/"
        f"{ext['n_segments_per_episode_min']}/{ext['n_segments_per_episode_max']} "
        f"(extract took {time.time()-t0:.1f}s)"
    )

    if len(data["segs"]) < 100:
        print(f"⛔ 段數過少（{len(data['segs'])}），無法訓練，停下回報")
        sys.exit(1)

    model, results, all_codes = rc.train_and_eval(
        data, K=args.k, latent_dim=args.latent_dim, steps=args.steps, batch=args.batch,
        lr=args.lr, beta=args.beta, hidden=args.hidden, decay=args.decay,
        dead_steps=args.dead_steps, seed=args.seed,
    )
    m = results["metrics"]
    print(
        f"train_mse={m['recon_mse_train']:.5f} val_mse={m['recon_mse_val']:.5f} "
        f"baseline_val={m['baseline_mse_val']:.5f} improve={m['val_improvement_over_baseline_pct']:.1f}% "
        f"perplexity={m['perplexity']:.2f}/{args.k} active={m['n_active_codes']} "
        f"top1={m['top1_usage_frac']*100:.1f}% collapsed={m['collapsed']} "
        f"restarts={m['restarts_total']} steps_used={m['total_steps_used']} "
        f"train_seconds={results['timing']['train_seconds']:.1f}"
    )

    results["config"] = dict(
        K=args.k, latent_dim=args.latent_dim, steps=args.steps, batch=args.batch, lr=args.lr,
        beta=args.beta, hidden=args.hidden, decay=args.decay, dead_steps=args.dead_steps,
        seed=args.seed, split_seed=args.split_seed, val_frac=args.val_frac,
        arc_length=args.arc_length, n_resample=args.n_resample, dataset=rc.DATASET_NAME,
    )
    results["data"] = dict(
        n_segments_total=int(len(data["segs"])),
        n_train=int(len(data["train_idx"])),
        n_val=int(len(data["val_idx"])),
        seg_dim=int(data["seg_dim"]),
        extraction={k: v for k, v in ext.items() if k != "segs"},
    )

    results_dir = os.path.join(args.out_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    # --- 視覺化 1：K 個路線字疊圖（decode 每個 codebook 向量本身） ---
    model.eval()
    with torch.no_grad():
        code_ids_all = torch.arange(args.k, dtype=torch.long)
        decoded = model.decode_code(code_ids_all).numpy()  # (K, seg_dim)
    word_polylines = [decoded[k].reshape(args.n_resample, 2) for k in range(args.k)]
    words_png = os.path.join(results_dir, f"route_words_{tag}.png")
    rc.draw_route_words(word_polylines, words_png, title=f"route words (decoded prototypes) - K={args.k}")
    print(f"saved: {words_png}")

    # --- 視覺化 2：抽 3 個最常用的字，各疊 n_cluster_examples 條真實段 ---
    counts = np.array(m["usage_counts"])
    top3_codes = np.argsort(-counts)[:3].tolist()
    combined_idx = np.concatenate([data["train_idx"], data["val_idx"]])
    combined_segs = data["segs"][combined_idx]  # 對齊 all_codes 的順序（同 train_and_eval 的串接順序）
    rng = np.random.default_rng(args.seed + 999)
    examples_per_code = {}
    for c in top3_codes:
        pos = np.flatnonzero(all_codes == c)
        n_pick = min(args.n_cluster_examples, len(pos))
        pick = rng.choice(pos, size=n_pick, replace=False) if n_pick > 0 else np.array([], dtype=int)
        examples_per_code[c] = [combined_segs[p].reshape(args.n_resample, 2) for p in pick]
    decoded_per_code = {c: word_polylines[c] for c in top3_codes}
    cluster_png = os.path.join(results_dir, f"route_clusters_{tag}.png")
    rc.draw_code_examples(top3_codes, examples_per_code, decoded_per_code, cluster_png,
                           title=f"top-3 codes x {args.n_cluster_examples} real segments - K={args.k}")
    print(f"saved: {cluster_png}")

    results["artifacts"] = dict(
        route_words_png=os.path.relpath(words_png, args.out_dir),
        route_clusters_png=os.path.relpath(cluster_png, args.out_dir),
        top3_codes=top3_codes,
        top3_codes_usage_frac=[float(m["usage_frac"][c]) for c in top3_codes],
        top3_codes_n_examples=[len(examples_per_code[c]) for c in top3_codes],
    )

    out_json = os.path.join(results_dir, f"{tag}.json")
    rc.save_json(results, out_json)
    print(f"saved: {out_json}")
    print(f"=== done {tag} (wall {time.time()-t0:.1f}s) ===")


if __name__ == "__main__":
    main()
