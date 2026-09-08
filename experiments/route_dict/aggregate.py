#!/usr/bin/env python
"""彙整路線字典 3 格結果成一份 summary.json（純數字彙整，不重跑模型）。"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import route_common as rc

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
KS = [8, 16, 32]


def main():
    cells = []
    missing = []
    for K in KS:
        path = os.path.join(RESULTS_DIR, f"K{K}.json")
        if not os.path.exists(path):
            missing.append(f"K{K}")
            continue
        with open(path) as f:
            r = json.load(f)
        m = r["metrics"]
        cells.append(dict(
            K=K, val_mse=m["recon_mse_val"], train_mse=m["recon_mse_train"],
            baseline_mse_val=m["baseline_mse_val"], improve_pct=m["val_improvement_over_baseline_pct"],
            perplexity=m["perplexity"], n_active_codes=m["n_active_codes"],
            top1_usage_frac=m["top1_usage_frac"], top3_usage_frac=m["top3_usage_frac"],
            collapsed=m["collapsed"], restarts_total=m["restarts_total"],
            steps_used=m["total_steps_used"], extended=r["plateau"]["extended"],
            n_segments_total=r["data"]["n_segments_total"],
            n_degenerate_rotation=r["data"]["extraction"]["n_degenerate_rotation"],
            route_words_png=r["artifacts"]["route_words_png"],
            route_clusters_png=r["artifacts"]["route_clusters_png"],
        ))

    if missing:
        print(f"⛔ 缺 {len(missing)} 格未完成：{missing}")

    summary = dict(cells=cells, missing_cells=missing)
    out_path = os.path.join(RESULTS_DIR, "summary.json")
    rc.save_json(summary, out_path)
    print(f"saved: {out_path}")

    print("\n3-cell table (K, val_mse, improve%, perplexity/K, top1%, collapsed):")
    for c in cells:
        print(f"  K={c['K']:2d} val_mse={c['val_mse']:.5f} improve={c['improve_pct']:.1f}% "
              f"perplexity={c['perplexity']:.2f}/{c['K']} top1={c['top1_usage_frac']*100:.1f}% "
              f"collapsed={c['collapsed']} n_segments={c['n_segments_total']} "
              f"n_degenerate_rotation={c['n_degenerate_rotation']}")


if __name__ == "__main__":
    main()
