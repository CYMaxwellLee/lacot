#!/usr/bin/env python
"""彙整 24 格結果：總表、膝點曲線 PNG、因子有效性表、推薦格。

用法：
  python aggregate.py

讀 results/G*_K*.json（24 份），寫 results/summary.json + results/knee_curve.png。
推薦邏輯（自訂門檻，非任務指定，見下方 THRESH_* 常數與 summary.json 的
recommendation.reasoning 欄）：
  1. 候選 = 未塌陷（any_group_collapsed=False）
  2. 且「因子有效」：每組 delta_mse 都 > baseline_mse_val 的 1%（沒有死因子）
     且 mean_abs_offdiag_corr < 0.5（組間方向不太重疊）；G=1 沒有組間相關可言，
     視為「因子有效」恆真（只有一個因子，無重疊問題）。
  3. 在候選裡選「總容量離膝點最近」的格，同距離時取 val MSE 較低者。
  4. 若候選集合是空的：明確標註「無格三條件同時滿足」，退而只用膝點+未塌陷兩條件，
     並在 summary.json 標記 recommendation.fallback=true。
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gk_common as gc

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
GS = [1, 2, 4, 8, 16, 32]
KS = [8, 16, 32, 64]

THRESH_DEAD_FACTOR_FRAC_OF_BASELINE = 0.01  # min_delta_mse 門檻 = 1% baseline_mse_val
THRESH_MEAN_ABS_OFFDIAG_CORR = 0.5


def find_knee(env_caps, env_vals):
    """在 (env_caps, env_vals) 這條下緣線上找 elbow：正規化後離首尾連線最遠的點。"""
    if len(env_caps) < 3:
        return env_caps[0] if env_caps else None
    x = np.array(env_caps, dtype=np.float64)
    y = np.array(env_vals, dtype=np.float64)
    xn = (x - x.min()) / max(x.max() - x.min(), 1e-9)
    yn = (y - y.min()) / max(y.max() - y.min(), 1e-9)
    p0 = np.array([xn[0], yn[0]])
    p1 = np.array([xn[-1], yn[-1]])
    line_vec = p1 - p0
    line_len = np.linalg.norm(line_vec)
    if line_len < 1e-9:
        return x[0]
    line_unit = line_vec / line_len
    dists = []
    for i in range(len(xn)):
        p = np.array([xn[i], yn[i]]) - p0
        proj_len = np.dot(p, line_unit)
        proj = proj_len * line_unit
        perp = p - proj
        dists.append(np.linalg.norm(perp))
    knee_i = int(np.argmax(dists))
    return float(x[knee_i])


def main():
    cells = []
    missing = []
    for G in GS:
        for K in KS:
            path = os.path.join(RESULTS_DIR, f"G{G}_K{K}.json")
            if not os.path.exists(path):
                missing.append(f"G{G}_K{K}")
                continue
            with open(path) as f:
                r = json.load(f)
            m = r["metrics"]
            fv = r["factor_validity"]
            capacity = G * float(np.log2(K))
            eps_effect = THRESH_DEAD_FACTOR_FRAC_OF_BASELINE * m["baseline_mse_val"]
            dead_factor = fv["min_delta_mse"] <= eps_effect
            corr_ok = (fv["mean_abs_offdiag_corr"] is None) or (fv["mean_abs_offdiag_corr"] < THRESH_MEAN_ABS_OFFDIAG_CORR)
            factor_healthy = (not dead_factor) and corr_ok
            cells.append(dict(
                G=G, K=K, capacity=capacity,
                val_mse=m["recon_mse_val"], train_mse=m["recon_mse_train"],
                baseline_mse_val=m["baseline_mse_val"],
                improve_pct=m["val_improvement_over_baseline_pct"],
                collapsed=m["any_group_collapsed"],
                steps_used=m["total_steps_used"], extended=r["plateau"]["extended"],
                restarts_total=m["restarts_total"],
                mean_delta_mse=fv["mean_delta_mse"], min_delta_mse=fv["min_delta_mse"],
                max_delta_mse=fv["max_delta_mse"],
                mean_abs_offdiag_corr=fv["mean_abs_offdiag_corr"],
                mean_top_dim_energy_frac=fv["mean_top_dim_energy_frac"],
                dead_factor=bool(dead_factor), corr_ok=bool(corr_ok),
                factor_healthy=bool(factor_healthy),
                eps_effect=eps_effect,
            ))

    if missing:
        print(f"⛔ 缺 {len(missing)} 格未完成：{missing}")

    # lower envelope for knee detection
    by_cap = {}
    for c in cells:
        by_cap.setdefault(c["capacity"], []).append(c["val_mse"])
    env_caps = sorted(by_cap.keys())
    env_vals = [min(by_cap[cap]) for cap in env_caps]
    knee_capacity = find_knee(env_caps, env_vals)

    candidates = [c for c in cells if (not c["collapsed"]) and c["factor_healthy"]]
    fallback = False
    if not candidates:
        fallback = True
        candidates = [c for c in cells if not c["collapsed"]]
    reasoning_notes = []
    if fallback:
        reasoning_notes.append("沒有格同時滿足『未塌陷 + 因子有效』，退回只用『未塌陷』篩選，見 dead_factor/corr_ok 逐格數字。")

    recommended = None
    if candidates and knee_capacity is not None:
        candidates_sorted = sorted(candidates, key=lambda c: (abs(c["capacity"] - knee_capacity), c["val_mse"]))
        recommended = candidates_sorted[0]
    elif candidates:
        recommended = sorted(candidates, key=lambda c: c["val_mse"])[0]
        reasoning_notes.append("knee 偵測失敗（點數不足或全部同容量），改用候選裡 val MSE 最低者。")

    for c in cells:
        c["recommended"] = bool(recommended is not None and c["G"] == recommended["G"] and c["K"] == recommended["K"])

    summary = dict(
        cells=sorted(cells, key=lambda c: (c["G"], c["K"])),
        missing_cells=missing,
        knee_capacity_bits=knee_capacity,
        thresholds=dict(
            collapse_top1_thresh=0.5,
            dead_factor_frac_of_baseline=THRESH_DEAD_FACTOR_FRAC_OF_BASELINE,
            mean_abs_offdiag_corr_thresh=THRESH_MEAN_ABS_OFFDIAG_CORR,
        ),
        recommendation=dict(
            fallback=fallback,
            reasoning_notes=reasoning_notes,
            cell=(dict(G=recommended["G"], K=recommended["K"]) if recommended else None),
        ),
    )

    out_path = os.path.join(RESULTS_DIR, "summary.json")
    gc.save_json(summary, out_path)
    print(f"saved: {out_path}")

    knee_png = os.path.join(RESULTS_DIR, "knee_curve.png")
    gc.draw_knee_curve(cells, knee_png, knee_capacity=knee_capacity)
    print(f"saved: {knee_png}")

    print(f"\nknee_capacity_bits = {knee_capacity}")
    if recommended:
        print(f"recommended cell: G={recommended['G']} K={recommended['K']} "
              f"val_mse={recommended['val_mse']:.5f} capacity={recommended['capacity']:.2f} "
              f"fallback={fallback}")
    else:
        print("⛔ 沒有任何未塌陷的格，無法推薦任何一格")

    print("\n24-cell table (G, K, capacity, val_mse, collapsed, factor_healthy, steps_used):")
    for c in sorted(cells, key=lambda c: (c["G"], c["K"])):
        print(f"  G={c['G']:2d} K={c['K']:2d} cap={c['capacity']:6.2f} val_mse={c['val_mse']:.5f} "
              f"collapsed={c['collapsed']!s:5s} factor_healthy={c['factor_healthy']!s:5s} "
              f"steps={c['steps_used']}")


if __name__ == "__main__":
    main()
