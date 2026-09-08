#!/usr/bin/env python
"""彙整 26 格結果（humanoidmaze-medium-stitch-v0）：24 主格總表＋膝點曲線 PNG＋
因子有效性表＋推薦格＋2 個段長 8 對照格的單獨比較。

跟 gk_scan/aggregate.py（ant 版）同一套推薦邏輯，只在「候選池」跟「膝點/曲線」
都只用 24 主格（跟 ant 完全同格，可比性優先，任務明講）；段長 8 的兩格對照
不進入推薦候選池，只在 summary.json 的 extra_seglen8_comparison 另外報告
（跟同 G,K 的段長 4 格並排比較 val MSE / 因子有效性）。

用法：
  python aggregate.py

讀 results/G*_K*.json（24 主格 + G16_K16_L8/G1_K32_L8 兩格），寫
results/summary.json + results/knee_curve.png。
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hd_common as hc

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
GS = [1, 2, 4, 8, 16, 32]
KS = [8, 16, 32, 64]
EXTRA_CELLS = [(16, 16, 8), (1, 32, 8)]  # (G, K, seg_len)

THRESH_DEAD_FACTOR_FRAC_OF_BASELINE = 0.01
THRESH_MEAN_ABS_OFFDIAG_CORR = 0.5


def find_knee(env_caps, env_vals):
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


def _cell_row(r, G, K):
    m = r["metrics"]
    fv = r["factor_validity"]
    capacity = G * float(np.log2(K))
    eps_effect = THRESH_DEAD_FACTOR_FRAC_OF_BASELINE * m["baseline_mse_val"]
    dead_factor = fv["min_delta_mse"] <= eps_effect
    corr_ok = (fv["mean_abs_offdiag_corr"] is None) or (fv["mean_abs_offdiag_corr"] < THRESH_MEAN_ABS_OFFDIAG_CORR)
    factor_healthy = (not dead_factor) and corr_ok
    return dict(
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
        train_seconds=r["timing"]["train_seconds"],
    )


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
            cells.append(_cell_row(r, G, K))

    if missing:
        print(f"⛔ 缺 {len(missing)} 主格未完成：{missing}")

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

    # --- 段長 8 對照格（不進候選池，跟同 G,K 的段長 4 格並排比較） ---
    extra_missing = []
    extra_compare = []
    cells_by_gk = {(c["G"], c["K"]): c for c in cells}
    for (G, K, L) in EXTRA_CELLS:
        tag = f"G{G}_K{K}_L{L}"
        path = os.path.join(RESULTS_DIR, f"{tag}.json")
        if not os.path.exists(path):
            extra_missing.append(tag)
            continue
        with open(path) as f:
            r = json.load(f)
        row8 = _cell_row(r, G, K)
        row8["seg_len"] = L
        row4 = cells_by_gk.get((G, K))
        row4_summary = None
        if row4 is not None:
            row4_summary = dict(
                seg_len=4, val_mse=row4["val_mse"], improve_pct=row4["improve_pct"],
                collapsed=row4["collapsed"], mean_abs_offdiag_corr=row4["mean_abs_offdiag_corr"],
                mean_top_dim_energy_frac=row4["mean_top_dim_energy_frac"],
            )
        extra_compare.append(dict(
            G=G, K=K, seg_len_8=row8, seg_len_4_same_GK=row4_summary,
        ))
    if extra_missing:
        print(f"⛔ 缺 {len(extra_missing)} 段長 8 對照格未完成：{extra_missing}")

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
        extra_seglen8_comparison=extra_compare,
        extra_seglen8_missing=extra_missing,
    )

    out_path = os.path.join(RESULTS_DIR, "summary.json")
    hc.save_json(summary, out_path)
    print(f"saved: {out_path}")

    knee_png = os.path.join(RESULTS_DIR, "knee_curve.png")
    hc.draw_knee_curve(cells, knee_png, knee_capacity=knee_capacity,
                       title="humanoidmaze-medium: val MSE vs total capacity G*log2(K)")
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

    print("\nseg_len=8 comparison cells:")
    for e in extra_compare:
        r8 = e["seg_len_8"]
        r4 = e["seg_len_4_same_GK"]
        print(f"  G={e['G']} K={e['K']}  L8: val_mse={r8['val_mse']:.5f} collapsed={r8['collapsed']} "
              f"offdiag_corr={r8['mean_abs_offdiag_corr']}  |  "
              f"L4: val_mse={(r4['val_mse'] if r4 else None)}")


if __name__ == "__main__":
    main()
