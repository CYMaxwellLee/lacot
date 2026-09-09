#!/usr/bin/env python
"""加測：開環漂移的增長率有沒有變平緩 —— 最好顆 vs sigma=0 對照顆。

【在證明什麼、判準是什麼】
證明：噪音 fine-tune 過的 decoder，開環執行時 xy 漂移隨 chunk index 成長的
速度（斜率）有沒有比 sigma=0 對照顆低——這是「decoder 看過偏移姿態後執行變
穩」的機制性證據，不只是終點的 per-leg 數字。
判準：在同一批 200 題（schedule，跟 eval_all.py 同一組 build_tasks 產出）上，
逐 chunk 邊界記錄 xy 漂移（沿用 run_drift_analysis.run_one_drift，import 未改
一行），比較兩條 p50 曲線：
  (a) 終點漂移（j=50）：最好顆是否明顯低於 sigma=0；
  (b) 成長斜率：早段 slope(j:1→10) 跟晚段 slope(j:10→50) 分別比較兩顆——
      最好顆的斜率如果系統性比 sigma=0 低（尤其晚段），就是「變平緩」的證據；
      斜率打平或更高，就是沒有這個效果，如實報。
  分不出來也要報（不是只有「有變平緩／沒有」兩個合法答案，也可能是「訊號不夠
  乾淨」，見下面的落地判讀）。

流程（有硬 gate）：
  STEP 1：REUSE CHECK —— 用本檔對 run_one_drift 的呼叫方式，重放「原始未動」
    ckpt 的 200 題，completion_ratio 與 accounting_table 幾個 j 點的
    d_cont_xy_p50 都要跟已存檔的 drift_summary.json 一致（同代碼路徑、同資料、
    同 200 題，沒有理由不一致）。不一致 = 本檔重用 run_one_drift 的方式有 bug，
    停手，不繼續比較 sigma=0 vs 最好顆。
  STEP 2：sigma=0 vs 最好顆（tag 從 eval_all.py 存的 eval_summary.json 讀
    gif_tag，除非 --best-tag 覆寫），200 題全量 run_one_drift，逐 chunk
    邊界的 xy 漂移（completed+failed 兩組合併，跟 run_drift_analysis.py 印
    accounting_table 那行的定義完全一致——那邊本來就是
    `xy_drift_by_j_completed[j] + xy_drift_by_j_failed[j]` 兩組合併著看）。

⛔ 不改動任何既有檔案；只 import 同目錄上一層 walk_verify/drift_analysis 的
   run_drift_analysis（run_one_drift / band_stats，純函式重用，import 未改
   一行）與它間接帶出來的 wv_common / drift_plots.draw_percentile_band。
⛔ CPU-only，跑在 CPU sbatch（drift_compare.sbatch，不掛 --gres）。
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
WV_DIR = os.path.normpath(os.path.join(HERE, "..", "walk_verify"))
DA_DIR = os.path.join(WV_DIR, "drift_analysis")
sys.path.insert(0, DA_DIR)

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("HIP_VISIBLE_DEVICES", "")
os.environ.setdefault("MUJOCO_GL", "osmesa")

import numpy as np  # noqa: E402

import run_drift_analysis as da  # noqa: E402 - 既有腳本當模組 import，未修改一行
import wv_common as wv  # noqa: E402 - 既有共用模組，未修改（經 da 的 sys.path 設定可 import）

RESULTS_DIR = os.path.join(HERE, "results")
EVAL_SUMMARY_JSON = os.path.join(RESULTS_DIR, "eval_summary.json")
DRIFT_GT_JSON = os.path.join(DA_DIR, "results", "drift_summary.json")


def ckpt_path(tag):
    if tag in ("orig", "original"):
        return da.DEFAULT_CKPT
    return os.path.join(RESULTS_DIR, f"ckpt_{tag}.pt")


def run_arm1_pooled(env, u, model, raw, tasks, rho):
    """對 200 題全跑 run_one_drift（原封 import），回傳 completion_ratio +
    每個 chunk index j 的 xy 漂移『completed+failed 兩組合併』list——這跟
    run_drift_analysis.py 印 accounting_table 那行的定義完全一致。"""
    n_completed = 0
    xy_by_j = defaultdict(list)
    for task in tasks:
        r = da.run_one_drift(env, u, model, raw, task, rho)
        if r["completed"]:
            n_completed += 1
        for j, xd in zip(r["boundary_j"], r["boundary_xy_drift"]):
            xy_by_j[int(j)].append(float(xd))
    return n_completed / len(tasks), xy_by_j


def slope(xs_map, j_a, j_b):
    if j_a not in xs_map or j_b not in xs_map or j_b == j_a:
        return None
    a = np.median(xs_map[j_a]) if xs_map[j_a] else float("nan")
    b = np.median(xs_map[j_b]) if xs_map[j_b] else float("nan")
    if not (np.isfinite(a) and np.isfinite(b)):
        return None
    return float((b - a) / (j_b - j_a))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-traj", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20260908)
    ap.add_argument("--best-tag", type=str, default=None,
                     help="不給就從 eval_summary.json 的 gif_tag 讀")
    ap.add_argument("--ctrl-tag", type=str, default="sigma0")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", type=str, default=os.path.join(RESULTS_DIR, "drift_compare_summary.json"))
    args = ap.parse_args()

    import torch
    torch.set_num_threads(max(1, args.threads))
    rho = da.RHO_DEFAULT
    t0 = time.time()
    print(f"=== drift_compare（schedule, {args.n_traj} 題, seed={args.seed}, rho={rho}）===")

    ruler = json.load(open(da.DEFAULT_RULER))
    fall_line = ruler["torso_z"]["fall_line_p1"]
    raw = wv.load_npz(da.DD, da.DATASET, "val")
    tasks, skipped = da.trb.build_tasks(raw, args.n_traj, args.seed, ruler, rho)
    assert len(tasks) >= args.n_traj, "⛔ 可用軌跡不足，停手回報"
    print(f"tasks: {len(tasks)} usable, skipped={len(skipped)}")
    env = wv.make_env(da.DD, da.DATASET)
    u = env.unwrapped

    # ================= STEP 1：REUSE CHECK（原始未動 ckpt vs 已存檔 drift_summary.json）=================
    print("\n=== STEP 1：REUSE CHECK（原始 ckpt，比對 drift_summary.json）===")
    if not os.path.exists(DRIFT_GT_JSON):
        print(f"⛔ 找不到 {DRIFT_GT_JSON}，無法核對，停手回報。")
        sys.exit(1)
    gt = json.load(open(DRIFT_GT_JSON))
    orig_model, _ = wv.load_dict_v1(da.DEFAULT_CKPT)
    compl_orig, xy_by_j_orig = run_arm1_pooled(env, u, orig_model, raw, tasks, rho)
    compl_ok = abs(compl_orig - gt["completion_ratio_arm1"]) < 1e-9
    print(f"completion_ratio: mine={compl_orig:.10f} vs gt={gt['completion_ratio_arm1']:.10f} "
          f"-> {'PASS' if compl_ok else 'FAIL'}")
    row_checks = []
    for row in gt["accounting_table"]:
        j = row["j"]
        vals = xy_by_j_orig.get(j, [])
        med = float(np.median(vals)) if vals else float("nan")
        ok = np.isfinite(med) and abs(med - row["d_cont_xy_p50"]) < 1e-6
        row_checks.append(dict(j=j, mine=med, gt=row["d_cont_xy_p50"], ok=bool(ok)))
        print(f"  j={j:>3}  mine={med:.6f}  gt={row['d_cont_xy_p50']:.6f}  {'PASS' if ok else 'FAIL'}")
    reuse_ok = compl_ok and all(r["ok"] for r in row_checks)
    if not reuse_ok:
        print("⛔ REUSE CHECK 失敗：本檔重放 run_one_drift 的方式跟已存檔的 drift_summary.json 對不上，"
              "停手回報，不比較 sigma=0 vs 最好顆。")
        wv.save_json(dict(status="REUSE_CHECK_FAILED", completion_check=dict(mine=compl_orig,
                     gt=gt["completion_ratio_arm1"], ok=compl_ok), row_checks=row_checks), args.out)
        sys.exit(1)
    print("REUSE CHECK PASS（completion_ratio 與 accounting_table 全部核對點一致）")

    # ================= tags =================
    best_tag = args.best_tag
    if best_tag is None:
        if not os.path.exists(EVAL_SUMMARY_JSON):
            print(f"⛔ 找不到 {EVAL_SUMMARY_JSON} 且未指定 --best-tag，停手回報。")
            sys.exit(1)
        ev = json.load(open(EVAL_SUMMARY_JSON))
        best_tag = ev["gif_tag"]
        print(f"best_tag 從 eval_summary.json 讀到：{best_tag}（gif_tag_note: {ev.get('gif_tag_note')}）")
    ctrl_tag = args.ctrl_tag
    print(f"比較對象：ctrl={ctrl_tag}  best={best_tag}")

    # ================= STEP 2：ctrl vs best =================
    print(f"\n=== STEP 2：{ctrl_tag} vs {best_tag}（200 題全量 run_one_drift）===")
    curves = {}
    for tag in [ctrl_tag, best_tag]:
        cp = ckpt_path(tag)
        if not os.path.exists(cp):
            print(f"⛔ 找不到 {cp}，停手回報。")
            sys.exit(1)
        m, _ = wv.load_dict_v1(cp)
        compl, xy_by_j = run_arm1_pooled(env, u, m, raw, tasks, rho)
        j_max = max(xy_by_j.keys())
        xs_all = list(range(0, j_max + 1))
        band = da.band_stats(xy_by_j, xs_all)
        curves[tag] = dict(completion_ratio=compl, band=band, xy_by_j=xy_by_j, j_max=j_max)
        print(f"[{tag}] completion_ratio={compl:.3f}  j_max={j_max}  "
              f"d_xy_p50(j=1)={np.median(xy_by_j.get(1, [np.nan])):.4f}  "
              f"d_xy_p50(j=10)={np.median(xy_by_j.get(10, [np.nan])):.4f}  "
              f"d_xy_p50(j=50 or j_max)={np.median(xy_by_j.get(min(50, j_max), [np.nan])):.4f}")

    # ================= 斜率比較 =================
    print("\n=== 增長斜率比較（p50 xy drift, m/chunk）===")
    slopes = {}
    for tag in [ctrl_tag, best_tag]:
        xy = curves[tag]["xy_by_j"]
        s_early = slope(xy, 1, 10)
        s_late = slope(xy, 10, 50) if 50 in xy else slope(xy, 10, curves[tag]["j_max"])
        slopes[tag] = dict(slope_j1_10=s_early, slope_j10_50=s_late)
        print(f"  [{tag}] slope(j:1→10)={s_early}  slope(j:10→50)={s_late}")
    flattened = None
    if all(v is not None for s in slopes.values() for v in s.values()):
        best_flatter_early = slopes[best_tag]["slope_j1_10"] < slopes[ctrl_tag]["slope_j1_10"]
        best_flatter_late = slopes[best_tag]["slope_j10_50"] < slopes[ctrl_tag]["slope_j10_50"]
        flattened = dict(early=bool(best_flatter_early), late=bool(best_flatter_late))
        print(f"  最好顆斜率比對照顆低：早段(j1-10)={best_flatter_early}  晚段(j10-50)={best_flatter_late}")
    else:
        print("  ⛔ 有斜率算不出來（某個 j 點沒有樣本），不下『有沒有變平緩』的判斷，如實標記")

    # ================= 存圖 =================
    png = os.path.join(RESULTS_DIR, "drift_compare_curve_xy.png")
    da.draw_percentile_band(
        [dict(label=f"{ctrl_tag} (control, completion={curves[ctrl_tag]['completion_ratio']:.3f})",
              color_idx=7, xs=curves[ctrl_tag]["band"]["xs"], p25=curves[ctrl_tag]["band"]["p25"],
              p50=curves[ctrl_tag]["band"]["p50"], p75=curves[ctrl_tag]["band"]["p75"]),
         dict(label=f"{best_tag} (best, completion={curves[best_tag]['completion_ratio']:.3f})",
              color_idx=2, xs=curves[best_tag]["band"]["xs"], p25=curves[best_tag]["band"]["p25"],
              p50=curves[best_tag]["band"]["p50"], p75=curves[best_tag]["band"]["p75"]),
         dict(label="original ckpt (reuse-check reference)", color_idx=4,
              xs=list(range(0, max(xy_by_j_orig.keys()) + 1)),
              p50=[float(np.median(xy_by_j_orig.get(j, [np.nan]))) for j in range(0, max(xy_by_j_orig.keys()) + 1)],
              p25=[float(np.percentile(xy_by_j_orig.get(j, [np.nan]), 25)) if xy_by_j_orig.get(j) else np.nan
                   for j in range(0, max(xy_by_j_orig.keys()) + 1)],
              p75=[float(np.percentile(xy_by_j_orig.get(j, [np.nan]), 75)) if xy_by_j_orig.get(j) else np.nan
                   for j in range(0, max(xy_by_j_orig.keys()) + 1)])],
        png, "drift-compare: xy drift vs chunk index (schedule, pooled completed+failed)",
        "chunk index j (state right before chunk j starts)", "xy drift vs teacher xy (m)")
    print(f"saved: {png}")

    # ================= 存 json =================
    out = dict(
        status="OK", ctrl_tag=ctrl_tag, best_tag=best_tag,
        reuse_check=dict(status="PASS", completion=dict(mine=compl_orig, gt=gt["completion_ratio_arm1"]),
                         row_checks=row_checks),
        curves={tag: dict(completion_ratio=curves[tag]["completion_ratio"], band=curves[tag]["band"])
               for tag in [ctrl_tag, best_tag]},
        slopes=slopes, flattened=flattened,
        artifacts=dict(curve_png=png),
        config=dict(n_traj=args.n_traj, seed=args.seed, rho=rho),
        wall_seconds=time.time() - t0,
    )
    wv.save_json(out, args.out)
    print(f"saved: {args.out}")
    print(f"=== done wall={time.time()-t0:.1f}s ===")


if __name__ == "__main__":
    main()
