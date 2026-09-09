#!/usr/bin/env python
"""experiments/gen_sft/analyze_dyn.py —— 主人裁示①②：收六顆 dyn 訓練+eval 的表，
套用工單給定的判準，算兩問判讀。CPU-only、極輕量（純讀 JSON 算數），走一個小 sbatch
只是為了在 Slurm dependency 鏈的最後一棒（等全部訓練+eval 完工才跑），本身不需要
任何運算資源。

⛔ 新檔（dyn 系列），不改動任何既有檔案；baseline（2 刀版）數字直接讀既有
   rewrite_eval_aug_s{seed}_summary.json（本 worktree 既有檔案，rewrite-v1 步2 產出，
   非重新輸入的手抄數字）。

判準（工單原文，逐字照抄）：
  Q1：R=8 重寫的 3-seed 均值超過 2 刀版 .6639（差 >= .02）「或」3 顆散佈（std）
      明顯小於 2 刀版的 ≈.026 —— 兩個子判準各自獨立檢查，都不成立才是「無差」。
      「明顯小於」工單沒給精確數字，本檔自訂 <= 一半（.013）當「明顯」門檻，明標
      非工單既有判準（同 NOTE-2026-09-09-midstart-aug.md §五姿態置換探針的門檻自訂
      慣例）。
  Q2：擾動組相對無擾動組的 R8 均值差 >= .02 算有效，否則無效/無差（兩問各自「無差」
      都是合法答案，如實報，不因為想要正面結果就放寬)。
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
RESULTS_DIR = os.path.join(HERE, "results")

MEAN_DELTA_THRESHOLD = 0.02
STD_CLEARLY_SMALLER_RATIO = 0.5   # 自訂「明顯小於」門檻：dyn std <= ratio * baseline std
PERTURB_DELTA_THRESHOLD = 0.02

BASELINE_SEEDS = [20260909, 20260910, 20260911]
DYN_SEEDS = [20260909, 20260910, 20260911]


def load(path):
    with open(path) as f:
        return json.load(f)


def mean_std(xs):
    import numpy as np
    a = np.asarray(xs, dtype=np.float64)
    # ⚠️ 2026-09-09 校正：一開始這裡用 population std（ddof=0），baseline R8 算出來是
    # .0217，但工單原文引用的 2 刀版 std 是「≈.026」——反推發現工單那個數字是 sample
    # std（ddof=1，除以 n-1）：baseline [.6899,.6651,.6368] 的 ddof=1 std=.02657≈.026,
    # 跟 ddof=0 的 .0217 對不上。3 個點的小樣本，ddof=1（無偏估計）本來就是比較標準
    # 的選擇，改成跟工單引用數字口徑一致，重跑後兩種算法下 Q1「更穩」子判準的結論
    # 沒有變（都是「未明顯小」），只是顯示數字精度更對得上，見 NOTE §四footnote。
    return float(a.mean()), float(a.std(ddof=1))  # sample std, ddof=1 —— 對齊工單引用的 baseline std≈.026 口徑


def main():
    # ---- baseline（2 刀版，既有檔案直接引用，不重跑）----
    baseline_r8, baseline_rinf = [], []
    baseline_rows = []
    for seed in BASELINE_SEEDS:
        p = os.path.join(RESULTS_DIR, f"rewrite_eval_aug_s{seed}_summary.json")
        d = load(p)
        r8 = d["results"]["8"]["per_leg_untimed_reach"]
        rinf = d["results"]["inf"]["per_leg_untimed_reach"]
        baseline_r8.append(r8)
        baseline_rinf.append(rinf)
        baseline_rows.append(dict(seed=seed, R8=r8, Rinf=rinf, source=p))
    base_r8_mean, base_r8_std = mean_std(baseline_r8)
    base_rinf_mean, base_rinf_std = mean_std(baseline_rinf)

    def load_group(tag_prefix):
        r8s, rinfs, rows = [], [], []
        for seed in DYN_SEEDS:
            eval_p = os.path.join(RESULTS_DIR, f"rewrite_eval_dyn_{tag_prefix}_s{seed}_summary.json")
            train_p = os.path.join(RESULTS_DIR, f"gensft_dyn_{tag_prefix}_s{seed}_train_summary.json")
            ed = load(eval_p)
            td = load(train_p) if os.path.exists(train_p) else None
            r8 = ed["results"]["8"]["per_leg_untimed_reach"]
            rinf = ed["results"]["inf"]["per_leg_untimed_reach"]
            r8s.append(r8)
            rinfs.append(rinf)
            rows.append(dict(seed=seed, R8=r8, Rinf=rinf, eval_source=eval_p,
                             train_summary=(dict(total_steps_run=td["total_steps_run"],
                                                 best_step=td["best_step"],
                                                 smoke_drop=td["smoke"]["drop"],
                                                 held_out_top1_codes=td["best_held_out"]["top1_codes_only"],
                                                 held_out_top1_eos=td["best_held_out"]["top1_eos"],
                                                 cut_stats=td["dyn_meta"]["cut_stats"])
                                            if td is not None else None)))
        m, s = mean_std(r8s)
        mi, si = mean_std(rinfs)
        return dict(r8_values=r8s, r8_mean=m, r8_std=s, rinf_values=rinfs, rinf_mean=mi, rinf_std=si, rows=rows)

    nopert = load_group("nopert")
    pert = load_group("pert")

    # ---- Q1：dyn(無擾動) vs 2刀版 ----
    q1_mean_diff = nopert["r8_mean"] - base_r8_mean
    q1_better = q1_mean_diff >= MEAN_DELTA_THRESHOLD
    q1_std_threshold = STD_CLEARLY_SMALLER_RATIO * base_r8_std
    q1_more_stable = nopert["r8_std"] <= q1_std_threshold
    if q1_better and q1_more_stable:
        q1_verdict = "更好且更穩"
    elif q1_better:
        q1_verdict = "更好"
    elif q1_more_stable:
        q1_verdict = "更穩"
    else:
        q1_verdict = "無差"

    # ---- Q2：擾動 vs 無擾動 ----
    q2_mean_diff = pert["r8_mean"] - nopert["r8_mean"]
    q2_effective = q2_mean_diff >= PERTURB_DELTA_THRESHOLD
    q2_verdict = "有效" if q2_effective else "無效/無差"

    out = dict(
        criteria=dict(mean_delta_threshold=MEAN_DELTA_THRESHOLD,
                     std_clearly_smaller_ratio_selfdefined=STD_CLEARLY_SMALLER_RATIO,
                     perturb_delta_threshold=PERTURB_DELTA_THRESHOLD),
        baseline_2cut=dict(rows=baseline_rows, r8_mean=base_r8_mean, r8_std=base_r8_std,
                           rinf_mean=base_rinf_mean, rinf_std=base_rinf_std),
        dyn_nopert=nopert, dyn_pert=pert,
        q1_dyn_vs_2cut=dict(mean_diff=q1_mean_diff, better=q1_better,
                            std_threshold=q1_std_threshold, more_stable=q1_more_stable,
                            verdict=q1_verdict),
        q2_perturb_vs_nopert=dict(mean_diff=q2_mean_diff, effective=q2_effective, verdict=q2_verdict),
    )

    print("=== 主表：R8 / Rinf per-leg，3 組 x 3 seed ===")
    print(f"{'group':<12} {'seed':>10} | {'R8':>7} | {'Rinf':>7}")
    for r in baseline_rows:
        print(f"{'2cut(引用)':<12} {r['seed']:>10} | {r['R8']:.4f} | {r['Rinf']:.4f}")
    print(f"  -> mean={base_r8_mean:.4f} std={base_r8_std:.4f}  (Rinf mean={base_rinf_mean:.4f} std={base_rinf_std:.4f})")
    for r in nopert["rows"]:
        print(f"{'dyn_nopert':<12} {r['seed']:>10} | {r['R8']:.4f} | {r['Rinf']:.4f}")
    print(f"  -> mean={nopert['r8_mean']:.4f} std={nopert['r8_std']:.4f}  (Rinf mean={nopert['rinf_mean']:.4f} std={nopert['rinf_std']:.4f})")
    for r in pert["rows"]:
        print(f"{'dyn_pert':<12} {r['seed']:>10} | {r['R8']:.4f} | {r['Rinf']:.4f}")
    print(f"  -> mean={pert['r8_mean']:.4f} std={pert['r8_std']:.4f}  (Rinf mean={pert['rinf_mean']:.4f} std={pert['rinf_std']:.4f})")

    print(f"\n=== Q1：dyn(無擾動) vs 2刀版（門檻：差>=+{MEAN_DELTA_THRESHOLD} 或 std<=剩下不到一半({q1_std_threshold:.4f})）===")
    print(f"  mean_diff = {nopert['r8_mean']:.4f} - {base_r8_mean:.4f} = {q1_mean_diff:+.4f}  "
          f"{'>= +.02 (更好 PASS)' if q1_better else '< +.02 (未達更好門檻)'}")
    print(f"  dyn std = {nopert['r8_std']:.4f}  vs  threshold {q1_std_threshold:.4f}(=0.5x baseline std {base_r8_std:.4f})  "
          f"{'明顯小 (更穩 PASS)' if q1_more_stable else '未明顯小'}")
    print(f"  => 判讀：{q1_verdict}")

    print(f"\n=== Q2：擾動 vs 無擾動（門檻：差>=+{PERTURB_DELTA_THRESHOLD}）===")
    print(f"  mean_diff = {pert['r8_mean']:.4f} - {nopert['r8_mean']:.4f} = {q2_mean_diff:+.4f}  => 判讀：{q2_verdict}")

    out_path = os.path.join(RESULTS_DIR, "dyn_analysis.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\nsaved: {out_path}")


if __name__ == "__main__":
    main()
