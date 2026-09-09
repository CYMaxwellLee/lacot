#!/usr/bin/env python
"""experiments/gen_sft/analyze_bell_aug.py —— rewrite-v1 步2：讀六顆 eval summary，
算鐘形定讞（判準1）、增強判讀（判準2）、R=4 代價檢查，輸出主表/差值表 + 判讀 JSON。

在證明什麼、判準是什麼（工單原文，逐字）：
  判準1（鐘形定讞）：3 個原配方訓練 seed，各量 (R=8 - R=inf) 差值——全部 >0 且平均
    >= .03 算「重寫增益為真」。
  判準2（增強判讀）：增強版 R=8 的 per-leg 平均 >= .66（錨 .588 + .07 之外）算清楚贏；
    並檢查增強版 R=4 是否不再明顯低於 R=inf（病灶治對的直接證據）。

⛔ 新檔，只讀 JSON、算數字、印表格，不碰任何既有檔案，不跑任何模擬。
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")

ORIG_SEEDS = [20260909, 20260910, 20260911]
AUG_SEEDS = [20260909, 20260910, 20260911]
ANCHOR = 0.5875
BELL_DIFF_MEAN_THRESHOLD = 0.03
AUG_CLEAR_WIN_THRESHOLD = 0.66


def load(tag):
    p = os.path.join(RESULTS_DIR, f"{tag}_summary.json")
    with open(p) as f:
        return json.load(f), p


def per_leg(d, lbl):
    return d["results"][lbl]["per_leg_untimed_reach"]


def main():
    orig = {}
    orig_paths = {}
    for s in ORIG_SEEDS:
        d, p = load(f"rewrite_eval_orig_s{s}")
        orig[s] = d
        orig_paths[s] = p
    aug = {}
    aug_paths = {}
    for s in AUG_SEEDS:
        d, p = load(f"rewrite_eval_aug_s{s}")
        aug[s] = d
        aug_paths[s] = p

    print("=== 主表：6 顆 x R 格 per-leg ===")
    header = f"{'recipe':>6} {'seed':>10} | " + " | ".join(f"{lbl:>7}" for lbl in ["4", "8", "16", "inf"])
    print(header)
    print("-" * len(header))
    main_table = {}
    for s in ORIG_SEEDS:
        row = {}
        for lbl in ["8", "inf"]:
            row[lbl] = per_leg(orig[s], lbl)
        main_table[f"orig_s{s}"] = row
        print(f"{'orig':>6} {s:>10} | {'':>7} | {row['8']:7.4f} | {'':>7} | {row['inf']:7.4f}")
    for s in AUG_SEEDS:
        row = {}
        for lbl in ["4", "8", "16", "inf"]:
            row[lbl] = per_leg(aug[s], lbl)
        main_table[f"aug_s{s}"] = row
        print(f"{'aug':>6} {s:>10} | {row['4']:7.4f} | {row['8']:7.4f} | {row['16']:7.4f} | {row['inf']:7.4f}")

    # ---- 判準1：鐘形定讞 ----
    print("\n=== 判準1：鐘形定讞（原配方 3 seed，(R=8 - R=inf) 差值）===")
    diffs = []
    inf_vals = []
    for s in ORIG_SEEDS:
        d8, dinf = main_table[f"orig_s{s}"]["8"], main_table[f"orig_s{s}"]["inf"]
        diff = d8 - dinf
        diffs.append(diff)
        inf_vals.append(dinf)
        print(f"  seed={s}  R8={d8:.4f}  Rinf={dinf:.4f}  diff={diff:+.4f}")
    all_positive = all(d > 0 for d in diffs)
    mean_diff = sum(diffs) / len(diffs)
    bell_verdict = all_positive and (mean_diff >= BELL_DIFF_MEAN_THRESHOLD)
    print(f"  全部 >0: {all_positive}   平均 diff={mean_diff:+.4f}  (門檻 >= {BELL_DIFF_MEAN_THRESHOLD})")
    print(f"  判準1結果：{'PASS —— 重寫增益為真' if bell_verdict else 'FAIL —— 重寫增益不算「為真」（不滿足全部>0 且平均>=.03）'}")

    # ---- 訓練抖動幅度（順便量出來，報出來——這個數字本身有價值）----
    import statistics
    jitter_mean = statistics.mean(inf_vals)
    jitter_std = statistics.pstdev(inf_vals)
    jitter_min, jitter_max = min(inf_vals), max(inf_vals)
    print(f"\n=== 訓練抖動幅度（原配方 3 seed 的 R=inf per-leg，對錨點 {ANCHOR}）===")
    for s, v in zip(ORIG_SEEDS, inf_vals):
        print(f"  seed={s}  Rinf={v:.4f}  對錨點差={v-ANCHOR:+.4f}")
    print(f"  mean={jitter_mean:.4f}  std={jitter_std:.4f}  range=[{jitter_min:.4f},{jitter_max:.4f}]  "
          f"錨點={ANCHOR}  mean-錨點={jitter_mean-ANCHOR:+.4f}")

    # ---- 判準2：增強判讀 ----
    print("\n=== 判準2：增強判讀（增強版 3 seed 的 R=8 per-leg 平均）===")
    aug_r8_vals = [main_table[f"aug_s{s}"]["8"] for s in AUG_SEEDS]
    for s, v in zip(AUG_SEEDS, aug_r8_vals):
        print(f"  seed={s}  R8={v:.4f}")
    aug_r8_mean = sum(aug_r8_vals) / len(aug_r8_vals)
    aug_verdict = aug_r8_mean >= AUG_CLEAR_WIN_THRESHOLD
    print(f"  平均={aug_r8_mean:.4f}  門檻>={AUG_CLEAR_WIN_THRESHOLD}")
    print(f"  判準2結果：{'PASS —— 增強版清楚贏' if aug_verdict else 'FAIL —— 增強版沒有清楚贏'}")

    # ---- R=4 代價檢查（病灶治對的直接證據）----
    print("\n=== R=4 代價檢查：增強版 R=4 是否不再明顯低於增強版 R=inf ===")
    r4_checks = []
    for s in AUG_SEEDS:
        r4v = main_table[f"aug_s{s}"]["4"]
        rinfv = main_table[f"aug_s{s}"]["inf"]
        delta = r4v - rinfv
        r4_checks.append(delta)
        print(f"  seed={s}  R4={r4v:.4f}  Rinf={rinfv:.4f}  (R4-Rinf)={delta:+.4f}  "
              f"{'R4 仍低於 Rinf' if delta < 0 else 'R4 未低於 Rinf（含打平/反超）'}")
    r4_mean_delta = sum(r4_checks) / len(r4_checks)
    # 對照：原始（未增強、單 seed）NOTE-2026-09-09-rewrite-receding.md 記錄的 R4 代價
    orig_note_r4, orig_note_rinf = 0.5415, 0.5875
    orig_note_delta = orig_note_r4 - orig_note_rinf
    print(f"  平均 (R4-Rinf)={r4_mean_delta:+.4f}   對照：原始未增強單 seed 的 (R4-Rinf)={orig_note_delta:+.4f}"
          f"（來自 docs/NOTE-2026-09-09-rewrite-receding.md §2.2，R=4 .5415 vs Rinf .5875，非本次量測）")
    penalty_gone = r4_mean_delta > orig_note_delta  # 沒有嚴謹判準門檻，質性比較：代價是否比原始版縮小/消失/反轉
    print(f"  質性判讀（無工單既有精確門檻，比較「代價有沒有比原始版縮小」）："
          f"{'代價縮小/消失（支持病灶治對）' if penalty_gone else '代價沒有縮小（不支持病灶治對）'}")

    # ---- R=16 甜蜜點漂移（附帶看一下）----
    print("\n=== 附：增強版 R=16（看甜蜜點是否漂移）===")
    for s in AUG_SEEDS:
        print(f"  seed={s}  R16={main_table[f'aug_s{s}']['16']:.4f}")

    out = dict(
        main_table=main_table,
        bell_curve=dict(diffs=dict(zip(ORIG_SEEDS, diffs)), all_positive=all_positive,
                        mean_diff=mean_diff, threshold=BELL_DIFF_MEAN_THRESHOLD, verdict=bell_verdict),
        training_jitter=dict(values=dict(zip(ORIG_SEEDS, inf_vals)), mean=jitter_mean, std=jitter_std,
                             min=jitter_min, max=jitter_max, anchor=ANCHOR, mean_minus_anchor=jitter_mean - ANCHOR),
        aug_verdict=dict(r8_values=dict(zip(AUG_SEEDS, aug_r8_vals)), mean=aug_r8_mean,
                         threshold=AUG_CLEAR_WIN_THRESHOLD, verdict=aug_verdict),
        r4_penalty_check=dict(deltas=dict(zip(AUG_SEEDS, r4_checks)), mean_delta=r4_mean_delta,
                              orig_note_r4=orig_note_r4, orig_note_rinf=orig_note_rinf,
                              orig_note_delta=orig_note_delta, penalty_reduced_qualitative=penalty_gone),
        best_orig_seed=max(ORIG_SEEDS, key=lambda s: main_table[f"orig_s{s}"]["8"]),
        best_aug_seed=max(AUG_SEEDS, key=lambda s: main_table[f"aug_s{s}"]["8"]),
        source_files=dict(orig={s: orig_paths[s] for s in ORIG_SEEDS}, aug={s: aug_paths[s] for s in AUG_SEEDS}),
    )
    out_path = os.path.join(RESULTS_DIR, "bell_aug_analysis.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nbest_orig_seed(by R8)={out['best_orig_seed']}  best_aug_seed(by R8)={out['best_aug_seed']}")
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
