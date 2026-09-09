#!/usr/bin/env python
"""experiments/gen_sft/analyze_cuts_sweep.py —— rewrite-v1 步3（刀數加碼）：讀六顆新 eval
summary（4刀/8刀 x 3 seed），跟 2 刀版（工單原文引用數字，⛔ 不重新讀檔／不重新計算——工單
明文「⛔ 不重跑 2 刀」）並排，算主表 + 判準(1)(2) + 三選一判讀。

在證明什麼、判準是什麼（工單原文，逐字）：
  判準(1) 更好：R=8 重寫的 3-seed 均值超過 2 刀版的 .6639，差 >=.02 才算方向清楚。
  判準(2) 更穩：3 顆的 R=8 散佈（max-min 或 std）比 2 刀版（.637~.690, std≈.026）明顯縮小。
  兩條都不成立＝「2 刀已夠、加碼無益」，也是合法答案。
判準對 4 刀、8 刀「各自」相對 2 刀版分別判一次——工單標題是「2→4→8」的加碼掃描，兩個新
設定都要跟基準比，不是只看 8 刀一格。

⛔ 新檔，只讀 JSON、算數字、印表格，不碰任何既有檔案，不跑任何模擬。
"""
import json
import os
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")

SEEDS = [20260909, 20260910, 20260911]

# 2 刀版：工單原文直接引用的數字（來自 docs/NOTE-2026-09-09-midstart-aug.md §三 aug 三列）。
# ⛔ 本檔不重新讀 rewrite_eval_aug_s*_summary.json、不重新計算——工單明文「⛔ 不重跑 2 刀」。
REF_C2 = dict(
    label="2刀(引用)",
    R8=dict(zip(SEEDS, [0.6899, 0.6651, 0.6368])),
    Rinf=dict(zip(SEEDS, [0.5900, 0.5337, 0.5931])),
    source="docs/NOTE-2026-09-09-midstart-aug.md §三主表 aug 三列（工單原文引用，非本檔重跑）",
)

BETTER_DIFF_THRESHOLD = 0.02
# 「明顯縮小」工單沒給精確數字（同款質性用語見上一份 NOTE 姿態置換探針的 .10 自訂門檻慣例）。
# 本檔自訂：std 與 range 都要縮到 2 刀版的 <=70%（即縮小 >=30%）才算「明顯」，非工單既有數字，
# 在此明標，不冒充工單既有判準。
STABLER_RATIO_THRESHOLD = 0.70


def load(tag):
    p = os.path.join(RESULTS_DIR, f"{tag}_summary.json")
    with open(p) as f:
        return json.load(f), p


def per_leg(d, lbl):
    return d["results"][lbl]["per_leg_untimed_reach"]


def mean(xs):
    return sum(xs) / len(xs)


def sample_std(xs):
    return statistics.stdev(xs) if len(xs) > 1 else float("nan")


def pop_std(xs):
    return statistics.pstdev(xs)


def main():
    groups = {}
    paths = {}
    for nc in (4, 8):
        for lbl, rlbl in (("R8", "8"), ("Rinf", "inf")):
            vals = {}
            for s in SEEDS:
                d, p = load(f"rewrite_eval_c{nc}_s{s}")
                vals[s] = per_leg(d, rlbl)
                paths[f"c{nc}_s{s}_{lbl}"] = p
            groups[f"c{nc}_{lbl}"] = vals

    print("=== 主表：{2刀(引用), 4刀, 8刀} x 3 seed x {R8, inf} ===")
    header = f"{'cuts':>12} {'seed':>10} | {'R8':>8} | {'Rinf':>8}"
    print(header)
    print("-" * len(header))
    for s in SEEDS:
        print(f"{'2(引用)':>12} {s:>10} | {REF_C2['R8'][s]:8.4f} | {REF_C2['Rinf'][s]:8.4f}")
    for nc in (4, 8):
        for s in SEEDS:
            print(f"{nc:>12} {s:>10} | {groups[f'c{nc}_R8'][s]:8.4f} | {groups[f'c{nc}_Rinf'][s]:8.4f}")

    print("\n=== 均值 / 散佈（sample std ddof=1；population std ddof=0；max-min）===")
    summary = {}
    for name, vals_dict, ref_note in [
        ("2刀(引用)_R8", REF_C2["R8"], "工單引用"),
        ("2刀(引用)_Rinf", REF_C2["Rinf"], "工單引用"),
        ("4刀_R8", groups["c4_R8"], "本次量測"),
        ("4刀_Rinf", groups["c4_Rinf"], "本次量測"),
        ("8刀_R8", groups["c8_R8"], "本次量測"),
        ("8刀_Rinf", groups["c8_Rinf"], "本次量測"),
    ]:
        xs = list(vals_dict.values())
        m, sstd, pstd = mean(xs), sample_std(xs), pop_std(xs)
        rng = max(xs) - min(xs)
        summary[name] = dict(values=vals_dict, mean=m, sample_std=sstd, pop_std=pstd,
                             range=rng, min=min(xs), max=max(xs), note=ref_note)
        print(f"  {name:>14}: mean={m:.4f}  sample_std={sstd:.4f}  pop_std={pstd:.4f}  "
              f"range(max-min)={rng:.4f}  [{min(xs):.4f},{max(xs):.4f}]  ({ref_note})")

    # ---- 自檢：重算 2 刀版引用數字，確認跟工單給的 .6639/std約.026/.637~.690 對得上 ----
    c2_r8 = summary["2刀(引用)_R8"]
    print("\n=== 自檢：2 刀版引用數字重算核對（驗證我方法跟工單給的口徑一致，才敢套用到新數字）===")
    print("  工單給：mean=.6639  std約.026  range約.637~.690")
    print(f"  本檔重算：mean={c2_r8['mean']:.4f}  sample_std(ddof=1)={c2_r8['sample_std']:.4f}  "
          f"pop_std(ddof=0)={c2_r8['pop_std']:.4f}  range=[{c2_r8['min']:.4f},{c2_r8['max']:.4f}]")
    mean_ok = abs(c2_r8["mean"] - 0.6639) < 0.0001
    std_ok = abs(c2_r8["sample_std"] - 0.026) < 0.001
    print(f"  mean 對得上: {mean_ok}   sample_std(ddof=1) 對得上 std約.026: {std_ok}"
          f"  (工單的 std約.026 對應 sample std ddof=1 口徑，不是 population std ddof=0，"
          f"後者算出來是 {c2_r8['pop_std']:.4f}——本檔後續一律用 sample_std ddof=1 跟工單口徑對齊)")

    # ---- 判準(1) 更好：R8 均值差 >=.02（4刀、8刀各自跟 2刀比）----
    print("\n=== 判準(1) 更好：R8 3-seed 均值 - 2刀版.6639，差 >=.02 才算方向清楚 ===")
    verdict1 = {}
    for nc in (4, 8):
        m = summary[f"{nc}刀_R8"]["mean"]
        diff = m - c2_r8["mean"]
        clear = diff >= BETTER_DIFF_THRESHOLD
        verdict1[nc] = dict(mean=m, diff=diff, clear=clear)
        print(f"  {nc}刀: mean={m:.4f}  diff(vs 2刀)={diff:+.4f}  "
              f"{'PASS(更好,方向清楚)' if clear else 'FAIL(沒有清楚變好)'}")

    # ---- 判準(2) 更穩：散佈明顯縮小（std 與 range 都要縮到 <=70%）----
    print(f"\n=== 判準(2) 更穩：R8 散佈(std/range) 比 2刀版(std={c2_r8['sample_std']:.4f}, "
          f"range={c2_r8['range']:.4f}) 明顯縮小（本檔自訂門檻：縮到<={STABLER_RATIO_THRESHOLD*100:.0f}%）===")
    c2_std, c2_range = c2_r8["sample_std"], c2_r8["range"]
    verdict2 = {}
    for nc in (4, 8):
        s = summary[f"{nc}刀_R8"]
        std_ratio = s["sample_std"] / c2_std if c2_std > 0 else float("inf")
        range_ratio = s["range"] / c2_range if c2_range > 0 else float("inf")
        clear = std_ratio <= STABLER_RATIO_THRESHOLD and range_ratio <= STABLER_RATIO_THRESHOLD
        verdict2[nc] = dict(std=s["sample_std"], range=s["range"], std_ratio=std_ratio,
                            range_ratio=range_ratio, clear=clear)
        print(f"  {nc}刀: std={s['sample_std']:.4f}(比值{std_ratio:.2f})  "
              f"range={s['range']:.4f}(比值{range_ratio:.2f})  "
              f"{'PASS(明顯縮小)' if clear else 'FAIL(沒有明顯縮小)'}"
              f"  [門檻<={STABLER_RATIO_THRESHOLD*100:.0f}%為本檔自訂，非工單既有數字]")

    # ---- 三選一判讀 ----
    print("\n=== 三選一判讀（每個 n_cuts 各自判一次，任一判準成立即為「有價值的答案」）===")
    final = {}
    for nc in (4, 8):
        v1, v2 = verdict1[nc]["clear"], verdict2[nc]["clear"]
        if v1 and v2:
            verdict = "更好且更穩"
        elif v1:
            verdict = "更好（判準1成立，判準2不成立）"
        elif v2:
            verdict = "更穩（判準2成立，判準1不成立）"
        else:
            verdict = "2刀已夠、加碼無益（兩條判準都不成立）"
        final[nc] = verdict
        print(f"  {nc}刀 vs 2刀: {verdict}")

    out = dict(main_table={f"c{nc}_{lbl}": groups[f"c{nc}_{lbl}"] for nc in (4, 8) for lbl in ("R8", "Rinf")},
               ref_c2=REF_C2, summary_stats=summary, self_check=dict(mean_ok=mean_ok, std_ok=std_ok),
               verdict1_better=verdict1, verdict2_stabler=verdict2, final_verdict=final,
               stabler_ratio_threshold=STABLER_RATIO_THRESHOLD, better_diff_threshold=BETTER_DIFF_THRESHOLD,
               source_files=paths)
    out_path = os.path.join(RESULTS_DIR, "cuts_sweep_analysis.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nsaved: {out_path}")


if __name__ == "__main__":
    main()
