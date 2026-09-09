#!/usr/bin/env python
"""experiments/gen_sft/corpus_length_report_v2.py —— rewrite-v1 步3（刀數加碼）：讀三份語料
（2刀＝既有 corpus_aug_v1.pt，直接引用、本檔不重建；4刀/8刀＝本工單新建的
corpus_aug_v2_c{4,8}.pt），印樣本數帳 + 長度分佈（p25/50/75）並排比較——直接回答
acceptance_criteria 第1點「短譜偏置有沒有發生」。

⛔ 新檔，只讀既有 .pt、算 percentile、印表格，不碰任何既有檔案，不跑任何模擬/訓練。
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402
import torch  # noqa: E402

import gen_sft_common as gc  # noqa: E402

CORPORA = [
    ("2刀(引用,corpus_aug_v1.pt)", os.path.join(gc.RESULTS_DIR, "corpus_aug_v1.pt")),
    ("4刀(corpus_aug_v2_c4.pt)", os.path.join(gc.RESULTS_DIR, "corpus_aug_v2_c4.pt")),
    ("8刀(corpus_aug_v2_c8.pt)", os.path.join(gc.RESULTS_DIR, "corpus_aug_v2_c8.pt")),
]


def pct(arr, q):
    return float(np.percentile(arr, q)) if len(arr) else None


def main():
    print("=== 語料帳 + 長度分佈並排比較（p25/50/75，code_seq len）===")
    header = (f"{'corpus':>26} | {'n_orig':>7} | {'n_sub':>7} | {'n_total':>8} | {'sub%':>6} | "
              f"{'all_p25':>7} | {'all_p50':>7} | {'all_p75':>7} | {'sub_p25':>7} | {'sub_p50':>7} | {'sub_p75':>7}")
    print(header)
    print("-" * len(header))
    rows = {}
    for label, path in CORPORA:
        blob = torch.load(path, weights_only=False)
        all_samples = blob["train"] + blob["val"]
        sub_samples = [s for s in all_samples if s.get("is_aug")]
        all_lens = np.array([len(s["code_seq"]) for s in all_samples])
        sub_lens = np.array([len(s["code_seq"]) for s in sub_samples])
        n_orig = len(all_samples) - len(sub_samples)
        sub_frac = len(sub_samples) / len(all_samples)
        row = dict(
            n_orig=n_orig, n_sub=len(sub_samples), n_total=len(all_samples), sub_frac=sub_frac,
            all_p25=pct(all_lens, 25), all_p50=pct(all_lens, 50), all_p75=pct(all_lens, 75),
            all_min=int(all_lens.min()), all_max=int(all_lens.max()),
            sub_p25=pct(sub_lens, 25), sub_p50=pct(sub_lens, 50), sub_p75=pct(sub_lens, 75),
            sub_min=int(sub_lens.min()) if len(sub_lens) else None,
            sub_max=int(sub_lens.max()) if len(sub_lens) else None,
            cut_seed=blob["meta"].get("aug_cut_seed"),
            n_cuts=blob["meta"].get("aug_n_cuts_per_episode"),
            ledger=blob["meta"].get("aug_ledger"),
            path=path,
        )
        rows[label] = row
        print(f"{label:>26} | {row['n_orig']:7d} | {row['n_sub']:7d} | {row['n_total']:8d} | "
              f"{row['sub_frac']*100:5.1f}% | "
              f"{row['all_p25']:7.1f} | {row['all_p50']:7.1f} | {row['all_p75']:7.1f} | "
              f"{row['sub_p25']:7.1f} | {row['sub_p50']:7.1f} | {row['sub_p75']:7.1f}")

    print("\n=== ledger / cut_seed 明細（可重現性紀錄）===")
    for label, row in rows.items():
        print(f"{label}: n_cuts={row['n_cuts']} cut_seed={row['cut_seed']} ledger={row['ledger']}")

    print("\n=== 短譜偏置判讀 ===")
    labels = [lbl for lbl, _ in CORPORA]
    all_p50s = [rows[lbl]["all_p50"] for lbl in labels]
    sub_p50s = [rows[lbl]["sub_p50"] for lbl in labels]
    sub_fracs = [rows[lbl]["sub_frac"] for lbl in labels]
    print("整體語料(all, 原樣本+子樣本混合) p50 隨刀數變化："
          + "  ->  ".join(f"{lbl}={v:.1f}" for lbl, v in zip(labels, all_p50s)))
    print("子樣本本身(sub_only) p50 隨刀數變化："
          + "  ->  ".join(f"{lbl}={v:.1f}" for lbl, v in zip(labels, sub_p50s)))
    print("子樣本佔整體語料比例："
          + "  ->  ".join(f"{lbl}={v*100:.1f}%" for lbl, v in zip(labels, sub_fracs)))
    all_monotonic_down = all(all_p50s[i] >= all_p50s[i + 1] for i in range(len(all_p50s) - 1))
    sub_stable = (max(sub_p50s) - min(sub_p50s)) <= 2.0  # 子樣本本身分佈是否大致不變（質性參考，非工單門檻）
    print(f"\n判讀：整體語料 all_p50 隨刀數增加單調不增：{all_monotonic_down}"
          f"（{' -> '.join(f'{v:.1f}' for v in all_p50s)}）")
    print(f"      子樣本本身 sub_p50 三者差 <= 2.0（大致穩定，非工單既有門檻，質性參考）：{sub_stable}"
          f"（{' / '.join(f'{v:.1f}' for v in sub_p50s)}）")
    if all_monotonic_down and sub_stable:
        print("      => 短譜偏置的機制正是「子樣本比例上升」而非「子樣本本身變更短」："
              "個別子樣本長度分佈大致不變，但它在整體教材裡的佔比從 2刀"
              f"{sub_fracs[0]*100:.0f}% 一路推高到 8刀{sub_fracs[-1]*100:.0f}%，"
              "把整體教材的長度中位數往短的方向拖——短譜偏置確實發生，且機制可解釋（非雜訊）。")
    elif all_monotonic_down:
        print("      => 整體語料變短，但子樣本本身分佈也有明顯移動——短譜偏置發生，機制不只是比例效應。")
    else:
        print("      => 整體語料長度中位數沒有隨刀數增加而單調下降——短譜偏置沒有清楚出現。")

    out_path = os.path.join(gc.RESULTS_DIR, "corpus_length_report_v2.json")
    with open(out_path, "w") as f:
        json.dump(dict(rows=rows, all_monotonic_down=bool(all_monotonic_down), sub_stable=bool(sub_stable)),
                  f, indent=2, default=str)
    print(f"\nsaved: {out_path}")


if __name__ == "__main__":
    main()
