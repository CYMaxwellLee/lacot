#!/usr/bin/env python3
"""收 token head 的 2^3 全因子 ablation。

判讀重點：
  增益 = 1 - MSE(餵真 e_target) / MSE(u 餵零)
       ＝ head 從 u 裡真的抽出了多少。⛔ 不是「模型好不好」，是「head 讀不讀得到 u」。

⚠️ 一定要看 seed 之間的散布：某一格贏 0.5% 而 seed 標準差有 1.5%，那不算贏。
"""
import json
import os
import re
import sys
from statistics import mean, pstdev

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
ORDER = ["tok_000", "tok_r00", "tok_0w0", "tok_00u", "tok_rw0", "tok_r0u", "tok_0wu", "tok_rwu", "head_s"]

runs = {}
for fn in sorted(os.listdir(RES)):
    m = re.fullmatch(r"fronts_(.+)_s(\d+)\.json", fn)
    if not m or m.group(1) not in ORDER:
        continue
    with open(os.path.join(RES, fn)) as f:
        runs.setdefault(m.group(1), []).append(json.load(f))

if not runs:
    print(f"{RES} 裡還沒有 ablation 結果")
    sys.exit(0)

def agg(rows, key):
    v = [r[key] for r in rows]
    return mean(v), (pstdev(v) if len(v) > 1 else 0.0)

w = 78
print("=" * w)
print(f"{'':<16} {'readout':>7} {'wide':>5} {'uproj':>6} {'增益':>15} {'seed':>5} {'參數':>8}")
print("-" * w)
base = None
for tag in ORDER:
    rows = runs.get(tag)
    if not rows:
        continue
    g, sd = agg(rows, "gain")
    if tag == "tok_000":
        base = g
    sw = rows[0].get("switches") or {}
    if tag == "head_s":
        mark = "  ← concat 對照（不是 token head）"
        cols = f"{'—':>7} {'—':>5} {'—':>6}"
    else:
        mark = ""
        cols = (f"{'✔' if sw.get('deep_readout') else '·':>7} "
                f"{'✔' if sw.get('wide') else '·':>5} "
                f"{'✔' if sw.get('u_proj') else '·':>6}")
        if base is not None and tag != "tok_000":
            mark = f"  ({100*(g-base):+.1f} pt vs 基準)"
    print(f"{tag:<16} {cols} {100*g:>+9.1f}% ±{100*sd:>3.1f} {len(rows):>5} "
          f"{rows[0]['params']/1e6:>7.1f}M{mark}")
print("-" * w)

# 主效果：開 vs 關 的平均差（全因子才算得出來）
# ⚠️ 開關在 tag 的第 4~6 個字元（"tok_" 佔了 0~3），⛔ 不是 5~7。
#    第一版寫成 5+i，於是 readout 那一行印的其實是 wide 的數字 —— 標籤跟數字對不上，
#    而且它【不會報錯】（uproj 那格才越界），差點就照著讀了。
toks = {t: [r["gain"] for r in rs] for t, rs in runs.items() if t.startswith("tok_")}
if len(toks) == 8:
    allv = [g for v in toks.values() for g in v]
    sd_run = pstdev(allv) if len(allv) > 1 else 0.0
    print("\n主效果（八格全因子，同一個開關 開的四格 減 關的四格）：")
    print(f"  每邊各 {len(allv)//2} 次，隨機誤差約 ±{100*sd_run*(2/(len(allv)/2))**0.5:.1f} pt")
    for i, nm in enumerate(["readout", "wide", "uproj"]):
        on = [g for t, v in toks.items() if t[4 + i] != "0" for g in v]
        off = [g for t, v in toks.items() if t[4 + i] == "0" for g in v]
        print(f"  {nm:<9} {100*(mean(on)-mean(off)):+.1f} pt")
else:
    print(f"\n（八格還沒到齊，現在有 {len(toks)} 格，主效果先不算）")
print("=" * w)

# ── 協定對齊 2x2（+2）──────────────────────────────────────────
PROTO = ["pr_cat_ded", "pr_cat_shr", "pr_tok_ded", "pr_tok_shr", "pr_tokru_ded", "pr_tokru_shr"]
pruns = {}
for fn in sorted(os.listdir(RES)):
    m = re.fullmatch(r"fronts_(pr_.+)_s(\d+)\.json", fn)
    if m and m.group(1) in PROTO:
        with open(os.path.join(RES, fn)) as f:
            pruns.setdefault(m.group(1), []).append(json.load(f))

if pruns:
    print("\n協定對齊（增益＝head 有多依賴 u。⚠️ 兩種量法【不能互相比大小】，只能同一欄上下比）")
    print("=" * 62)
    print(f"{'':<14} {'專用 A對B':>14} {'共用 C自己比':>15}")
    print("-" * 62)
    for arch, lbl in [("cat", "concat"), ("tok", "token 基準"), ("tokru", "token r0u")]:
        cells = []
        for proto in ("ded", "shr"):
            rs = pruns.get(f"pr_{arch}_{proto}")
            if rs:
                v = [r["gain"] for r in rs]
                cells.append(f"{100*mean(v):+7.1f}% ±{100*(pstdev(v) if len(v)>1 else 0):>3.1f}")
            else:
                cells.append(f"{'—':>13}")
        print(f"{lbl:<14} {cells[0]:>14} {cells[1]:>15}")
    print("=" * 62)
