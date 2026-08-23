#!/usr/bin/env python3
"""把 job array 各格寫出的 JSON 收成一張表。

判讀（三件都要過，缺一個就會被騙）：
  ① 沒塌     —— batch 內兩兩 cosine 要往真 e_target 的值靠，離 1.0 遠
  ② 有內容   —— 從 u 還原路徑中點，要贏過「只用 (s,g) 內插」
  ③ 內容對   —— 跟真 e_target 的 cosine 要是正的、夠大

⚠️ 為什麼三件都要：實測過 var(VICReg) 的塌度 0.0319 是全場最漂亮的，
   但路徑資訊輸給內插、cos 只有 +0.229 —— 它只是把 u 撐成互不相干的噪聲。
   散開 ≠ 有內容。
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(HERE, "results")

rows = []
for fn in sorted(os.listdir(RES)) if os.path.isdir(RES) else []:
    if fn.startswith("anticollapse_") and fn.endswith(".json"):
        with open(os.path.join(RES, fn)) as f:
            rows.append(json.load(f))

if not rows:
    print(f"{RES} 裡還沒有結果")
    sys.exit(0)

# 排序：先看有沒有贏內插，再看塌度
rows.sort(key=lambda r: (r["info"] >= r["info_interp"] * 0.95, r["collapse"]))

w = 84
print("=" * w)
print(f"{'變體':<14} {'塌度':>8} {'有效維':>7} {'路徑資訊':>10} {'贏內插':>7} {'cos(真et)':>10} {'macc':>7}")
print("-" * w)
for r in rows:
    win = "✔" if r["info"] < r["info_interp"] * 0.95 else "✘"
    print(f"{r['tag']:<14} {r['collapse']:>8.4f} {r.get('edim', 0):>7} {r['info']:>10.4f} "
          f"{win:>7} {r['cos_et']:>+10.3f} {r['macc']:>7.3f}")
print("-" * w)
r0 = rows[0]
print(f"{'真 e_target':<14} {r0['collapse_et']:>8.4f} {r0.get('edim_et', 0):>7} "
      f"{r0['info_interp']:>10.4f} {'—':>7} {1.0:>+10.3f}")
print("=" * w)

passed = [r for r in rows if r["info"] < r["info_interp"] * 0.95 and r["collapse"] < 0.5]
print(f"\n三項都過的：{', '.join(r['tag'] for r in passed) if passed else '（沒有）'}")
print(f"共 {len(rows)} 個變體")
