#!/usr/bin/env python3
"""收 success-rate rollout 的結果。

⚠️ 檔名分兩代：舊的沒有 chunk 欄（`rollout_{cons}_K{k}_c{cond}_s{seed}.json`），
   新的有（`..._ch{chunk}_s{seed}.json`）。⛔ 不要用檔名解析設定 —— JSON 裡面
   本來就記了完整設定，直接讀內容。
"""
import json
import os
from collections import defaultdict
from statistics import mean, pstdev

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
rows = []
for fn in sorted(os.listdir(RES)):
    if fn.startswith("rollout_") and fn.endswith(".json"):
        with open(os.path.join(RES, fn)) as f:
            rows.append(json.load(f))
if not rows:
    print("還沒有 rollout 結果")
    raise SystemExit(0)

groups = defaultdict(list)
for r in rows:
    # ⚠️ 舊檔沒有 env 欄（那時只跑 medium-navigate），補上預設值
    groups[(r.get("env", "pointmaze-medium-navigate-v0").replace("pointmaze-", "").replace("-v0", ""),
            r["cons"], r["K"], r["cond"], r["chunk"], r.get("steps2", 2000))].append(r)

# bc ＝ 誠實地板（獨立 head、只吃 cond、cond 已 detach）
# null_u ＝ ⚠️ 把 u 塞 0，是【OOD 探針】不是地板 —— 訓練時 head 沒看過零。
# shuf ＝ 別人的 u（分布對、內容錯）—— 不製造 OOD，是目前最乾淨的「u 有沒有被讀」探針
keys = ["bc", "null_u", "shuf", "R0", "R1", "R3", "R5", "R8"]
hdr = f"{'設定':<40}" + "".join(f"{k:>10}" for k in keys) + "  seed"
print("=" * len(hdr))
print(hdr)
print("-" * len(hdr))
for (env, cons, K, cond, ch, st), rs in sorted(groups.items()):
    label = f"{env} {cons} K{K} c{cond} ch{ch} st{st}"
    line = f"{label:<40}"
    for k in keys:
        v = [r["rates"][k] for r in rs if k in r["rates"]]
        line += f"{mean(v):>10.3f}" if v else f"{'—':>10}"
    print(line + f"  {len(rs)}")
    if len(rs) > 1:
        sd = f"{'  (seed 標準差)':<40}"
        for k in keys:
            v = [r["rates"][k] for r in rs if k in r["rates"]]
            sd += f"{pstdev(v):>10.3f}" if len(v) > 1 else f"{'—':>10}"
        print(sd)
print("=" * len(hdr))
print("bc＝誠實地板　null_u＝⚠️OOD探針不是地板　shuf＝別人的u（分布對內容錯）　R*＝refine 幾輪")
