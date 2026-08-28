"""smoke：驗 dev_eval 的 sep 判準與靈敏度標籤符號（2026-08-28 修）。

⭐ 每個 case 都寫明【為什麼期望是這個答案】—— 不然測試自己壞了也看不出來。
"""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from lacot.dev_eval import paired_diff, sanity_check, _mcnemar_p

fails = []

def rows(bits):
    return [dict(idx=i, success=b, tier=0, steps=10, bfs_dist=5, best_dist=0.1)
            for i, b in enumerate(bits)]

# ── case 1：昨天那個坑 ── 300 題裡只有 4 題不一致、且全同向
#    真答案：丟四次銅板全同面，exact p = 2*(1/16) = 0.125 ⇒ ⛔ 什麼都沒證明
n = 300
a = [1] * n
b = [1] * n
for i in range(4):
    b[i] = 0                      # a 贏 4 題、b 贏 0 題 ⇒ discordant = 4，全同向
pd = paired_diff(rows(a), rows(b))
print(f"case1  mean={pd['mean']:+.4f} CI={[round(x,4) for x in pd['ci95']]} "
      f"p={pd['mcnemar_p']:.4f}  discordant={pd['n_a_only']}+{pd['n_b_only']}")
if abs(pd["mcnemar_p"] - 0.125) > 1e-9:
    fails.append(f"case1 McNemar p 應為 0.125，得到 {pd['mcnemar_p']}")
ci_says_sep = (pd["ci95"][0] > 0) or (pd["ci95"][1] < 0)
if not ci_says_sep:
    fails.append("case1 舊版 CI 判準應該【誤判】成分得開 ⇒ 這個 case 才有意義（換一組數字）")

res = sanity_check({"x": rows(a), "y": rows(b)}, report_pairs=[("x", "y")])
ref = [ln for ln in res["notes"] if ln.startswith("[參考]")][0]
print(f"       {ref}")
if "分不開" not in ref:
    fails.append(f"case1 新判準應判【分不開】(p=0.125>0.05)，實際：{ref}")

# ── case 2：真的分得開 ── 40 題不一致、35 比 5
a2, b2 = [1] * n, [1] * n
for i in range(35):
    b2[i] = 0
for i in range(35, 40):
    a2[i] = 0
pd2 = paired_diff(rows(a2), rows(b2))
print(f"case2  mean={pd2['mean']:+.4f} p={pd2['mcnemar_p']:.2e} "
      f"discordant={pd2['n_a_only']}+{pd2['n_b_only']}")
res2 = sanity_check({"x": rows(a2), "y": rows(b2)}, report_pairs=[("x", "y")])
ref2 = [ln for ln in res2["notes"] if ln.startswith("[參考]")][0]
print(f"       {ref2}")
if "分不開" in ref2:
    fails.append(f"case2 差 10 個百分點、p 極小，應判【分得開】，實際：{ref2}")

# ── case 3：靈敏度標籤的符號要跟數值一致 ──────────────────────
#    sens_pair=("random","bc")，paired_diff 算的是 random − bc ⇒ random 爛 ⇒ 必須是【負】的
rnd = rows([0] * 250 + [1] * 50)      # random 成功率 0.167
bc  = rows([1] * 270 + [0] * 30)      # bc     成功率 0.90
# ⚠️ bc_rerun 要是【真的重跑】：同一顆模型換一條 action-noise stream ⇒ 少數題會翻面。
#    ⛔ 不可以直接把 bc 本人放進去 —— 那是 case6 在抓的退化輸入。
bc_rr = rows([1] * 268 + [0, 0] + [1, 1] + [0] * 28)   # 4 題不一致、2 比 2
res3 = sanity_check({"random": rnd, "bc": bc, "bc_rerun": bc_rr})
sens = [ln for ln in res3["notes"] if ln.startswith("[靈敏度]")][0]
print(f"case3  {sens}")
label_lhs = sens.split("]")[1].split("=")[0].strip()      # "random − bc"
value = float(sens.split("=")[1].split()[0])
if label_lhs != "random − bc":
    fails.append(f"case3 標籤應為 'random − bc'（paired_diff 算的順序），實際 '{label_lhs}'")
if value >= 0:
    fails.append(f"case3 random 比 bc 爛 ⇒ random − bc 必須為負，實際 {value:+.3f}")
print(f"       gates={res3['gates']} passed={res3['passed']}")
if not res3["passed"]:
    fails.append(f"case3 靈敏度＋特異度都該過，實際 {res3['gates']}")

# ── case 4：檢查器要能對故意寫壞的輸入叫 ────────────────────
#    ⭐ 一把不會叫的尺跟壞掉的尺長得一模一樣
#    random 是 50/300 = 0.167；sens_min = 0.30 ⇒ weak 要低於 0.467 才算「差距不夠大」
weak = rows([1] * 130 + [0] * 170)    # 0.433，差 0.267 < 0.30 ⇒ 靈敏度 gate 必須【擋下來】
weak_rr = rows([1] * 129 + [0] + [1] + [0] * 169)      # 同上，2 題不一致（真的重跑）
res4 = sanity_check({"random": rnd, "bc": weak, "bc_rerun": weak_rr})
print(f"case4  gates={res4['gates']} passed={res4['passed']}  (期望 sensitivity=False)")
if res4["gates"].get("sensitivity"):
    fails.append("case4 差距 0.267 < sens_min 0.30 時靈敏度 gate 應該擋下 ⇒ 沒擋 = 尺不會叫")

# ── case 5：_mcnemar_p 的邊界 ────────────────────────────────
cases = [((0, 0), 1.0), ((1, 0), 1.0), ((5, 0), 2 * (1 / 32)), ((3, 3), 1.0)]
for (nb, nc), want in cases:
    got = _mcnemar_p(nb, nc)
    if abs(got - want) > 1e-9:
        fails.append(f"case5 _mcnemar_p({nb},{nc}) 應為 {want}，得到 {got}")
print(f"case5  邊界 {len(cases)} 筆檢查完")

# ── case 6：特異度那格對【退化輸入】不准給滿分（2026-08-28 修）─────
#    🚨 舊版：兩臂逐位元相同 ⇒ 差恆為 0、CI 恆含 0 ⇒ 判 ✓
#       ⛔ 但那不是「沒有假訊號」，是【這格根本沒有驗到配對】。
#    ⚠️ 加 p 值救不了：nb+nc=0 ⇒ McNemar p 恆為 1.0 ⇒ 永遠不顯著。
#    ⭐ 主線真正的洞是【受測對象選錯】：bc 這條路徑不消耗 torch 亂數 ⇒ 換 tseed 是 no-op
#       ⇒ 它【必然】落進這個退化 case；有風險的是會抽樣的 lacot / shuf。
res6 = sanity_check({"random": rnd, "bc": bc, "bc_rerun": bc})   # ⛔ 兩臂是同一份 rows
spec6 = [ln for ln in res6["notes"] if ln.startswith("[特異度]")][0]
print(f"case6  {spec6}")
print(f"       gates={res6['gates']} passed={res6['passed']}  (期望 specificity=False)")
if res6["gates"].get("specificity"):
    fails.append("case6 兩臂逐位元相同時特異度 gate 應判 False ⇒ 判 True = 退化輸入拿滿分")
if _mcnemar_p(0, 0) != 1.0:
    fails.append("case6 立論前提：nb+nc=0 時 McNemar p 應為 1.0（⇒ 加 p 值救不了這格）")
# 真的重跑（有少數 discordant）則照樣要過 ⇒ ⛔ 這個修法不可以把正常的重跑也擋掉
if not sanity_check({"random": rnd, "bc": bc, "bc_rerun": bc_rr})["gates"]["specificity"]:
    fails.append("case6 真的重跑（4 題不一致）應該通過特異度 ⇒ 修法把正常情況也擋掉了")

print()
if fails:
    print("🚨 FAIL")
    for f in fails:
        print("  -", f)
    sys.exit(1)
print("✅ 6/6 PASS")
