# 對抗判決逐條銷項（提案層）

u 編碼想像軌跡。每條列設計回應及可查證據，未把提案通過當 production 修復。

| 判決／要求 | v2 交代 | 證據／責任 |
|---|---|---|
| B1 identity 最優 | 真 flow＋own iterates 每輪 same-plan J；舊盒內機率/identity probe 作負對照 | C2/C3；b1_baselines.py，M1/M3 |
| B2 主線未碰 | scratch:1720–1734及1779唯一owner M2；共用compose三caller | ARCH所有權、WIRING-SKETCH；current AST原branch+total親跑FAIL |
| F5反轉 | 撤回；EMA建議vs self/R>=2呈裁；沿用CONS=ema原式 | C4、CHANGE-ORDER、consistency_test.py |
| exposure橋斷 | same-plan合格動作教師訓head；no-grad gap／valid／缺值null | C5，oracle_suite clean好但sample壞探針，M4 |
| M3空殼 | 真CPU訓練/B1/scaling/identity/baselines/mutations/merged三入口trace | M3 scientific_audit stub為主體，独立integration套件；非僅metrics self-check |
| C1 dtype/finite | cond可異dtype；autocast輸出放寬；nonfinite交scaler | objective_suite CPU bf16 & NaN propagation |
| R2 rounds0 | nf+anchor；refine相關零；不需provider；原測試合併必過 | C1/C7、r2_zero_rounds親跑PASS |
| six mutations | 實際替換六份實作，逐一assertion捕獲 | six_mutations.stdout.txt，6/6 |
| anchor-positive名稱 | 改anchor-gradient；新增exposure須扣其梯度後比anchor | composition_suite、C5 |
| loss重複 | M1 kernel/M2 compose，三caller僅adapter差異 | C6、WIRING-SKETCH；merged call trace列硬gate |
| optional：方法獨立gate | top只檢有效行為；EMA/total/梯度策略專屬另檔 | integration_test.py vs consistency_test.py/mutations.py |
| survey警告 | 29條讀畢，三候選+FQL coupling；BoN/GD計量不宣稱普遍勝 | ARCH、b1_baselines ledger；R6診斷 |
| 不可虛報 | raw feature兩次FAIL及feedback限制公开，codebase唯讀 | EVIDENCE/manifest，try-*輸出 |
