# CPU 研究迭代帳

u 編碼想像軌跡。以下是提案小世界研究；本輪沒有生產資料訓練、GPU或生產patch。

| 試次 | 構造／預算 | 結果 | 保存 |
|---|---|---|---|
| raw MLP A | 2D真Flow sample；3→24→2 residual MLP，lr=.008，300步×3seeds | seed1品質FAIL，深度增加變差 | try-reference.txt |
| raw MLP B | 同架構，lr=.003，600步×3seeds | seed1仍FAIL，非只多訓練即可 | try-reference-600.txt |
| feedback MLP | 加 toy decoder sign feature，5→24→2，lr=.003，600步×3seeds | 三seed過。final另把cond按4重複固定，完整data×sample route factorial探針 | try-feedback.txt；最終proposal_reference.stdout.txt |
| 真 RefineOperator | 真Flow 4D、既有LayerNorm residual、hidden32、lr=.003、700步×3seeds；配對同計畫的解析雙路端點 | seed0 FAIL，seed1/2可過；保留失敗，不調寬threshold | real_operator_diagnostic.stdout.txt |

所有最終命令與原始stderr/exit由run_evidence.py保存；final source hash在manifest。探索期輸出不是最終程式逐位元重播，以上列明其與最終程式的參數／features差异；沒有宣稱那些舊輸出由目前唯一配置產生。

最終baseline ledger的秒數欄名明示不含共同的初始flow.sample時間，BoN另含新增3次sample；是固定初值上的增量成本，不是宣稱完整端到端walltime預算匹配。真環境合併gate需另量同walltime、同離線訓練預算、成功率/模式覆蓋。toy真值是解析世界，production教師可信度尚待M4交證。
