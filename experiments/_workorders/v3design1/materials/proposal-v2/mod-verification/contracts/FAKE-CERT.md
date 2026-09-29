# Fake 自證與效力範圍

u 編碼想像軌跡。這些是開工前接口替身，worker solution 全部仍為 NotImplementedError；self-check 不等於 worker 完工。

| 替身 | 能保證 | 不保證 | 親跑自證 |
|---|---|---|---|
| fakes.refinement_terms | flow初值＋自身展開、逐輪quality、EMA target、full BPTT/cond、R0 | maze cost 可信度、非線性 convergence | objective_suite 手展 recurrence、梯度、混精/非有限 |
| fakes.compose/Adapter | density/clean features/anchor、sample 一次、total 項、exposure、R0 | scratch FSQ/intent/teacher mask 實際接入 | composition_suite independent sum/gradient/call count |
| oracle_fake.Quality/gauge | toy 真 branch+goal、frozen decoder、same-plan action、實算 no-grad gap | 真 TrajDecoder 的 OOD 準確度、ant dynamics | oracle_suite swap/collapse/clean可讀但sample不可讀 |
| acceptance.assess_run | 固定 observable threshold、不接受 identity | 真訓練、真外部成功率、baseline fairness | schema tests；M3 仍須做 live scientific_audit |
| toy.TinyRefine | 小型可訓 MLP＋明示 toy sign feedback | 現有 production RefineOperator 架構可解 | 三 seed CPU train；raw features 失敗也公開 |
| current_inline AST | 執行唯讀當前 branch 的原運算/依賴 | 整支 trainer 環境、optimizer/data loader 配置 | source path/hash/line + F1負對照 |

mutation 六種以實際程式替換執行捕獲；不採只檢 source 文字或寫死 PASS。C4 的 EMA test 單獨跑；頂層 integration 只判 sample/label 因果隔離、off-manifold品質、模式、scaling、exposure。
