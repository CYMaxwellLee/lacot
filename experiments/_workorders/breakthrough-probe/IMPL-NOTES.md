# 實作紀錄（尚未放行）

最新狀態：第四棒 v6.3 S2 合併修項已落碼；完整驗收與變異結果見本文末第四棒節。
以下第一、二棒 BLOCKED 段落保留為歷史，不代表目前接線狀態。`PRODUCTION_READY=False`。

BLOCKED: 共同底座／凍結源 SHA256 不符。指定 experiments/scratch_lacot_rollout.py
實際 b7d3a0d98f75fada7e101af73e597af5e08bdc6e1df947852ee764333004a234，
契約要求 276c68fba6c41a05c1e3c2bd5bf4b25c39642d0dbac3ca7aa57cce381056ecfc。
依工單紅線停止真模型接線，不 import、不替換路徑、不改 PIN、不改原檔。
Move 3 / Move 1 的正式 runtime、s33 encode/head 驗證及 GPU smoke 因此未完成。
本機邏輯自測的成功不得取代這項必要驗收；完整 selftest 必須以非零退出。

BLOCKED: 路徑構造器 R donor 表的 task 2 規則為「另一 gate 幾何」，
但 task 2 並非 task 1/3，無法唯一選定其中一個；輸出 null 與原因，不擅選。
這不影響卡上 Move 3 task 2 A/C，也不擴增 Move 1 題集至 task 2。

本機無 torch/ogbench runtime、GPU 或 jasmine 存取；numpy 可用。

## 完成範圍與未完成範圍

此目錄是可供 lead 檢閱的 BLOCKED 中間產物，**不是已完成的 GPU harness**。
兩個正式 harness、正式 harvest 與 smoke 入口皆以 exit 2 拒絕啟動；
無繞過 PIN 選項。即使恢復正確源，入口仍拒絕，直到接線工作實際完成。
`PRODUCTION_READY=False` 的必要自測也會失敗，避免只恢復 SHA 就宣稱可發射。

| 工單項 | 已實作 | 未完成／阻塞 |
|---|---|---|
| builder | gate0b 機器題單、兩路全格序、DAG、字典序命名、分岔／合流／互斥段、同深 floor w、task 2 唯一路、gate 模板、JSON＋sha、雙進程 bytes 確定性 | task 2 R donor 無唯一規則；demo oracle 未核實，全部 unverified |
| Move 3 | 1280 列計畫、64..79 seeds、Collector A fresh 分支接點、共同噪音函式、逐步鎖存核心、即刻換段 ledger、分類／預算函式 | 未 import／接線真模型，未實作正式環境 runner／產物 writer；cap/stuck 後的完整 trace 協定待對齊 |
| Move 1 | N/P/Q/R 計畫、gate self-rollout P 來源、固定注入 head 邊界 hash、128 點正規化 encode helper、torch 確定性設定／RNG snapshot、manipulation 和同 u 重跑 helper | s33／EMA loader、實際 encode_u→head 形狀自測、policy／環境 cache 重置、真實 P/Q 重跑、noise-band 校準／decode 歸屬、正式 writer 未完成 |
| harvest | 獨立事件分類、缺列／重列／seed／reset／污染／trace／注入拒收、n64 reset 參照載入器、落格與 E0–E4／九格純函式、fixture-readout.json | 正式端到端收割、實際 crossing／collision/stuck／round-trip evidence extractor、gate 與各層自動合讀尚未接線 |
| sbatch / smoke | jasmine 單卡、archive 寫入面、本地 MuJoCo temp、禁 pycache、逐項 smoke-plan.json | 無真 GPU smoke、無測速、無 ETA／配額放行；入口保持阻擋 |
| selftest | 本機 numpy＋標準庫；24 項，其中 22 項通過 | 2 項必要驗收失敗：source PIN、正式 adapter 完整性；exit 1 |
| notes | 決策、差異、輸入 hash audit、自測全文 | 本文不構成設計／發射放行 |

## 實作決策

- 路圖從本機 `/home/cymaxwelllee/Projects/ogbench/ogbench/locomaze/maze.py`
  用 AST 讀 literal；不 import ogbench。僅接受 ndarray `<i8` bytes hash 等於 gate0b
  的 `72e8e5978a6d24be3e9128613450424d516631390756fbc60ea187dd43cdf63a` 的唯一圖。
  該圖是純格序構造來源，不以其 BFS tie-break 代替 demo 標籤。
- builder 使用排序 BFS DAG 枚舉，再依整條格序命名 A/B。task 4 w 深度 4，
  route-A=(1,6)、route-B=(5,10)；task 5 深度 2，A=(1,3)、B=(3,1)；
  task 2 深度 9、w=(3,5)。每個輸出含輸入圖與 gate0b hash，JSON sorted keys、
  UTF-8、固定 compact separators、尾端 newline，無時間戳／亂數。
- donor 明確部分：task 4←5 ep5 A、5←4 ep4 A、gate 1←3 ep16 A、
  gate 3←1 ep10 A。task 2=null，附 blocker；未擴增 Move 1 題集。
- `common.reset_env` 保留 np.random.seed、action_space.seed、
  reset(seed=1000*task+ep) 三行協定。digest 與 read-only collector 實際函式交叉測試。
- flow sample 是 calls 唯一計數單位；head hash 次數不能當 flow sample 次數。
  A/B/C 共用 fresh-flow seed 與 CPU noise stream；原 Collector 的 A action 方法不加噪音，
  因此新 helper 明確對三臂施加同 σ=.05、clip→noise→clip，沒有改 collector。
- Move 3 state core 是 **mock 可執行核心**，不是 GPU rollout。cap/stuck 不切 g；
  達 w 時廢棄 chunk 剩餘動作並要求立即重抽。碰到繼續舊 chunk，記無效而非 d1-d5。
  reach 嚴格 `<1.875` 與目前唯讀源的 leg reach 式相符，但仍需正確 PIN 源重核。
- 有效性先於分類；超 budget 的 draw 保留列但不入任何分母。C 實際觀察到成功優先；
  無 C 成功且任一臂有效數<8／w-miss>50%，進觀測不足。無 d4 時合併 A/B 的
  d2/d3/d5 唯一最大；平手進歧義，不計世界主導數。
- E2 分母只用 14／6 配對題數，分子 P∨Q，嚴格 >50%。第二層九格純描述性，
  不製造 p 值。B/C exact 僅在同幾何層計算；<3 只作軸一降優先註記。
- `FixedInjection` 在傳入 head 的呼叫邊界 hash，檢查前後 u 不被改動。
  P/Q 重跑標為 determinism-verification，非新 draw。輔助函式 mock 通過不能
  證成真 head 已收到注入，故未發 E0 通過 receipt。
- 確定性清單依 [PyTorch 官方 randomness 文件](https://docs.pytorch.org/docs/stable/notes/randomness.html)：
  deterministic algorithms（warn_only=False）、固定 cuDNN backend、各 RNG 狀態存檔。
  另固定 math SDPA、禁 TF32、batch=1／float32／單執行緒，CUBLAS 設定於啟動前。
  機器上的 torch／CUDA 版本與模型行為仍需 smoke 驗，現有 mock 只驗旗標接法。
- harvest 不 import harness 的 classify；從 XY 與逐步 success、flow ledger 重算 d 類。
  幾何決策純函式共用於離線測試，正式 observation extractor 未完成。
  crossing verifier 必須明示提供；正式缺口不以「任意碰到 w」冒充穿越核實。
- 所有 synthetic fixture 顯式標 `synthetic=true`、`production_evidence=false`，
  production validator 拒收 synthetic。`fixture-readout.json` 不是探針判決。
- `input-audit.json` 是唯讀輸入快照，**不是執行 receipt**；真 receipt 建立必先通過 PIN。
  不寫假的 ckpt sha、GPU 硬體／版本、實跑 seeds 或 manipulation 成績。

## 尚須對齊的差異／參數

BLOCKED: 共同 trace 契約要求每 rollout 真正 H+1 點，而目前唯讀源 rollout 在
成功／terminated／truncated 時提早結束。卡未說明提早結束與 cap/stuck 的後續觀測
如何湊足真 H+1 點。本實作不 padding、不重置拼接、不假稱每集已觀察 1000 steps；
validator 拒絕短 trace／填補 trace。此處需拿正確 PIN 源對齊，不能在錯源上定稿。

BLOCKED: d4「專屬互斥段穿越核實」未指定完整穿越的機械判準（首入、全段有序、
分岔到合流或其他），卡的 Move 1 首入定義不能自行套作 Move 3 完整穿越。
收割器保留明示 verifier 接口，mock 只測分類分流，不把任選準則升為契約。

待 smoke 校準／凍結（目前無預設數字）：步/格帳與 w cap、stuck 判準、
同路重抽噪聲帶的重抽方式／統計口徑、XY 落格／撞牆證據抽取。
`cap_steps` 和 `noise_band` 必須顯式提供；並未以自己挑的數值啟動正式試驗。

BLOCKED: Move 1 卡的規模算式 20×4＋8×2＝96 未納 gate 的 8 次 N 前置跑。
按 N→P/R 且 N 全成功為 80＋24＝104，P/Q 每幾何各加一次符合性重跑再加 4，
最多 108；N 失敗 gate 題不跑 P/R、表示層不過不跑該層 P/Q，實際總量依閘減少。
計畫列反映臂清單、N 前置、條件執行，不修改卡／配額；ETA 與規模待 lead/vFinal 定版。

BLOCKED（本機資料可得性，非設計變更）：gate0b-result.json 只存 fingerprint_match，
不含 initial/goal 的原始 SHA。真正獨立驗 reset 需載入 receipt 指向的兩份 n64
/archive 產物並先驗檔 SHA；本機檔案不存在。`historical_fingerprints` 拒絕缺檔，
不信任新 rollout 自報的「fingerprint_match=true」。

## 檔案與重跑

- `builder.json` / `.sha256` 是實際由 gate0b＋已匹配圖產生的確定性產物。
- `smoke-plan.json` 完整列各幾何×臂與卡上負例；CPU/GPU 證據分別標記。
- 三支 sbatch 只有阻擋中的啟動框架；30 分鐘是明示佔位時限，**不是估計 ETA**。
  `/archive/cymaxwelllee/breakthrough1/` 需在正式提交前準備，Slurm 才能開 log。
  沒有提交任何 job；沒有使用 jasmine；沒有 commit。
- 未改凍結源、發射單或 ucontrast*。新目錄外原有 git 工作保持不動。

本機驗證指令：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 experiments/_workorders/breakthrough-probe/selftest.py
```

預期目前 exit 1；以下全文如實保留必要驗收失敗，不宣稱 selftest 全 PASS。

```text
CPU contract selftest; mock tests are not GPU smoke evidence.
Frozen source expected=276c68fba6c41a05c1e3c2bd5bf4b25c39642d0dbac3ca7aa57cce381056ecfc
Frozen source actual=b7d3a0d98f75fada7e101af73e597af5e08bdc6e1df947852ee764333004a234
test_00_actual_frozen_source_pin_REQUIRED (__main__.ContractTests.test_00_actual_frozen_source_pin_REQUIRED) ... FAIL
test_01_pin_guard_rejects_before_import (__main__.ContractTests.test_01_pin_guard_rejects_before_import) ... ok
test_02_builder_double_run_bytes (__main__.ContractTests.test_02_builder_double_run_bytes) ... ok
test_03_geometry_and_machine_question_sets (__main__.ContractTests.test_03_geometry_and_machine_question_sets) ... ok
test_04_draw_plans_streams (__main__.ContractTests.test_04_draw_plans_streams) ... ok
test_05_reset_three_lines_and_collector_digest (__main__.ContractTests.test_05_reset_three_lines_and_collector_digest) ... ok
test_06_d1_d5_and_251_calls (__main__.ContractTests.test_06_d1_d5_and_251_calls) ... ok
test_07_invalid_draw_denominators_and_c_priority (__main__.ContractTests.test_07_invalid_draw_denominators_and_c_priority) ... ok
test_08_grids_v5_tie_and_d5 (__main__.ContractTests.test_08_grids_v5_tie_and_d5) ... ok
test_09_state_machine_reach_cap_stuck_and_midchunk (__main__.ContractTests.test_09_state_machine_reach_cap_stuck_and_midchunk) ... ok
test_10_strict_reach_and_bypass (__main__.ContractTests.test_10_strict_reach_and_bypass) ... ok
test_11_behavior_delivery_gates_separate (__main__.ContractTests.test_11_behavior_delivery_gates_separate) ... ok
test_12_paired_response_nine_cells (__main__.ContractTests.test_12_paired_response_nine_cells) ... ok
test_13_e2_pair_denominator_and_e3 (__main__.ContractTests.test_13_e2_pair_denominator_and_e3) ... ok
test_14_e4_first_match_v4_and_d5 (__main__.ContractTests.test_14_e4_first_match_v4_and_d5) ... ok
test_15_exit_priority (__main__.ContractTests.test_15_exit_priority) ... ok
test_16_fixed_head_input_mutation_and_manipulation (__main__.ContractTests.test_16_fixed_head_input_mutation_and_manipulation) ... ok
test_17_harvester_negative_matrix (__main__.ContractTests.test_17_harvester_negative_matrix) ... ok
test_18_independent_recompute_all_d_and_switch_mutant (__main__.ContractTests.test_18_independent_recompute_all_d_and_switch_mutant) ... ok
test_19_overwritten_injection_rejected (__main__.ContractTests.test_19_overwritten_injection_rejected) ... ok
test_20_representation_pass_does_not_hide_rollout_failure (__main__.ContractTests.test_20_representation_pass_does_not_hide_rollout_failure) ... ok
test_21_gate_source_and_smoke_coverage (__main__.ContractTests.test_21_gate_source_and_smoke_coverage) ... ok
test_22_deterministic_flags_mock (__main__.ContractTests.test_22_deterministic_flags_mock) ... ok
test_23_production_adapter_REQUIRED (__main__.ContractTests.test_23_production_adapter_REQUIRED) ... FAIL

======================================================================
FAIL: test_00_actual_frozen_source_pin_REQUIRED (__main__.ContractTests.test_00_actual_frozen_source_pin_REQUIRED)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "/home/cymaxwelllee/Projects/lacot/experiments/_workorders/breakthrough-probe/selftest.py", line 34, in test_00_actual_frozen_source_pin_REQUIRED
    self.assertEqual(file_sha(SOURCE), PIN, 'BLOCKED: the required frozen source is not present at the specified path')
AssertionError: 'b7d3a0d98f75fada7e101af73e597af5e08bdc6e1df947852ee764333004a234' != '276c68fba6c41a05c1e3c2bd5bf4b25c39642d0dbac3ca7aa57cce381056ecfc'
- b7d3a0d98f75fada7e101af73e597af5e08bdc6e1df947852ee764333004a234
+ 276c68fba6c41a05c1e3c2bd5bf4b25c39642d0dbac3ca7aa57cce381056ecfc
 : BLOCKED: the required frozen source is not present at the specified path

======================================================================
FAIL: test_23_production_adapter_REQUIRED (__main__.ContractTests.test_23_production_adapter_REQUIRED)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "/home/cymaxwelllee/Projects/lacot/experiments/_workorders/breakthrough-probe/selftest.py", line 368, in test_23_production_adapter_REQUIRED
    self.assertTrue(m3.PRODUCTION_READY and m1.PRODUCTION_READY,
AssertionError: False is not true : BLOCKED: source mismatch stopped approved-module runtime integration; helpers/mocks do not satisfy the full GPU harness delivery

----------------------------------------------------------------------
Ran 24 tests in 0.295s

FAILED (failures=2)
STATUS: BLOCKED — mandatory acceptance checks failed
```

STATUS: BLOCKED — 凍結源 SHA256 不符；正式 runtime／收割／GPU smoke 尚未完成。


## 第二棒：v6.1 前置核對中止（2026-10-01）

本節為第二棒目前狀態；前文與首棒 selftest-output.txt 保留為歷史紀錄。
**未完成接線，不是本棒 DONE 交付。**

### 決策與已核實事項

- 已讀 v6.1 發射單。M9 實體 SHA256 親算為
  `276c68fba6c41a05c1e3c2bd5bf4b25c39642d0dbac3ca7aa57cce381056ecfc`，命中 PIN。
  來源尋址問題已有明確解法；尚未修改 common.SOURCE 或 import runtime。
- donor、實際 steps+1 trace、d4 機械判準均有明文裁定。
- 發現下述規模句仍有契約出入，依本棒工單「出入=停下記錄」中止實作，
  已向使用者提出裁定問題；未擅自裁掉 gate 題或把上限硬壓成 106。
- PRODUCTION_READY 兩旗維持 False。未提交 job、未 commit，未修改主樹源、
  M9 源、發射單或 ucontrast* 既有檔；本棒只新增前置自測紀錄及本節。

### 殘留 BLOCKED：規模句與 gate 執行規則仍不一致

v6.1 原句：

> 主判 20×4＋gate（N 前置 8＋合格題×2≤14）＋確定性符合性重跑 4
> ＝上限 ~106 rollouts

但同卡要求 gate 集「8 題全收」，只有 N 失敗題除名；N 成功者各跑 P/R。
由 gate0b 與 builder 機器重列：task 1 episode 10,15,20,23,25,32,34，
以及 task 3 episode 16，共 8 題。沒有「最多 7 題合格」或預先排除一題的規則。
若 8 題 N 都成功，gate P/R 為 16 次，不滿足原句的 ≤14；完整上限為：

`20×4 + 8 + 8×2 + 4 = 108`。

現有 rollout_plan 的機器核對結果：main=80、gate_N=8、gate_PR=16、
base_plan=104、另加 conformance=4、maximum=108。
問題不只在「~106」的近似字樣，而在明寫的「合格題×2≤14」沒有規則支持。
需裁定保留全部合格題並採 108 上限，或提供最多 7 題的確定選取規則後再續工。

### 本棒前置 selftest 全文（未修改程式，非完成驗收）

指令：`PYTHONDONTWRITEBYTECODE=1 python3 experiments/_workorders/breakthrough-probe/selftest.py`

退出碼 1；24 項中 22 PASS、2 FAIL，未降標、未修改測試換 PASS。
這兩個失敗仍是首棒 SOURCE 指向主樹與 runtime 未接線造成，不能解讀成 M9 SHA 不符。
全文另存 `second-baton-preflight-selftest.txt`；完成接線後仍須擴充測試並全 PASS。

```text
CPU contract selftest; mock tests are not GPU smoke evidence.
Frozen source expected=276c68fba6c41a05c1e3c2bd5bf4b25c39642d0dbac3ca7aa57cce381056ecfc
Frozen source actual=b7d3a0d98f75fada7e101af73e597af5e08bdc6e1df947852ee764333004a234
test_00_actual_frozen_source_pin_REQUIRED (__main__.ContractTests.test_00_actual_frozen_source_pin_REQUIRED) ... FAIL
test_01_pin_guard_rejects_before_import (__main__.ContractTests.test_01_pin_guard_rejects_before_import) ... ok
test_02_builder_double_run_bytes (__main__.ContractTests.test_02_builder_double_run_bytes) ... ok
test_03_geometry_and_machine_question_sets (__main__.ContractTests.test_03_geometry_and_machine_question_sets) ... ok
test_04_draw_plans_streams (__main__.ContractTests.test_04_draw_plans_streams) ... ok
test_05_reset_three_lines_and_collector_digest (__main__.ContractTests.test_05_reset_three_lines_and_collector_digest) ... ok
test_06_d1_d5_and_251_calls (__main__.ContractTests.test_06_d1_d5_and_251_calls) ... ok
test_07_invalid_draw_denominators_and_c_priority (__main__.ContractTests.test_07_invalid_draw_denominators_and_c_priority) ... ok
test_08_grids_v5_tie_and_d5 (__main__.ContractTests.test_08_grids_v5_tie_and_d5) ... ok
test_09_state_machine_reach_cap_stuck_and_midchunk (__main__.ContractTests.test_09_state_machine_reach_cap_stuck_and_midchunk) ... ok
test_10_strict_reach_and_bypass (__main__.ContractTests.test_10_strict_reach_and_bypass) ... ok
test_11_behavior_delivery_gates_separate (__main__.ContractTests.test_11_behavior_delivery_gates_separate) ... ok
test_12_paired_response_nine_cells (__main__.ContractTests.test_12_paired_response_nine_cells) ... ok
test_13_e2_pair_denominator_and_e3 (__main__.ContractTests.test_13_e2_pair_denominator_and_e3) ... ok
test_14_e4_first_match_v4_and_d5 (__main__.ContractTests.test_14_e4_first_match_v4_and_d5) ... ok
test_15_exit_priority (__main__.ContractTests.test_15_exit_priority) ... ok
test_16_fixed_head_input_mutation_and_manipulation (__main__.ContractTests.test_16_fixed_head_input_mutation_and_manipulation) ... ok
test_17_harvester_negative_matrix (__main__.ContractTests.test_17_harvester_negative_matrix) ... ok
test_18_independent_recompute_all_d_and_switch_mutant (__main__.ContractTests.test_18_independent_recompute_all_d_and_switch_mutant) ... ok
test_19_overwritten_injection_rejected (__main__.ContractTests.test_19_overwritten_injection_rejected) ... ok
test_20_representation_pass_does_not_hide_rollout_failure (__main__.ContractTests.test_20_representation_pass_does_not_hide_rollout_failure) ... ok
test_21_gate_source_and_smoke_coverage (__main__.ContractTests.test_21_gate_source_and_smoke_coverage) ... ok
test_22_deterministic_flags_mock (__main__.ContractTests.test_22_deterministic_flags_mock) ... ok
test_23_production_adapter_REQUIRED (__main__.ContractTests.test_23_production_adapter_REQUIRED) ... FAIL

======================================================================
FAIL: test_00_actual_frozen_source_pin_REQUIRED (__main__.ContractTests.test_00_actual_frozen_source_pin_REQUIRED)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "/home/cymaxwelllee/Projects/lacot/experiments/_workorders/breakthrough-probe/selftest.py", line 34, in test_00_actual_frozen_source_pin_REQUIRED
    self.assertEqual(file_sha(SOURCE), PIN, 'BLOCKED: the required frozen source is not present at the specified path')
AssertionError: 'b7d3a0d98f75fada7e101af73e597af5e08bdc6e1df947852ee764333004a234' != '276c68fba6c41a05c1e3c2bd5bf4b25c39642d0dbac3ca7aa57cce381056ecfc'
- b7d3a0d98f75fada7e101af73e597af5e08bdc6e1df947852ee764333004a234
+ 276c68fba6c41a05c1e3c2bd5bf4b25c39642d0dbac3ca7aa57cce381056ecfc
 : BLOCKED: the required frozen source is not present at the specified path

======================================================================
FAIL: test_23_production_adapter_REQUIRED (__main__.ContractTests.test_23_production_adapter_REQUIRED)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "/home/cymaxwelllee/Projects/lacot/experiments/_workorders/breakthrough-probe/selftest.py", line 368, in test_23_production_adapter_REQUIRED
    self.assertTrue(m3.PRODUCTION_READY and m1.PRODUCTION_READY,
AssertionError: False is not true : BLOCKED: source mismatch stopped approved-module runtime integration; helpers/mocks do not satisfy the full GPU harness delivery

----------------------------------------------------------------------
Ran 24 tests in 0.311s

FAILED (failures=2)
STATUS: BLOCKED — mandatory acceptance checks failed
```

STATUS: BLOCKED — v6.1 gate「8 題全收、N 成功各跑 P/R」與規模句「合格題×2≤14」衝突；依出入即停紅線，待 108 上限或明確選題規則裁定後續工。


## 第三棒：v6.2 接線完成、selftest 全 PASS（2026-10-01）

本節取代前兩棒的實作阻塞狀態；契約僅採已裁定 v6.2，Move 1 上限為
**80 主判＋8 gate N 前置＋16 gate P/R＋4 符合性重跑＝108**。
本棒完成 CPU 實作驗收，**不等於 GPU smoke／正式發射放行**。

### 1–4 接線與規模

1. **M9 凍結源**：`common.SOURCE` 指向唯讀 M9 實體，PIN 不變。
   `runtime.load_frozen()` 在任何模型 import 前親驗 source SHA 與 s33 checkpoint SHA；
   用 `importlib` 執行原檔，在第 2480 行舊環境評估入口之前以 trace exception 停止。
   AST 只定位停止行，未改 AST／source bytes、未做 sed derived、未改 flat-only guard。
   載入沿用原 source 的 raw encoder/decoder 與 consumer EMA；逐 tensor 核實 EMA，
   normalization 由指定資料集依原 source 重算，資料 SHA、mu/sd、設定均進 runtime receipt。
   `STEPS1=STEPS2=CONT_TRAIN=0`，停在舊評估與寫檔流程前。
2. **encode_u → head**：128 點弧長重採樣、XY normalization、原 `encode_u` 的全 False mask；
   真 s33 CPU 測得 u `[1,8,256]`、EMA head `[1,4,2]`，輸出 finite。
   head 呼叫邊界逐 chunk hash；P/Q/R 不抽 flow。共享 `run_draw()` 已接環境 reset、
   cache 清除、RNG 快照、N／Move 3 Collector A fresh-flow、Move 3 共同動作噪音、
   換段即時重抽與真實 steps+1 trace。
   兩幾何 P/Q 各做一次實際程式路徑的同 u 重跑：使用真模型＋明示 synthetic 短環境，
   XY、hash、RNG 記錄全部重現；另以真 ogbench 環境＋真模型驗合法 w 的 1-step cap。
   這些是 CPU 接線驗證，不充作真 GPU 符合性或正式 draw。
3. **donor 表**：只含 task 4←5 ep5 A、5←4 ep4 A、1←3 ep16 A、3←1 ep10 A；
   刪除 task 2 key 與舊 blocker。Move 1 runtime 按表取 donor，gate P 僅取該題 N 成功
   trace；獨立收割器拒收錯 donor 或錯 self-rollout trace hash。
4. **trace／d4 validator**：驗 `len(xy)=observed_steps+1`、`0<steps≤1000`、
   同長逐步 success、終止原因與 trace SHA；按實際 steps 驗 flow ledger／head chunks。
   拒絕長度不一致、padding、拼接、缺終止原因。終止當步不再要求抽下一段。
   d4 從全程 XY 落格重算「本臂 exclusive ≥1、對臂 exclusive=0」，A/B 對稱；
   走過本臂又折返對臂為 d5，不信任 producer 的分類欄。task 2 沒有專屬互斥段，
   不製造 d4 穿越證據，仍列單路 control 的 w/g 與 A/C 成績。

`rollout_plan()` 現在直接含四列 `kind=determinism-verification`、
`additional_draw=false`，主 rollout 104 列與重跑 4 列分開驗收，總上限 108；
N 失敗／表示層未過依條件減列，收割器獨立重導預期集合，禁止靠 skipped 掩蓋漏列。
`builder.json`、`move1-plan.json`、`smoke-plan.json` 及 sidecar SHA 已重生，
selftest 另驗磁碟快照與生成器 byte-identical。

`runtime.execute()` 接上計畫、載入、環境 runner、獨立驗證與 exclusive JSON writer；
收割 CLI 可用 `--input result.json --out readout.json` 重新驗 receipt／原 n64 reset
指紋與條件題單，重算 d 類／Move 1 首入、配對九格、行為 gate、表示層及同 u 重跑。
smoke CLI 與兩支 harness 共用這條路，所有 sbatch 帶顯式 `PROBE_CALIBRATION`。
fixture 入口仍明示 synthetic，不能混入正式產物。

### 原 2 FAIL 的修復與驗證

- `test_00`：修正尋址後實測 M9 SHA 命中 `276c68fb…056ecfc`，未改 PIN。
- `test_23`：原本要求兩個 `PRODUCTION_READY=True` 的断言不符合本棒要求；改為
  **必跑真 M9／s33 EMA／encode／head／runtime 的 CPU 子進程驗證**，並另外硬驗兩旗
  都是 False、正式入口拒絕啟動。缺 Python／torch／資料／checkpoint 或任一實際接口
  不符都會 FAIL，無 skip、無以 mock 取代必要接口驗收。
- 新判例含 108 規模與減列、M9 SHA 先於 import、短 trace 正反例、雙臂 d4/d5 正反例、
  donor 無 task 2／錯 donor 拒收、完整 108 列 fixture 的獨立收割、錯同 u 重跑拒收。
- 最終 **30 PASS、0 FAIL、0 ERROR，exit 0**；三支 sbatch `bash -n` 通過。
  `git diff --name-only` 為空；本棒修改均在本工單目錄。

### 放行與可重跑條件

- 兩個 `PRODUCTION_READY` 保持 **False**。未使用 GPU、未提交 Slurm job、未 commit。
  主樹 rollout、M9、collector、ucontrast* 既有檔未改；親驗前後 source／collector SHA 一致。
- GPU smoke、測速／ETA、vFinal 及 lead 翻旗仍待後續工序；本棒沒有聲稱它們已完成。
- 正式／smoke runner 仍先驗 receipt 指向的 n64 `/archive` 原產物 SHA 與 reset 指紋；
  這兩檔目前本機不可得，上機需可讀，不能用新 rollout 自報 fingerprint 代替。
- 校準 JSON 必填 `dataset_dir`、`steps_per_cell`、`stuck_window`、`stuck_distance`、
  `noise_band`（鍵 `4`／`5`）。cap 依 `ceil(w_depth*steps_per_cell*2)`（不超 H），
  stuck 檢查指定時間窗的最大位移；數值須在 GPU smoke 校準／凍結，不提供臆測預設。
- 本機 selftest 用系統 numpy 加 `.venv/bin/python` 的真模型子進程；可用
  `PROBE_TEST_PYTHON`、`PROBE_TEST_DATASET_DIR` 指定等價環境。正式 source／ckpt PIN 不可覆寫。
  source 原有零訓練迴圈診斷會印 `match-acc(train batch) nan`；本輪不做訓練，
  encoder／head／decode 真正輸出另有 finite 斷言且均通過。
- `input-audit.json` 為首棒歷史快照；本次輸入、程式、生成物 SHA 在
  `third-baton-input-audit.json`，不是 GPU receipt。
  首棒 selftest 原文另保留 `first-baton-selftest.txt`；`selftest-output.txt` 現為最新結果，
  與 `third-baton-selftest.txt` 完全相同。

### 第三棒 selftest 全文

指令：`PYTHONDONTWRITEBYTECODE=1 python3 experiments/_workorders/breakthrough-probe/selftest.py`

```text
CPU contract selftest; mock tests are not GPU smoke evidence.
Frozen source expected=276c68fba6c41a05c1e3c2bd5bf4b25c39642d0dbac3ca7aa57cce381056ecfc
Frozen source actual=276c68fba6c41a05c1e3c2bd5bf4b25c39642d0dbac3ca7aa57cce381056ecfc
test_00_actual_frozen_source_pin_REQUIRED (__main__.ContractTests.test_00_actual_frozen_source_pin_REQUIRED) ... ok
test_01_pin_guard_rejects_before_import (__main__.ContractTests.test_01_pin_guard_rejects_before_import) ... ok
test_02_builder_double_run_bytes (__main__.ContractTests.test_02_builder_double_run_bytes) ... ok
test_03_geometry_and_machine_question_sets (__main__.ContractTests.test_03_geometry_and_machine_question_sets) ... ok
test_04_draw_plans_streams (__main__.ContractTests.test_04_draw_plans_streams) ... ok
test_05_reset_three_lines_and_collector_digest (__main__.ContractTests.test_05_reset_three_lines_and_collector_digest) ... ok
test_06_d1_d5_and_251_calls (__main__.ContractTests.test_06_d1_d5_and_251_calls) ... ok
test_07_invalid_draw_denominators_and_c_priority (__main__.ContractTests.test_07_invalid_draw_denominators_and_c_priority) ... ok
test_08_grids_v5_tie_and_d5 (__main__.ContractTests.test_08_grids_v5_tie_and_d5) ... ok
test_09_state_machine_reach_cap_stuck_and_midchunk (__main__.ContractTests.test_09_state_machine_reach_cap_stuck_and_midchunk) ... ok
test_10_strict_reach_and_bypass (__main__.ContractTests.test_10_strict_reach_and_bypass) ... ok
test_11_behavior_delivery_gates_separate (__main__.ContractTests.test_11_behavior_delivery_gates_separate) ... ok
test_12_paired_response_nine_cells (__main__.ContractTests.test_12_paired_response_nine_cells) ... ok
test_13_e2_pair_denominator_and_e3 (__main__.ContractTests.test_13_e2_pair_denominator_and_e3) ... ok
test_14_e4_first_match_v4_and_d5 (__main__.ContractTests.test_14_e4_first_match_v4_and_d5) ... ok
test_15_exit_priority (__main__.ContractTests.test_15_exit_priority) ... ok
test_16_fixed_head_input_mutation_and_manipulation (__main__.ContractTests.test_16_fixed_head_input_mutation_and_manipulation) ... ok
test_17_harvester_negative_matrix (__main__.ContractTests.test_17_harvester_negative_matrix) ... ok
test_18_independent_recompute_all_d_and_switch_mutant (__main__.ContractTests.test_18_independent_recompute_all_d_and_switch_mutant) ... ok
test_19_overwritten_injection_rejected (__main__.ContractTests.test_19_overwritten_injection_rejected) ... ok
test_20_representation_pass_does_not_hide_rollout_failure (__main__.ContractTests.test_20_representation_pass_does_not_hide_rollout_failure) ... ok
test_21_gate_source_and_smoke_coverage (__main__.ContractTests.test_21_gate_source_and_smoke_coverage) ... ok
test_22_deterministic_flags_mock (__main__.ContractTests.test_22_deterministic_flags_mock) ... ok
test_23_production_adapter_REQUIRED (__main__.ContractTests.test_23_production_adapter_REQUIRED) ... 
device: cpu
  teacher 題庫：4096 條（細格步數 p50 26 p90 42 max 62），mix=0.5
設定：seed=33 cons=self ema_m=0.996 K=8 COND=256
stage 1 e_target 目標=recon_ictr ...  w_var=0.0 w_cov=0.0 w_ictr=0.2 sigma=0.05  warmup=500
  e_target match-acc(train batch) nan
stage 2 flow+refine+action ...
  ⭐ 已切到 EMA 影子權重（m=0.999，五模組）
✅ 載入 ckpt_large-stitch_self_K8_c256_ch4_st8000_T128_ep2_gu_eorecon_ictr_tch0.5_emw0.999_wu500_dssoft_norf_cd0.1_bci_s33.pt（跳過訓練，只跑評估）  cfg={'K': 8, 'T_CAP': 128, 'ENC_OBJ': 'recon_ictr', 'LEARNED_REFINE': 0, 'COND_DROP': 0.1}
  decoder 內部點 RMSE 0.0981   打亂 u 之後 1.8960   用到的 u 值 +1.7979   ✓
REAL_CPU_PASS: M9 import, s33/EMA, encode_u false mask, head, decode, four P/Q same-u reruns, fresh-flow seed, legal-w short cap; GPU evidence=false
{"ckpt_sha256": "88180676e1f8d8df22c47bf4eeaf23eba6a73c5e92cacbf522ce3044095af787", "encoded_u_sha256": "fd89a90c7385dfd5c57288d7a2511b188fcb03c867cdf5e23bc0ab72ebb961a9", "encoder_shape": [1, 8, 256], "head_shape": [1, 4, 2], "head_u_hashes": ["fd89a90c7385dfd5c57288d7a2511b188fcb03c867cdf5e23bc0ab72ebb961a9"], "mask_all_false": true, "production_evidence": false, "source_sha256": "276c68fba6c41a05c1e3c2bd5bf4b25c39642d0dbac3ca7aa57cce381056ecfc"}
ok
test_24_move1_108_limit_and_conditional_rows (__main__.ContractTests.test_24_move1_108_limit_and_conditional_rows) ... ok
test_25_short_trace_contract (__main__.ContractTests.test_25_short_trace_contract) ... ok
test_26_d4_exclusive_crossing_symmetric (__main__.ContractTests.test_26_d4_exclusive_crossing_symmetric) ... ok
test_27_m9_pin_and_import_boundary (__main__.ContractTests.test_27_m9_pin_and_import_boundary) ... ok
test_28_full_artifact_conditional_coverage_and_donors (__main__.ContractTests.test_28_full_artifact_conditional_coverage_and_donors) ... ok
test_29_generated_contract_snapshots (__main__.ContractTests.test_29_generated_contract_snapshots) ... ok

----------------------------------------------------------------------
Ran 30 tests in 4.718s

OK
STATUS: DONE
```

STATUS: DONE


## 第四棒：v6.3 S2 雙軌合併修項（2026-10-01）

本節取代前三棒的現行狀態；前三棒與 S2 判決原文保留為稽核歷史。
第四棒完成後 harvest 已具正式合讀接線，輸出可稱為契約判讀；GPU smoke 與正式
發射仍是後續工序。本棒未提交 job、未用 GPU、未 commit。

### A1／A2：凍結契約、固定程式清單、驗敗保留 raw

- `CONTRACT-FROZEN.md` 是讀入 v6.3 活卡的逐 byte 複本，SHA256：
  `7525254c83a62031ad761f6993112b14df59595fadfdfd743a45e9313bc91cf4`。
  `common.CARD_PIN` 及同名 `.sha256` 記錄此值；receipt 與收割均驗此複本。
  發射單日後補 ETA 不會改變本次 receipt。原發射單及 M9 源未改。
- `CODE_FILES` 明列十個 runtime／收割／校準依賴檔，無 `HERE/*.py` glob。
  額外加入測試腳本、更新說明與翻旗不改 code hash 集合；受檢程式本身改動仍會拒收。
- 每完成一個 rollout，先用獨占建檔保存 `shards/NNNN.json`，再驗該列。
  RNG before/after 獨立存成 `NNNN.rng.json.gz`，原始 SHA 綁在 shard 內；
  `raw.json`／`result.json` 引用 shard，不重複展開 RNG。
  最終先獨占寫 `raw.json`，再做完整驗收；任何驗敗另寫 `validation_error.json`，
  保留完成列與失敗列，不產生假成功 result。接口等較早例外同樣留下 raw／error。
- 每 50 rollouts 輸出一次 `PROGRESS`；Move 3 sbatch 已改 `02:00:00`。
  新增 `--resume-from`：驗舊 checkpoint receipt、校準、runtime 設定、逐 shard SHA、
  RNG sidecar 與本次編碼內容後，在**新目錄**續跑缺列。完整列不重跑，來源不覆寫；
  損壞／缺證據的 shard 拒收，不把未知資料当成功。續跑不是自動翻旗或自動重提 job。

### B1：正式合讀與 exact-zero 硬資格

- `harvest --move1 ... --move3 ...` 對兩份產物分別獨立驗收，再按 task 4／5 合讀。
  校準、runtime provenance 與 smoke／synthetic scope 必須一致。
  正式路徑实际呼叫 `rules.e2`、`ceiling`、`adsorption`、`content_sensitivity`、
  `exact_one_sided` 及 `exits -> e4`。E0 全局停，E1–E4 分層；輸出各層
  E0／E1／E2 分子分母與下界旗／E3 天花板／E4 行選擇、行為 gate、P/Q 九格表、
  n_diff、B-vs-C 兩向 discordant、exact p、軸一停損旗。
- B/C discordance 以題為單位，B 的 d4 與 C 成功獨立彙整；兩臂有效 draws 不足半數
  的題列 excluded。task 2 保留獨立對照，不混入兩個幾何主判。
- exact-zero 必填 named keys：`torch_deterministic`、`environment_policy_reset`、
  `rng_archived`、`dtype_batch_backend_frozen`、`head_deterministic`、
  `paired_protocol`、`same_u_rerun`。逐鍵要求 boolean True；並驗實際 deterministic／
  backend 設定與同層 P、Q 各一次實際同 u 重跑。任意非空字典不算資格。
  conformance False 會自動輸出「行為差異觀測（不可歸因）」，保留 n_diff 及描述性九格表。
- smoke 題數不足完整幾何層時明列 INCOMPLETE，不以少數題虛構 14／6 分母。
  行為 gate 不足 5 題的正式出口為探針改期；表示層未過的層仍明列 E1。

### B5／B6／C：不變量與新裁定

- `encode_trajectory()` 在真正餵入 encoder 的原始 XY 折線上計算
  `injection_traj_sha256`，隨實際 u 傳遞；不是事後依臂標籤另算一份自證。
  收割以 builder route 或該題 N trace 重建比對，涵蓋 gate P、gate R、主判 R、P/Q；
  同 u 重跑也驗同一來源。保留原 P/Q encoded-u、逐 chunk head hash 檢查。
- N 臂 fresh-flow ledger 必填、非空、精確等於實際步數的逐 4-step g call；
  chunk_steps 僅准 integer 4。success aggregate、終止原因與末步 success 雙向一致，
  不准成功後繼續觀測。receipt.interface 必有 false mask 與 `[1,8,256]`／`[1,4,2]`，
  並有有效 head delivery hash。rng_before 的 torch CPU／noise bytes 用 seed 重算驗。
- task 2 單路 A 核實改為入過唯一 A 路內段即可；無另臂段，不再把所有成功誤標 d5。
- d1 不入 d2/d3/d5 主導分流；在既定觀測有效性與 d4 優先序後，兩臂 d1 合計 ≥8
  落 C 格並加 `bypass_version` 註記。w-miss=d2／有效 draws；復現明釘同段 **and** 成功。
- Move 1 每臂與 Move 3 共用校準 stuck_check；以 termination_reason 獨立記錄卡死。
  未校準資料／未掛檢查資料的 E2 明列 lower_bound。沒有把撞牆自行猜成新幾何判準。
- `flags.json` 是唯一放行旗檔：目前 `PRODUCTION_READY=false`、`SMOKE_ENABLED=true`。
  CLI 與 runtime API 都驗旗；selftest 在隔離旗檔測四種組合，不再硬釘 `.py` 常數，
  正式翻旗可沿用相同受審 .py SHA。

### B7：可重跑校準工具

`calibrate.py` 吃兩份有 SHA 的 smoke 產物，先經相同完整收割驗證再估計：
同路三次重新編碼的最大 pairwise L2（task 1／3 gate noise 作保守下界，再合 task 4／5）；
命中 w 的 steps/depth 第 90 百分位估步／格（runtime 仍乘固定 cap 係數 2）；
有實際進展 trace 的正位移第 10 百分位與低位移最長停留窗估 stuck_distance／window。
確定性重編碼噪聲帶可為零，沒有任意加一個正數；不同路線距離不拿來充同路噪聲。
估計器方法、樣本數、輸入 SHA 與 provisional 註記進輸出，之後隨 calibration 進 receipt。
無命中 w／無進展／缺同路重抽證據即 fail-fast，不編造校準值。

`calibration.example.json` 是含 null 的 provisional 範本；必須填實測／明示的 bootstrap 值。
必填六鍵缺漏、型別錯、負值等在載模型前拒絕。資料集本身新增 `DATASET_PIN`，
固定 `9add335e598e48ebc483447d61415a9755ffb738ddfc406cb9708b2a238992e8`。

使用方式（以下只列重跑指令，未執行 GPU 工作）：

```bash
# 另建校準範本；填值後供 smoke 使用，null 不能直接執行。
python -B calibrate.py --template --out /tmp/probe-calibration-template.json
# 由完成的兩份 smoke 產物估校準（在具有原 n64、ogbench 的環境）。
python -B calibrate.py --move1 /path/smoke/move1/result.json --move3 /path/smoke/move3/result.json --out /path/calibration.json
# 合讀：不相信 artifact 內嵌的 producer readout。
python -B harvest.py --move1 /path/move1/result.json --move3 /path/move3/result.json --out /path/combined.json
# 中斷後重用已驗完成列；原目錄不覆寫，旗與 --smoke 選項仍適用。
python -B harness_move3.py --resume-from /path/interrupted --outdir /archive/cymaxwelllee/breakthrough1/resumed --calibration /path/calibration.json
```

### B2：驗收牙齒與結果

- selftest 新增 `fourth_selftest.py`，原 30 項保留並更新 v6.3 已裁定預期，總共 44 項。
  包含完整 108 列 runtime 協調、4／5 題 gate 邊界、真實 raw／壓縮 sidecar 往返、
  驗敗保留、續跑 107 列／完整重用零重跑、每 50 rollout 心跳，以及 E0–E4 合讀。
- 收入 S2 p15 的獨立腳本軌跡 oracle 思路，180 組固定種子 fuzz 跑真實 run_draw／
  TwoStage／recompute 路徑，逐步對噪音、cap、stuck、命中時刻、立即重抽、w/g cond。
  這是 stub model／scripted environment 測試，不冒稱 GPU 或真迷宮科學證據。
- 收入 S2 p16 凍結 oracle A 協定等價測試：真 s33／真迷宮 CPU，三組各 160 步，
  動作與 XY 均 byte-identical；僅測試端補 `_bon_plan` 的 BON_N<=1 sample_plan 存根，
  原凍結源不改。原真模型 encode／mask／head／P/Q 重跑與 cap 測試照跑。
- Move 1 n_diff 正向例、conformance 負向例、復現需成功、head chunks、historical A/B
  指紋、ceil 門檻、E3 4/6 vs 5/6、主導差 1 vs 2、停損 2 vs 3 均有殺手測試。
- `s2-surviving-mutations.json` 保存原主錘存活變異（35 個＋S3 無改動對照）；
  `mutation_audit.py` 每個變異在 `/tmp` 獨立 mirror 執行完整 selftest，無 live code mutation。
  最終 35/35 實質存活變異全部殺掉，S3 無改動對照維持存活；
  指定 runtime／Move 1 判決目標 19/19 全滅，無 STALE_PATTERN、無 CRASH。
  完整結果：`fourth-baton-mutations.json`／`.txt`。
- `fourth-baton-selftest.txt` 與 `selftest-output.txt` 保存最新全文；
  `fourth-baton-input-audit.json` 保存輸入、固定 runtime 清單、測試、旗檔與輸出 SHA。
  最終 **44 PASS／0 FAIL／0 ERROR，exit 0**；三支 sbatch `bash -n` 通過。
  M9、主樹源、collector、checkpoint、資料集、gate0b SHA 與原稽核一致，
  活 v6.3 卡與凍結複本逐 byte 相同；`git diff --name-only` 空白。

本棒交付為實作／CPU 驗收完成；校準值的 GPU 實測、GPU 同 u 符合性與正式發射放行
仍由後續 smoke／lead 工序處理，不以 CPU PASS 代替。

STATUS: DONE
