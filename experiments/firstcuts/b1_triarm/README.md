**在證明什麼、判準是什麼**：把 21.5% 零漂移失敗地板拆到三個病因（資料重放極限／
decoder 回授／字典量化）——判準（預註冊）：N 臂相對 L 臂救回 ≥12 題且弄壞 ≤2 題
〔churn 校準：null 介入能矇中 23/43，此門檻按其上緣定〕⇒ 存在「decoder 條件可移除」
的失敗；α 劑量下十倍小殘差仍同等失敗〔配對 bootstrap 95% CI〔拍〕〕⇒ 才支持飽和
放大器說；真降精度對照一步殘差 ≥ 資料一步殘差的 50%〔拍〕⇒ 精度病才立案。

B1 實作：`run_b1.py`、`run_b1_precision.py`、`test_b1_wiring.py`。

本機 `zeldajr` 的官方 val 位於本機 ext4：`/home/cymaxwelllee/.ogbench/data/`。
runner 以 `findmnt FSTYPE` 拒絕 NFS 等網路檔案系統，並檢查資料檔存在。
未跑 200 題、未修改 MuJoCo／venv、未改 firstcuts 外既有檔案、未 git commit。

執行方式（repo 根目錄，用已存在 `.venv`）：

```bash
# 不讀 dataset；不能替代官方 val 全通路 smoke
.venv/bin/python -B experiments/firstcuts/b1_triarm/test_b1_wiring.py --snapshot-only

# 明確人工fixture測試，不讀官方資料、不等於官方wiring驗收
.venv/bin/python -B experiments/firstcuts/b1_triarm/test_b1_wiring.py \
  --synthetic --n-tasks 1 --out-dir experiments/firstcuts/b1_triarm/synthetic_smoke_new

# zeldajr 本機官方 val：4 題、24 ulp/題、完整 200 步
export OGBENCH_DATA_DIR=/home/cymaxwelllee/.ogbench/data
.venv/bin/python -B experiments/firstcuts/b1_triarm/test_b1_wiring.py \
  --n-tasks 4 --out-dir experiments/firstcuts/b1_triarm/results_wiring

# 分開跑三臂與精度（兩者可獨立啟動；不要覆蓋已存在的 summary）
.venv/bin/python -B experiments/firstcuts/b1_triarm/run_b1.py \
  --data-dir "$OGBENCH_DATA_DIR" \
  --n-tasks 2 --out-dir experiments/firstcuts/b1_triarm/triarm_results
.venv/bin/python -B experiments/firstcuts/b1_triarm/run_b1_precision.py \
  --data-dir "$OGBENCH_DATA_DIR" \
  --n-tasks 2 --out-dir experiments/firstcuts/b1_triarm/precision_results
```

精度程式也可加 `--source-dir experiments/firstcuts/b1_triarm/triarm_results`，
直接讀已存原生 R 中途快照；會驗證題數、seed、資料／checkpoint／ruler hash 與 MuJoCo 版本。
完整主跑入口保留 `--n-tasks 200 --main`，本次禁止執行。全量先只跑全部 L；
leg-1 失敗不等於 43 就寫 failed summary 並停止，T/N/劑量/ulp 不會啟動。
smoke 的 43/200 檢驗永遠標 `not_evaluated_smoke_only`，不冒充全量基線複現。

接口與語義：

- 任務直接呼叫既有 `run_teacher_relay_byleg.build_tasks`，seed=20260908、原 val 的
  順序前綴，chunk=4、K=32、200 步、ρ=1.875。到達只看 leg-1 每步後 xy；
  `first_hit` 為一基底步數，未到達=-1。所有臂到達後仍記滿 200 步以對齊連續曲線。
  舊 relay 的「全 leg 完成後於 chunk 末停止」不影響 leg-1；另抽第一題直接呼叫
  舊 `run_one`，比較 reached、共同步數內所有 action/xy，要求逐位一致。
- T 重放裁切後的官方教師动作，從完整同快照產生本機 R。
  N 固定原字串，每 chunk 使用 R 在 `4k` 時刻的 obs 解碼；可預先計算整條帶，
  因為 obs 與 code 都固定。N 物理只吃 N 解碼的動作，從不重置到 R、不重放 T 動作。
  L 每 chunk 用自身 obs，原 schedule 的逐 chunk encoder/decoder batch shape 不變。
- `obs[t]` 是第 t 步動作前的自身觀測；`action[t]` 將狀態 t 推至 t+1。
  每臂 qpos/qvel/obs/xy/yaw/shape 有 201 行，action 有 200 行。
  N 額外存 50 行 `decoder_obs`、`decoder_ref_step` 與來源標記。
  負控取 R.obs[4k+3]（0-based），末次 199 合法；越界直接報錯，不 clamp。
  action 及 qpos 的 offset 差異都必須 >1e-6 才通過負控。
- 劑量 `a(alpha)=(1-alpha)*a_N+alpha*a_T`，只作用於已裁切动作帶，
  alpha 固定 0、0.9、0.99、1（相對 T 殘差比 1、0.1、0.01、0）；
  0/1 端點各自與 N/T 的 qpos/qvel/obs/action 逐位一致自檢。
- ULP 每題固定 24 次，以 seed/task/replicate 產生 ±一格 float32 spacing，
  加到原生 float64 qpos/qvel；free/ball joint 四元數正規化。
  每條 ULP 軌跡存初始 delta 與全流，可從起點快照及 delta 還原。
  每題輸出成功比例；「穩定失敗」只定義為本次 24 次均失敗，並非宣稱真成功機率=0。
- xy 是世界座標；yaw 以正規化 torso quaternion 求得，yaw 誤差 wrap 至 [-π,π]。
  去群 shape 為所有非 world 身體質心：扣 torso xyz，再移除 torso yaw；
  shape 誤差為相對 R 的逐身體歐式距離 RMS（公尺）。使用 scratch MjData 算當前姿態，
  不污染主模擬器 warmstart。三種連續誤差逐步存檔，summary 有 mean/final/max。

快照沿用可讀的 `~/Projects/lacot-vqoracle/experiments/lacot_vqo.py:86`
之 save_state/restore_state 慣例：set_state（含 mj_forward）後還原 act/time/warmstart。
沒有 runtime import 外部 repo。補存 MuJoCo INTEGRATION（ctrl/外力/mocap 等）、
Python／NumPy／Torch CPU／env／action_space／observation_space RNG、wrapper 計步，
環境 goal/task 與 leg bookkeeping。JSON metadata + 純數值 NPZ，不用 pickle。
qpos/qvel/act/warmstart/time 是權威欄位，restore 先套 integration 再覆寫這些欄位，
因此 (ii)/(iii) 可只更動各自指定欄位；integration 中冗餘值不會覆蓋介入。

精度對照：

- R 第 100 步後的 native float64 完整快照為共同起點，固定同一教師动作後半段 100 步。
- (i) 完整還原，要求每一步 qpos/qvel/obs/action 完全相等；零容差。
- (ii) 只將 qpos/qvel float64→float32→float64；不額外正規化四元數，
  因工單此項指定只做 cast，與 ULP 合法擾動是不同操作。
- (iii) 原生 qpos/qvel，單獨把 warmstart 清零；不疊加 (ii)。
- e_data 從同個官方資料索引的 qpos/qvel 重放同动作，對比資料後續狀態；
  資料沒有 warmstart，因此明定清零，不能繼承 R 的 solver 狀態。
  e_data 不能被解釋成單純精度誤差，還可能含未保存狀態／版本／資料生成差。
- 各控制與 e_data 都輸出 xy-L2、qpos-L∞、qvel-L∞ 的第一步、全段 RMS、末步、最大值，
  全段逐步誤差與兩側完整狀態也落盤。精度 50% 門檻事先操作化為各題第一步 xy-L2
  平均之比；e_data=0 時標 null，不能硬除。smoke 只驗接口，不下精度病的科學結論。
- 此機 `.venv` 實测 MuJoCo **3.11.0**，工單描述的 3.12.0 不是這台環境；
  程式保存實際版本，沒有安裝、升降版或改 lock。

產物：

```text
OUT/preregistered.json          # 跑前門檻、環境、資料/checkpoint/ruler hash
OUT/summary.json                # 完成錨、救回2×2、劑量曲線/CI、ULP分布、各題記錄
OUT/task_NNN_epE/task.json
OUT/task_NNN_epE/snapshot.npz   # leg-1起點完整快照
OUT/task_NNN_epE/R_mid_snapshot.npz
OUT/task_NNN_epE/{T,N,L,T_replay,N_offset3,dose_0,dose_0.9,dose_0.99,dose_1}.npz
OUT/task_NNN_epE/ulp_00.npz ... ulp_23.npz
PRECISION/task_NNN_epE/{native_mid_snapshot,i_snapshot,ii_snapshot,iii_snapshot,data_snapshot}.npz
PRECISION/task_NNN_epE/{reference,i,ii,iii,data_replay}.npz
PRECISION/summary.json          # (i)(ii)(iii)與e_data殘差表
```

`n_arms=3` 指 T/N/L；另列 `n_dose_arms=4`，24 組 ULP、2 組自檢（T_replay/N_offset3）。
`n_steps_total` 計實際 env.step 次數，含重複端點與舊 relay 比對，排除 env.reset 內部穩定步；
物理子步數為此值×frame_skip=5。`n_rollouts` 同樣含所有控制。
精度 `n_arms=3` 對應 (i)(ii)(iii)，另列 dataset comparator=1；
從既存 R 讀取時每題400步，獨立生成 R 時每題600步。

判讀表（遵照工單）：

| 觀察 | 可支持的判讀 |
| --- | --- |
| N 相對 L 救回≥12且弄壞≤2 | 存在 decoder 條件可移除的失敗；標 churn-calibrated |
| N 比 L 差 | decoder 回授正在救場（有正貢獻） |
| 固定參考下非零殘差縮十倍仍同等失敗 | α=0.9 相對 α=0 的配對失敗率差 95% CI 全在 ±0.05 內，才支持飽和放大器 |
| ulp 24次穩定失敗率高 | 地板可能是分布性質，非單次壞運 |
| 真降精度一步殘差≥e_data的50% | 精度病達立案門檻，需全量結果判讀 |

不得把三臂失敗率直接相加拆病因份額。
所有〔拍〕值與 churn 門檻固定於 `PREREGISTERED`，跑前落盤，summary 原樣附回；
工單舊值與實作新值並列，未依結果改門檻。
配對 bootstrap 固定 seed、10000次、95% percentile CI，比較各劑量相對 N 的失敗差。
「同等失敗」的預註冊等效界限為 α=0.9 相對 α=0 的失敗率差 |Δ|≤0.05〔拍〕；
配對 95% CI 全落在此界限內才稱同等失敗。CI 包含 0 但未落在界限內只表示未檢出差異。

200題 L=43 不在本次獲准執行範圍；只實作 gate，不把 smoke 當其證據。
舊 α 網格 {0,0.1,1} 無十倍小殘差點；本版改為 {0,0.9,0.99,1}，
`old_new` 保留兩版與改動理由。

完成錨將 3 主臂、4 劑量、24 ULP 與 2 自檢拆開；官方 4 題 wiring 已通過。

已執行的無資料測試原樣：

```text
Ran 3 tests in 0.369s

OK
B1 SNAPSHOT PASS full_wiring=NOT_RUN
```

本版人工全通路 smoke（包含獨立 precision 模式）關鍵輸出：

```text
B1 SYNTHETIC PASS tests=10 n_tasks=1 n_arms=3 n_ulp=24 n_steps_total=6604
B1 PRECISION PASS n_tasks=1 n_arms=3 n_ulp=0 n_steps_total=400 full_restore_max_error=0
B1 PRECISION_STANDALONE PASS n_tasks=1 n_arms=3 n_ulp=0 n_steps_total=600 full_restore_max_error=0
B1 MAIN NOT_RUN expected_L_failures=43/200
B1 OFFICIAL_WIRING NOT_RUN synthetic_fixture_only=true
```

本版 zeldajr 官方 val 4 題 wiring 關鍵輸出：

```text
Ran 10 tests in 17.756s

OK
B1 WIRING PASS tests=10 n_tasks=4 n_arms=3 n_ulp=24 n_steps_total=26600
B1 PRECISION PASS n_tasks=4 n_arms=3 n_ulp=0 n_steps_total=1600 full_restore_max_error=0
B1 PRECISION_STANDALONE PASS n_tasks=4 n_arms=3 n_ulp=0 n_steps_total=2400 full_restore_max_error=0
B1 MAIN NOT_RUN expected_L_failures=43/200
```

官方每題 4 個 dose 檔、33 個完整流檔；`n_rollouts=133`、`n_steps_total=4×33×200+200=26600`。
本次 α=0.9 相對 α=0 的失敗率差配對 CI 為 [-0.75, 0.75]；僅屬未檢出差異，未達同等失敗界限。
本版產物：`results_synthetic_b1_back/`、`results_official_b1_back/`。
人工資料的 e_data=0 是本機決定性參考的預期結果，不能作官方資料精度判讀。
獨立 precision 路徑曾發現 scalar metadata 切片錯誤，已修正且列入第10項測試。
