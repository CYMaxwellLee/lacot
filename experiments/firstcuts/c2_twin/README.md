**在證明什麼、判準是什麼**：證明「閉環回饋在擾動後有真實修正價值（方向對）、且集中
在哪個子空間」——判準（預註冊）：真實 obs 臂相對名義雙胞胎臂與錯配臂，擾動後恢復
誤差降低 ≥20%〔拍〕且配對 bootstrap 95% CI 下界 >0〔拍〕、不得靠增加跌倒換 xy；
Γ 配對均值的 CI 上界 <0〔拍〕才判該子空間「回饋確實把偏差往回修」。
⭐ 撤掉「δ=0 必須已經贏」條款：完美名義雙胞胎在零擾動時本應打平（改作接線自檢）。

A 段實作位於 `run_c2.py`，接線檢查位於 `test_c2_wiring.py`。CPU、官方資料只在
zeldajr 使用 `~/.ogbench/data`；`findmnt` guard 及快照 save/restore/RNG/NPZ 函式
原樣複製自 `../b1_triarm/run_b1.py`，test 逐字核對函式。未修改既有 firstcuts 外檔案。

可從 repository 根目錄執行（輸出目錄必須不存在）：

```bash
.venv/bin/python -B experiments/firstcuts/c2_twin/test_c2_wiring.py \
  --synthetic --n-tasks 4 \
  --out-dir experiments/firstcuts/c2_twin/results_example_synthetic_fresh_01

.venv/bin/python -B experiments/firstcuts/c2_twin/test_c2_wiring.py \
  --source-dir experiments/firstcuts/b1_triarm/results_main \
  --n-tasks 4 \
  --out-dir experiments/firstcuts/c2_twin/results_example_official_fresh_01
```

`run_c2.py` 接受相同參數，可單獨跑程式。primary codec `p0_dict_v1` 使用 B1 同款
`p0_dict_v1_L4K32_50k.pt`（SHA256 `b6df04119899942abb7d9d073bdf6a65915eaf38d1dfeb6459309300b25bae53`）；
B1 的 24.5 點回授帳量的是這款。`--ckpt-p0` 預設指向它。`--ckpt-sigma0` / `--ckpt-sigma1` 預設為
`experiments/robust_decoder/results/ckpt_sigma0.pt` / `ckpt_sigma1.0.pt`。
交付沒有啟動 200 題；未來主跑需要同時指定 `--n-tasks 200 --main`，wiring test
無條件禁止主跑。synthetic 最多四題，使用真 MuJoCo Ant 與真 checkpoint，只有起點與
code 教材是合成；不讀官方 dataset、不冒充官方證據。M 必須有另一題 donor，最少兩題。

B1 來源 schema：每題 `task_NNN_epE/task.json`、`snapshot.npz`、`L.npz`。
快照 numeric 欄為 `qpos/qvel/act/warmstart/time/integration`，其他欄放在
`metadata_json`。L 包含 201 個狀態、200 個動作與 50 個 codes；只取起點和 code tape，
不用 B1 的教師 T/R 軌跡作參考。先核對三 checkpoint 的 encoder/codebook 與 B1
完全相同，才復用 code IDs。來源可尚未完成主跑，但所選題的這三個檔案必須完整；
缺檔或 schema 不符即失敗。hash、checkpoint config、來源、套件版本均存 provenance。

三臂與時序固定如下：

- 分岔在 leg-1 起點（chunk 邊界 step 0），每個 codec 先無擾動閉環自跑 36 步，
  並自重放，作自己的名義參考。
- 脈衝只加在 step 0、1 的 actuator action。step 2、3 完成原本已解碼的 chunk；
  step 4 保存完整物理快照，R/N/M 全部從這份快照起跑，同一 code tape、同一時鐘。
  因此沒有在半個 chunk 重啟解碼造成的 δ=0 誤差。
- R 用自己的物理 obs；N 用該 codec 名義軌跡 step 4/8/…/32 的 obs。
  N 的 simulator 照常推進；只替換 decoder 輸入。
- M 用 `nominal_obs ⊕ scaled_other_task_innovation`。innovation 是 R 相對該題名義
  obs 的 28 維向量：14 維 qpos 切向位移與 14 維速度差。`mj_differentiatePos` /
  `mj_integratePos` 處理 quaternion，沒有 quaternion 線性加法。整段 donor 序列
  乘一個純量，使八個解碼時刻的 L2 幅度與 target R innovation 完全一致；保留別題
  的事件／時間結構，不使用白噪音，也不逐時刻偷接 target 的方向。
  donor 是題序循環置換；主跑在校準 40 題、檢驗 160 題內各自置換。
  兩／三題 smoke 為取得 donor 會跨分組，因此只可檢查接線。
- 讀數在共同 post-pulse 邊界之後的 1/2/4/8 chunks，即絕對 step 8/12/20/36。
  不因到達目標或跌倒提前停止，不因某臂好壞改觀測時窗。

工單未指定的數值操作定義，已在跑前寫入程式與 `preregistered.json`：

- 每題每 codec 的名義 4 步位移 RMS 定義為起點至 step 4 的 qpos 切向位移
  `sqrt(mean(dq**2))`（平移、旋轉與關節座標共 14 維）。事先從 task seed 抽 action
  方向；用同一完整快照試算兩步脈衝，校準 step 4 相對名義狀態的切向位移 RMS
  達到上述值的 50%，相對誤差 ≤1%。若飽和下不可達則明確失敗，不降門檻。
  δ=0/.5/1 是校準後 action 脈衝倍率，flip 是負號；非線性下實際位移不保證線性。
  所有 action 最後 clip 到 [-1,1]，每個校準試算的完整逐步流都保存。
- 主三臂各 codec 各自校準 50%。為滿足「同擾動」機制對照，另將 sigma0 的 δ=1
  脈衝投影到兩 codec 各步 action 可用範圍的交集，兩 codec 都另跑 R/N，確保
  **實際施加的加性脈衝相同**，而不是 clip 前相同、clip 後不同（無新增時機臂）。
  輸出保存原提案、共同指令與 realized pulse，test 核對實際增量 ≤1e-15。
  這個共同可行幅度的診斷與各自 50% 校準的主判準分開報。
- gait 為 B1 已用的機身 COM 相對 root、移除 yaw 的形狀向量，從 scratch MjData
  forward 取得；xy 為公尺座標；yaw 為以名義起點對齊的連續旋轉角度，跨 ±π
  不產生虛假的 2π 跳變。C2 **gait=39 維 body-COM 形狀**（W 加權、含 z、去 yaw、
  無速度），xy/yaw 也由 W 加權。C1 的 `project()` 子空間是 SE(2)-quotient＋速度、
  以訓練 sd 正規化；兩者不是同一種量測，**兩邊 per-subspace 數字不可同表比**。
  gait 主判準的理由：C1 判決碼-效果通道在 gait 子空間驗活（碼層事實）；本件
  dict_v1 decoder 無 ŝ 頭，C1 的 ŝ-頭病理由不適用於此處，特此記明。
- W 是每子空間的對角 inverse variance，以校準題名義軌跡所有相隔四步的位移
  擬合（三 codec 合併、population variance、floor=1e-8、除以維數）。主跑原始題序
  前 40 題校準、後 160 題檢驗。擬合後陣列設為唯讀、保存 hash，在擾動 rollout
  前凍結；不讀檢驗題、受擾動臂或最佳結果來擬合。
  四題 smoke 為 2/2，明標 `official_40_160=false`，不是正式 W。
- 恢復誤差是截止點的 `||Y_arm-Y_nominal||²_W`。
  Γ 嚴格使用 `r=Y_N−Y_nominal`、`d=Y_R−Y_N`、`2rᵀWd+dᵀWd`，並測試等於
  R 減 N 的恢復誤差。帳本逐題保存 r/d/交叉項/二次項與四型分類；aggregate
  保存四型計數、交叉項與 d² 項配對 CI。端點 W 讀數下 Γ CI 是 N-benefit CI 的鏡像，
  方向的獨立資訊在交叉項拆分。
- 跌倒是起點至截止點任一 torso z<0.25730210542678833m（約 0.2573m）；此
  `fall_line_p1` 出自 `experiments/walk_verify/results/ruler_pack.json` 的
  `torso_z.fall_line_p1`，與 relay/walk_verify 尺一致。原版 0.2m 為現場拍值，
  `preregistered.json.old_new` 留痕。目標進度為相對 leg-1 固定目標的距離
  減少量。不得用增加跌倒交換任何通過判決；R 相對 N/M 的跌倒率增量都須 ≤0。
- 每題逐劑量保存相對校準目標的 realized displacement ratio、action clip fraction；
  aggregate 以 realized displacement ratio 為劑量反應橫軸，並報中位數及 clip 比例；
  `dose_response_gait.svg` 依 codec／截止窗畫 N、M gait 誤差降幅對實現位移比例。
  δ 是 action 脈衝倍率，不是位移倍率。
- 在 160 檢驗題逐劑量、逐截止點做配對 bootstrap（10,000 次，固定 seed）。
  **唯一 primary_verdict** 是 `p0_dict_v1 × δ=1 × 4-chunk 窗 × gait`：對 N **及** M
  均須降低 ≥20%，誤差差值 CI 下界 >0，Γ 均值 CI 上界 <0，且無跌倒增加。
  其餘全部格照報、標 secondary，不合成另一個主判決。δ=0 是接線自檢；flip 是輔助；
  不把 bootstrap 當跨格多重比較校正。原版未定義合成規則，`old_new` 留痕。

機制比較保存逐題 action response RMS（R−N）及 Γ，並對檢驗題報 sigma1.0−sigma0
的配對 CI。**seed 與訓練噪音劑量綁定，不得單獨歸因訓練噪音**。synthetic/smoke 的
科學判決欄一律 null；PASS 只表示測試成立。

輸出包括 `summary.json`、`preregistered.json`、`frozen_W.npz`、名義自跑／重放、
脈衝與校準全流、post-pulse 完整快照、R/N/M 全流、decoder obs／索引／來源、donor
ID／innovation／scale、機制對照全流。wiring 額外保存 M 餵入 target innovation 的 mutant；
先證明正常 M 的差可見，再確認 mutant 把 R/M 差壓至正常值的 ≤1e-3，且同一錯配檢查
確實拋出 AssertionError。主跑不把「某題正常 M 天然無反應」誤當接線錯誤而剔除題目。

完成錨 `n_tasks/n_arms/n_doses/n_steps_total` 包含所有實驗模擬步數（名義、重放、校準、
脈衝、三臂、flip、mutant 與機制對照），測試從存下的全流重新加總核對。
synthetic 來源 fixture 的 200×n 步另報 `n_steps_fixture`，總量另有
`n_steps_including_fixture`。`.gitignore` 擋所有 NPZ、執行結果目錄與 bytecode。

B 段：**DESIGN_FROZEN / NOT_IMPLEMENTED**。候選庫救援前緣（C1 200 起點×4 碼全流）
vs 一次重寫時刻枚舉 vs 可實作觸發器，同預算對照。依賴 C1 碼庫，等 B1 地板歸屬判決
決定劑量後另開工單；本交付沒有時機臂程式。

2026-09-27 原版實測交付紀錄（zeldajr，`.venv/bin/python -B`；以下為改動前的歷史輸出）：

`results_synthetic_verified`：

```text
C2 SYNTHETIC PASS tests=16 n_tasks=4 n_arms=3 n_doses=3 n_auxiliary_doses=1 n_codecs=2 n_steps_total=4824
C2 ZERO PASS max_R_N_state_linf=0 max_nominal_replay_state_linf=0
C2 MUTANT CAUGHT cells=8 max_collapse_ratio=0
C2 W FROZEN calibration_tasks=2 evaluation_tasks=2 official_40_160=NOT_RUN
C2 MAIN NOT_RUN scientific_verdict=NONE
C2 B DESIGN_FROZEN NOT_IMPLEMENTED
C2 OFFICIAL_WIRING NOT_RUN synthetic_fixture_only=true
```

`results_official_verified`：

```text
C2 OFFICIAL_WIRING PASS tests=16 n_tasks=4 n_arms=3 n_doses=3 n_auxiliary_doses=1 n_codecs=2 n_steps_total=4848
C2 ZERO PASS max_R_N_state_linf=0 max_nominal_replay_state_linf=0
C2 MUTANT CAUGHT cells=8 max_collapse_ratio=0
C2 W FROZEN calibration_tasks=2 evaluation_tasks=2 official_40_160=NOT_RUN
C2 MAIN NOT_RUN scientific_verdict=NONE
C2 B DESIGN_FROZEN NOT_IMPLEMENTED
```

上面的 OFFICIAL_WIRING NOT_RUN 僅指 synthetic 那次 invocation；下一段是另一次官方四題 smoke。
40/160 正式擬合與 200 題科學判決均未執行；未 commit。
