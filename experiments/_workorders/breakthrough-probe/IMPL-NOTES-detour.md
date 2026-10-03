# detour 第四棒（回鍋）：S2 測試與留痕定稿

本棒只補測試、notes、evidence、README。判準＝detour 48/48 全 PASS、17 條新測試各有改壞紅／還原綠證據、全文解析 raw 計數一致、舊三入口 PASS、PB 79 檔與五支程式／sbatch 的第三棒 SHA 不變。已全部驗過；程式邏輯零改動，未跑 GPU／Slurm／jasmine，未 commit。

以下「第三棒」段落與原始輸出保留歷史；第四棒結果見「第四棒驗收與兩態證據」。

## 第三棒歷史紀錄：v8＋S2 MF-1／3／4

把發射單 v8 的改動與 S2 必改項落進現有 detour harness，讓 smoke（E1 首方向表＋15 支）能上機。判準＝新 selftest 全 PASS、舊三個 selftest 原文不動 PASS、PB 79 檔 hash 不變、新閘殺手输入有實際 FAIL。

此段保存第三棒結果；不代表第四棒新增程式邏輯。前兩棒原文仍可在 lead 的 IMPL-REPORT-b1.out／b2.out 查閱；本棒修前／修後原始輸出完整附於下方及 detour-implementation-evidence.json。契約與 source API 已核對，沒有待跳過的 BLOCKED 項；GPU 執行仍未驗。

## 完成與實作決策

- MF-1：worker 與 harvester 的 trace 都改為 float64。initial／goal／chunk 起終點保留環境原數值；initial、goal 指紋用原 dtype 重建。goal 兩點 τ 與尾段 τ 以相同供點函式、原 observation／goal dtype 重建，另記 tau_dtype；不加 tolerance，也不把 raw 位置降精度。
- 先只改測試、未修程式時，test_29 的 float64 O／S／C 三個 subtest 都被 `initial trace` 拒；test_28 真 M9／maze 的第一個 C row 完整收割也同樣拒。原始 2 tests、4 errors、exit 1 已留存。修後 test_29 走 O3／F3／OF／Q-C、完整 goal／零長度／牆格腳本；降成 float32 trace 與錯 dtype τ hash 的反例都被收割拒絕。
- test_28 已從僅 validate_noise 改成完整 validate_row。最終使用實際 smoke 題單：Q-C task1/episode10；O3／F3／OF task4/episode4，draw80、H40。四臂都是真 M9＋s33/EMA＋真 maze、float64 位置、完整列級收割；flow 實呼叫 10／0／10／0。只在記憶體建立 CPU 參照格，沒有產正式 E1，沒有寫 /archive，不能作成績。
- O3 的 w／τ 完全取凍結 oracle route 前 3 步；w≠g 用格心、w=g 用該題實際 noisy goal。每 chunk 重算 cond(obs,target) 並記 w、target_xy、cond_target_xy、h、tau_endpoint。O3 tau_endpoint 獨立重建比對 target；F3 供點同規則，cond 目標獨立重建，解碼末距只記描述，解碼起點使用當下真實位置。recorded_condition 將實際傳入 condition 的目標與 cond tensor 一起回傳，並以 worker 真改傳最終 g 的 source mutant 驗證收割會拒收，非僅改 JSON 欄位。
- O3 靜態快取鍵 (task,c_eff,3)，OF (task,c_eff,None)，兩者均限 Provider 的單 draw 生命期；命中時仍重驗凍結來源格 hash。尾段不進 O3 靜態表、不驗靜態 u hash；此實作尾段每 chunk runtime 構造，完全沒有跨 draw 快取。task4 (6,4) 兩種合法 goal 產生不同 u hash，兩次 draw 共用模型也不串快取。goal 零長度照 v5 沿用原 u 與資格，無前一個 u 則拒。
- F3／Q-F3／Q-C 以原 sample_plan 取 u，臨時攔截真正 module.flow.sample 方法記錄呼叫與 chunk context，finally 還原。每個 chunk 必須恰一次；O3／OF／Q-O3 必須零次。sample_plan 繞過 flow.sample 的 mutant 會在 worker 失敗。F3 的首方向記 runtime-flow 資格；當 goal 與位置重合，P/G 零向量直接 FAIL，但仍照契約每 chunk 新抽 flow u，沒有對 F3 加末點閘。
- OF 保留 v5 O 的整條格心剩餘路、cond 最終 g、goal 兩點／零長度規則；v5 淨空與 Fréchet 只當描述。所有臂沿用原 reset、固定種子、CPU noise generator、每 env step 一次噪音、post success、H1000；不加卡住早停或 chunk 內切換。
- 原第二棒 factory 的 AST 與逐行原始註記不改。本體仍只把 seeds 換成 seed_fn；2組題×舊draw64/71/79 的 stream/flow/noise/global torch RNG bytes 全同，三個 factory mutant 都拒。原 common.seeds 仍拒 draw80。
- E1 門檻是量尺 real[*].cos_4 的 np.quantile(.1)，讀 pin 過的副本；不寫死四捨五入值。arc4 index 與 probe_e1_v6.py 相同；D、P、G 任一長度 <1e-6 直接 FAIL。O3 靜態資格的 G 使用表的 nominal g 格心；尾段使用該題實際 goal。表比較只查兩邊共同靜態格，列 PASS/FAIL 差異；背離格使用原表 turn_away 含 0，沒有另算一套格級規則。
- CPU checker：225 個非 g 正例＋5 個 goal 兩點正例；所有首步背離格直衝最終 g、反向首步皆 FAIL；p10±1e-7 與恰門檻、D/P/G 零向量通過邊界斷言。直衝近目標但切牆角的 PASS 列於 known_limitations，清楚表明「首方向合格」不能證明整段可執行。
- 表名 e1-table-v8.json。cells 僅含 w≠g 的 O3 靜態格；of_cells 保存 v5 OF 描述表。tail 只在 runtime chunks；表與 raw 都留下資格，不因某格 FAIL 拒絕整題或整層。靜態 E0 以凍結表為權威，runtime 預期 hash 在編碼／量化後、head 前記錄。
- 題單重產 408／15 支，只讀原 harvest-m3 questions 與 builder questions.gate。Q-S 已移除，未知臂／未知協定在 worker 與列級收割均拒。
- 三層 O3／F3 各自題級成功与落格，完全拿掉 trustworthy 題級過濾。R1–R5 互斥，與 v8 卡表格的顯示字串完全相等（只去除 Markdown ** 標記）；16 種組合皆測。Q-C<8 優先只加「機制檢查無參考線」，否則 O3／F3 各用自己的變差字串，依序可同時成立。失敗三分只描述；OF 各層報成功率與同題同 draw d_h，無舊刷新效益門檻。
- 舊輸出機制仍 exclusive write＋SHA、RNG 壓縮 sidecar、先落盤再驗。mock 全長15支與完整續跑不重跑、兩支後中斷再補13支均通過；時限 ETA 按408支各 task／arm 權重計算。
- MF-3：formal 註解與 README 明定「時限用命令列 sbatch --time 覆蓋，⛔ 不改檔」。sbatch 是 code hash 一部分，修改會使已凍結 E1 失效。

## 測試環境差異與沒把握處（MF-4）

| 已驗／環境差異 | 未驗與必要條件 |
|---|---|
| float64 腳本的不可由 float32 精確表示位置與真 maze float64 列都完整收割 | 不是只驗噪音；沒有用 tolerance 放寬。GPU 物理全長仍未驗 |
| CPU 真 M9，O3／F3／OF 各一主題＋Q-C 一易題，H40 | GPU 真解碼品質、全長救回率、smoke／formal、實際 ETA 都 UNVERIFIED-GPU |
| 同 CPU process 的 encode→_q→head hash 与 runtime 資格驗證 | **跨 process GPU u hash** 未驗；smoke GPU 表與 formal 重編碼必須逐 hash 相同，失配會拒收，不放寬 |
| coordinator 測試 patch hostname='jasmine'、output root、release flags、loader、synthetic 限制 | 真 `socket.gethostname().split('.')[0] == 'jasmine'` 尚未查；本棒沒有連線或讀 jasmine 指紋檔 |
| 本機測試只用記憶體與 /tmp；sbatch 腳本 bash -n PASS | **sbatch --output 父目錄要先建**，Slurm 在 Python 前開 log；lead 提交前準備 /archive/cymaxwelllee/breakthrough1/detour-u/ |
| loader 原樣用 repo 相對的 M9／collector／checkpoint | **凍結樹要帶 results/*.pt 與 lacot 套件**，並保留 PB 與 M9 相對結構；只帶 PB 不足 |
| receipt 比對模型、code、E1、normalization 與絕對路徑 | **smoke 與 formal 同根路徑、同 DETOUR_DATASET_DIR**、相同 interpreter／確定性環境；不能搬樹後拿舊表直接跑 |
| checker round-trip／直衝g／反向／邊界／零向量都經純 CPU 對照 | 首方向只看開頭，近目標切牆角 PASS 是已知限制；淨空／Fréchet 不再是主閘 |
| factory 與每步 CPU noise 用原函式；舊三個入口未修改 | CPU 並不能代替 GPU 的實際 noise 對軌跡影響或成功率 |

正式入口沒有測試 patch，兩個 flags 仍 False。正式表生成限 jasmine deterministic GPU，寫入仍限本機 /archive；本棒沒有生成正式表／送 sbatch／跑 GPU／連 jasmine／commit。沒有觀測到需要擅改契約的接口差異。

## 輸出契約兩欄

| 成功路徑保留的輸出 | 允許變更的行為 |
|---|---|
| protocol/task/episode/group/arm/draw、完整 seeds、source與模型／dataset／code／E1 SHA、環境屬性與原 dtype | 新協定 detour-u-v8，臂／題單照 v8；舊 PB 協定完全不改 |
| float64 trace 與 SHA、初始／goal 原 dtype SHA、每步成功／動作／噪音狀態、chunk step與位移 | 只因success/terminated/truncated/H停止，末chunk可不足4步 |
| 每chunk c_raw/c_eff、w/target_xy/cond_target_xy/h，O3 τ末點/dtype/hash、u與head hash、資格與來源 | tail runtime、零長度繼承；F3不驗解碼末點，僅記末距描述 |
| E1閾值來源、靜態格與OF描述格、checker對照、CPU比較差異、背離格；runtime資格 | CPU mock不能替代正式GPU表，不因E1 FAIL事後刪分母 |
| 三層O3/F3落格、逐題bits、R1-R5原字串、Q各自加註、失敗三分與順從度、OF成功率與d_h | smoke單draw標觀測不足；三分與d_h描述而不做病因定位 |
| checkpoint/raw/result、exclusive shards/SHA、失敗證據、每task/arm速度與加權ETA | 續跑新目錄重用已驗列；lead只用命令列覆蓋時限 |

## 完成錨計數

新 detour 檔共 21（原17＋量尺/CPU先量副本與sidecar共4）；PB 79/79 hash 全同，git diff 為空，git status 只有新檔。第三棒 unittest 31/31 PASS（第四棒為 48/48），FAIL0／ERROR0／SKIP0。舊 selftest 44/44、fourth 獨立14/14（已含於44中），runtime REAL_CPU_PASS＋PROTOCOL_EQUIVALENCE_PASS；不重複計成58個舊單元測試。

第三棒最終 raw 有 **74 條 TWO_STATE**，按前綴名稱去重 **72 種**（tail兩合法goal各執行同名殺手一次）。本棒已改用全文 `re.findall(r"TWO_STATE [^\r\n]*", raw)`，不加行首錨，補回與 test_22 標頭同列的 release-False；原始 raw 不改。第四棒最終 raw 則為 **89 條／87 種**，由第四棒 raw 重新解析。修前兩個測試的4個拒收另列，不混入最終兩態計數。

O3静態格各task數：`{"1": 42, "2": 42, "3": 37, "4": 42, "5": 42}`，共205；OF描述格225。門檻實值 `0.8144576277757829`。題單408、smoke15，兩旗False。AST parse、兩份bash -n、所有sidecar和來源逐byte副本、evidence.code_sha256活檔一致皆PASS。

## 新舊 SHA

原始發射單／trapset／量尺均只讀。CONTRACT-FROZEN-detour.md 升級為 v8 逐byte副本；plan 重產；其sidecar同步。code／測試／sbatch變更如下（notes與evidence不自參照hash）：

| 檔 | 修前 SHA256 | 修後 SHA256 |
|---|---|---|
| CONTRACT-FROZEN-detour.md.sha256 | b00247eadd09c2d23467b4367824a52347e3969f42103acb91c0014c0969aaed | 79e35af2618844948d4378c4c500821e1327e35098715b06210d6a4a79238960 |
| detour-formal.sbatch | 5bf51aa331fd9f2ad3d3700a046c371ff078e0bb3b033446b9a7f0390a1f8c22 | 004cb996c43b51b0931200fdbafdb735cf05ac9c92f9df9d1fd3345f78234e67 |
| README-detour.md | c119d92ca18e89002e71c37c79f565713d369725f91fb448eed7ff0d8c439905 | 47abee6cf90848e1a2c20abc70ff459885675da913f2f0e69f4c16294256938d |
| harvest_detour.py | 8b2c4f141b8126c19d1f5ac9f3a68e71c39a49c1f67aa24283a08326d1bb0600 | ad2d50b5ea34c255844457314aadcc3f359bdca2b25433f594c618c2d51ec1fc |
| detour_plan.py | 7701a46ba7d06192673c034426528a3e2e9e6ceccd7985f52f9d24af0017a400 | 81f404af2ab19d0ef4167e19312efa845208a11d4c2ce26d840a4e1bef58a176 |
| detour-smoke.sbatch | bf941253fb9cea4b24fcf2498d86b0760fc1042c788ca1c7f73befb2151208e2 | a39eba0b8d44914f99f776af525d6527ff85635183835fb5c3ece43731591fa9 |
| CONTRACT-FROZEN-detour.md | 3b816cf97e1d9e270264699bdcf3b0893de012389fc9e2ff5a8a78322f51e2c4 | 206306391e136e309b52e4184a3a9fa94dce8128df2a7d832289a8b6ca184051 |
| harness_detour.py | 04348fc37019aab69c988c1aca52e1c0d49b21639af03aeaa8c4efa9785fe45a | b6da954eea5b4982644c5613cbf3b24b6d6c35ae919b2be582acb804b9a28514 |
| detour-plan.json.sha256 | 486957e1467dc770468373c7ed03af5dab1180f7342f29928d0bba1b09b58b06 | 6a279e16709b80c29fe968cf154c9ea11a40da06005901ee99e7ec67388e1666 |
| detour_selftest.py | 19a5be77d3cbabbee3f8da2fd31c55372223f02f601c3d7e59afb717f420f9fa | 8ee52f7eab0fcc5ec44e100a0b78349870350e1161bf93f4ba6a6355c8fbc349 |
| detour-plan.json | 493ecf1c7b3f6b572a3dfb9fc16b8b2cbc7ba93eb90dc6dd7cff5a98b4cc8b93 | 34501e4301bb38b5461ca394b5fca49a3b7059b20b74a11ef80d05eaf413d1f7 |

新增量尺與CPU先量副本SHA：

- detour-horizon-ruler.json: `c3af2cc005f6a1a355fbb49af5af927d171d3d765aa32be4947ff89fd7536805`
- detour-e1-v6-h3.json: `a248ab6be1e968d572fda812bae4348852959bb407ba02134e8cb829454f0de8`

## 原始驗證輸出全文

修前先跑測試未修改harness；最終自測與舊三個入口各自記錄exit code與raw SHA。fourth_selftest.py沒有main runner，依原第二棒用unittest discovery執行14個測試，原檔未動。

### before

命令：`.venv/bin/python -B experiments/_workorders/breakthrough-probe/detour_selftest.py DetourTests.test_29_float64_full_harvest DetourTests.test_28_cpu_real_short_wiring`

exit code：1；raw SHA256：`96a2e441c0517f6a848e1e4134d38c08c5b9adba582a663732401512d8c34459`

```text
test_29_float64_full_harvest (__main__.DetourTests.test_29_float64_full_harvest) ... 
  test_29_float64_full_harvest (__main__.DetourTests.test_29_float64_full_harvest) (arm='O') ... ERROR
  test_29_float64_full_harvest (__main__.DetourTests.test_29_float64_full_harvest) (arm='S') ... ERROR
  test_29_float64_full_harvest (__main__.DetourTests.test_29_float64_full_harvest) (arm='C') ... ERROR
test_28_cpu_real_short_wiring (__main__.DetourTests.test_28_cpu_real_short_wiring)
Real M9/ckpt + real maze, in memory only; short wiring, never a score/E1 artifact. ... device: cpu
  teacher 題庫：4096 條（細格步數 p50 26 p90 42 max 62），mix=0.5
設定：seed=33 cons=self ema_m=0.996 K=8 COND=256
stage 1 e_target 目標=recon_ictr ...  w_var=0.0 w_cov=0.0 w_ictr=0.2 sigma=0.05  warmup=500
  e_target match-acc(train batch) nan
/home/cymaxwelllee/Projects/lacot/.venv/lib/python3.11/site-packages/torch/nn/modules/transformer.py:385: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.norm_first was True
  warnings.warn(
stage 2 flow+refine+action ...
  ⭐ 已切到 EMA 影子權重（m=0.999，五模組）
✅ 載入 ckpt_large-stitch_self_K8_c256_ch4_st8000_T128_ep2_gu_eorecon_ictr_tch0.5_emw0.999_wu500_dssoft_norf_cd0.1_bci_s33.pt（跳過訓練，只跑評估）  cfg={'K': 8, 'T_CAP': 128, 'ENC_OBJ': 'recon_ictr', 'LEARNED_REFINE': 0, 'COND_DROP': 0.1}
  decoder 內部點 RMSE 0.0981   打亂 u 之後 1.8960   用到的 u 值 +1.7979   ✓
ERROR

======================================================================
ERROR: test_29_float64_full_harvest (__main__.DetourTests.test_29_float64_full_harvest) (arm='O')
----------------------------------------------------------------------
Traceback (most recent call last):
  File "/home/cymaxwelllee/Projects/lacot/experiments/_workorders/breakthrough-probe/detour_selftest.py", line 620, in test_29_float64_full_harvest
    self.validate(row,env,plan)
  File "/home/cymaxwelllee/Projects/lacot/experiments/_workorders/breakthrough-probe/detour_selftest.py", line 109, in validate
    h.validate_row(row,plan,self.table,env.xy_to_ij,env.ij_to_xy,fixture=True)
  File "/home/cymaxwelllee/Projects/lacot/experiments/_workorders/breakthrough-probe/harvest_detour.py", line 87, in validate_row
    _validate_row(row,plan,table,xy_to_cell,cell_to_xy,fixture=fixture)
  File "/home/cymaxwelllee/Projects/lacot/experiments/_workorders/breakthrough-probe/harvest_detour.py", line 109, in _validate_row
    require(np.array_equal(trace[0],row['initial'][:2]),'initial trace')
  File "/home/cymaxwelllee/Projects/lacot/experiments/_workorders/breakthrough-probe/harvest_detour.py", line 18, in require
    raise Rejected(message)
harvest_detour.Rejected: initial trace

======================================================================
ERROR: test_29_float64_full_harvest (__main__.DetourTests.test_29_float64_full_harvest) (arm='S')
----------------------------------------------------------------------
Traceback (most recent call last):
  File "/home/cymaxwelllee/Projects/lacot/experiments/_workorders/breakthrough-probe/detour_selftest.py", line 620, in test_29_float64_full_harvest
    self.validate(row,env,plan)
  File "/home/cymaxwelllee/Projects/lacot/experiments/_workorders/breakthrough-probe/detour_selftest.py", line 109, in validate
    h.validate_row(row,plan,self.table,env.xy_to_ij,env.ij_to_xy,fixture=True)
  File "/home/cymaxwelllee/Projects/lacot/experiments/_workorders/breakthrough-probe/harvest_detour.py", line 87, in validate_row
    _validate_row(row,plan,table,xy_to_cell,cell_to_xy,fixture=fixture)
  File "/home/cymaxwelllee/Projects/lacot/experiments/_workorders/breakthrough-probe/harvest_detour.py", line 109, in _validate_row
    require(np.array_equal(trace[0],row['initial'][:2]),'initial trace')
  File "/home/cymaxwelllee/Projects/lacot/experiments/_workorders/breakthrough-probe/harvest_detour.py", line 18, in require
    raise Rejected(message)
harvest_detour.Rejected: initial trace

======================================================================
ERROR: test_29_float64_full_harvest (__main__.DetourTests.test_29_float64_full_harvest) (arm='C')
----------------------------------------------------------------------
Traceback (most recent call last):
  File "/home/cymaxwelllee/Projects/lacot/experiments/_workorders/breakthrough-probe/detour_selftest.py", line 620, in test_29_float64_full_harvest
    self.validate(row,env,plan)
  File "/home/cymaxwelllee/Projects/lacot/experiments/_workorders/breakthrough-probe/detour_selftest.py", line 109, in validate
    h.validate_row(row,plan,self.table,env.xy_to_ij,env.ij_to_xy,fixture=True)
  File "/home/cymaxwelllee/Projects/lacot/experiments/_workorders/breakthrough-probe/harvest_detour.py", line 87, in validate_row
    _validate_row(row,plan,table,xy_to_cell,cell_to_xy,fixture=fixture)
  File "/home/cymaxwelllee/Projects/lacot/experiments/_workorders/breakthrough-probe/harvest_detour.py", line 109, in _validate_row
    require(np.array_equal(trace[0],row['initial'][:2]),'initial trace')
  File "/home/cymaxwelllee/Projects/lacot/experiments/_workorders/breakthrough-probe/harvest_detour.py", line 18, in require
    raise Rejected(message)
harvest_detour.Rejected: initial trace

======================================================================
ERROR: test_28_cpu_real_short_wiring (__main__.DetourTests.test_28_cpu_real_short_wiring)
Real M9/ckpt + real maze, in memory only; short wiring, never a score/E1 artifact.
----------------------------------------------------------------------
Traceback (most recent call last):
  File "/home/cymaxwelllee/Projects/lacot/experiments/_workorders/breakthrough-probe/detour_selftest.py", line 604, in test_28_cpu_real_short_wiring
    h.validate_row(row,plan,module.detour_e1_table,env.unwrapped.xy_to_ij,env.unwrapped.ij_to_xy)
  File "/home/cymaxwelllee/Projects/lacot/experiments/_workorders/breakthrough-probe/harvest_detour.py", line 87, in validate_row
    _validate_row(row,plan,table,xy_to_cell,cell_to_xy,fixture=fixture)
  File "/home/cymaxwelllee/Projects/lacot/experiments/_workorders/breakthrough-probe/harvest_detour.py", line 109, in _validate_row
    require(np.array_equal(trace[0],row['initial'][:2]),'initial trace')
  File "/home/cymaxwelllee/Projects/lacot/experiments/_workorders/breakthrough-probe/harvest_detour.py", line 18, in require
    raise Rejected(message)
harvest_detour.Rejected: initial trace

----------------------------------------------------------------------
Ran 2 tests in 6.919s

FAILED (errors=4)
```

### detour

命令：`.venv/bin/python -B experiments/_workorders/breakthrough-probe/detour_selftest.py`

exit code：0；raw SHA256：`483047b547c0c4e82bbb8803ff21061c65fa096667710ede7d4aadac6a395188`

```text
test_01_pins_flags_seeds (__main__.DetourTests.test_01_pins_flags_seeds) ... ok
test_02_plan_machine_only_and_inventory (__main__.DetourTests.test_02_plan_machine_only_and_inventory) ... ok
test_03_table_all_cells_and_roundtrip (__main__.DetourTests.test_03_table_all_cells_and_roundtrip) ... ok
test_04_exact_segment_square (__main__.DetourTests.test_04_exact_segment_square) ... ok
test_05_all_checker_killers_and_boundaries (__main__.DetourTests.test_05_all_checker_killers_and_boundaries) ... ok
test_06_sampled_clearance_mutant (__main__.DetourTests.test_06_sampled_clearance_mutant) ... ok
test_07_unordered_frechet_mutant (__main__.DetourTests.test_07_unordered_frechet_mutant) ... ok
test_08_provider_head_and_noise_forward (__main__.DetourTests.test_08_provider_head_and_noise_forward) ... ok
test_09_goal_zero_wall_script (__main__.DetourTests.test_09_goal_zero_wall_script) ... ok
test_10_zero_inherits_failure_and_no_previous (__main__.DetourTests.test_10_zero_inherits_failure_and_no_previous) ... ok
test_11_provider_flow_mutant (__main__.DetourTests.test_11_provider_flow_mutant) ... ok
test_12_noise_removed_mutant (__main__.DetourTests.test_12_noise_removed_mutant) ... ok
test_13_raw_cache_mutant (__main__.DetourTests.test_13_raw_cache_mutant) ... ok
test_14_targets_flow_and_quantization (__main__.DetourTests.test_14_targets_flow_and_quantization) ... ok
test_15_reject_artifact_matrix (__main__.DetourTests.test_15_reject_artifact_matrix) ... ok
test_16_receipt_and_exclusive_shards (__main__.DetourTests.test_16_receipt_and_exclusive_shards) ... ok
test_17_question_layers_thresholds (__main__.DetourTests.test_17_question_layers_thresholds) ... ok
test_18_failure_partition_norm_and_compliance (__main__.DetourTests.test_18_failure_partition_norm_and_compliance) ... ok
test_19_exact_verdicts_and_exclusivity (__main__.DetourTests.test_19_exact_verdicts_and_exclusivity) ... ok
test_20_full_horizon_fifteen_mock_rollouts (__main__.DetourTests.test_20_full_horizon_fifteen_mock_rollouts) ... ok
test_21_original_factory_rejects_and_detour_passes (__main__.DetourTests.test_21_original_factory_rejects_and_detour_passes) ... ok
test_22_coordinator_freeze_fifteen_shards_resume (__main__.DetourTests.test_22_coordinator_freeze_fifteen_shards_resume) ... TWO_STATE release-False: correct=PASS; killer/mutant=FAIL (observed rejection)
TWO_STATE release-True: correct=PASS; killer/mutant=FAIL (observed rejection)
TWO_STATE plan-inventory: correct=PASS; killer/mutant=FAIL (observed rejection)
TWO_STATE mock-production-reject: correct=PASS; killer/mutant=FAIL (observed rejection)
TWO_STATE tail-in-static-table: correct=PASS; killer/mutant=FAIL (observed rejection)
TWO_STATE E1-final-g: true-tau=PASS; all departure-cell final-g straight lines=FAIL
TWO_STATE E1-reverse: true-tau=PASS; all departure-cell reversed first directions=FAIL
TWO_STATE E1-boundary: p10+1e-7=PASS; p10-1e-7=FAIL
TWO_STATE E1-zero: nonzero=PASS; D/P/G zero=FAIL
TWO_STATE E1-always-pass: correct=PASS; killer/mutant=FAIL (observed rejection)
TWO_STATE zero-no-previous: correct=PASS; killer/mutant=FAIL (observed rejection)
TWO_STATE provider-flow-E0: correct=PASS; killer/mutant=FAIL (observed rejection)
TWO_STATE noise-removed: correct=PASS; killer/mutant=FAIL (observed rejection)
TWO_STATE flow-seed-conditional-validator: correct=PASS; killer/mutant=FAIL (observed rejection)
TWO_STATE M3-c_raw-cache: correct=PASS; killer/mutant=FAIL (observed rejection)
TWO_STATE target-O3: correct=PASS; killer/mutant=FAIL (observed rejection)
TWO_STATE unknown-protocol-harvest-O3: correct=PASS; killer/mutant=FAIL (observed rejection)
TWO_STATE unknown-protocol-worker-O3: correct=PASS; killer/mutant=FAIL (observed rejection)
TWO_STATE target-F3: correct=PASS; killer/mutant=FAIL (observed rejection)
TWO_STATE unknown-protocol-harvest-F3: correct=PASS; killer/mutant=FAIL (observed rejection)
TWO_STATE unknown-protocol-worker-F3: correct=PASS; killer/mutant=FAIL (observed rejection)
TWO_STATE target-OF: correct=PASS; killer/mutant=FAIL (observed rejection)
TWO_STATE unknown-protocol-harvest-OF: correct=PASS; killer/mutant=FAIL (observed rejection)
TWO_STATE unknown-protocol-worker-OF: correct=PASS; killer/mutant=FAIL (observed rejection)
TWO_STATE target-Q-C: correct=PASS; killer/mutant=FAIL (observed rejection)
TWO_STATE unknown-protocol-harvest-Q-C: correct=PASS; killer/mutant=FAIL (observed rejection)
TWO_STATE unknown-protocol-worker-Q-C: correct=PASS; killer/mutant=FAIL (observed rejection)
TWO_STATE target-Q-O3: correct=PASS; killer/mutant=FAIL (observed rejection)
TWO_STATE unknown-protocol-harvest-Q-O3: correct=PASS; killer/mutant=FAIL (observed rejection)
TWO_STATE unknown-protocol-worker-Q-O3: correct=PASS; killer/mutant=FAIL (observed rejection)
TWO_STATE target-Q-F3: correct=PASS; killer/mutant=FAIL (observed rejection)
TWO_STATE unknown-protocol-harvest-Q-F3: correct=PASS; killer/mutant=FAIL (observed rejection)
TWO_STATE unknown-protocol-worker-Q-F3: correct=PASS; killer/mutant=FAIL (observed rejection)
TWO_STATE F3-final-g-target: correct=PASS; killer/mutant=FAIL (observed rejection)
TWO_STATE F3-worker-final-g-condition: correct=PASS; killer/mutant=FAIL (observed rejection)
TWO_STATE F3-call-count: correct=PASS; killer/mutant=FAIL (observed rejection)
TWO_STATE F3-fake-sampling: correct=PASS; killer/mutant=FAIL (observed rejection)
TWO_STATE O3-encoding-endpoint: correct=PASS; killer/mutant=FAIL (observed rejection)
TWO_STATE head-must-equal-frozen-quantization: correct=PASS; killer/mutant=FAIL (observed rejection)
TWO_STATE unknown-arm: correct=PASS; killer/mutant=FAIL (observed rejection)
TWO_STATE missing: correct=PASS; killer/mutant=FAIL (observed rejection)
TWO_STATE duplicate: correct=PASS; killer/mutant=FAIL (observed rejection)
TWO_STATE seed: correct=PASS; killer/mutant=FAIL (observed rejection)
TWO_STATE draw: correct=PASS; killer/mutant=FAIL (observed rejection)
TWO_STATE u-self-certify: correct=PASS; killer/mutant=FAIL (observed rejection)
TWO_STATE missing-qualification: correct=PASS; killer/mutant=FAIL (observed rejection)
TWO_STATE noise-state: correct=PASS; killer/mutant=FAIL (observed rejection)
TWO_STATE missing-chunk: correct=PASS; killer/mutant=FAIL (observed rejection)
TWO_STATE false-success: correct=PASS; killer/mutant=FAIL (observed rejection)
TWO_STATE early-stop: correct=PASS; killer/mutant=FAIL (observed rejection)
TWO_STATE missing-action: correct=PASS; killer/mutant=FAIL (observed rejection)
TWO_STATE receipt-card_sha256: correct=PASS; killer/mutant=FAIL (observed rejection)
TWO_STATE receipt-code_sha256: correct=PASS; killer/mutant=FAIL (observed rejection)
TWO_STATE receipt-collector_sha256: correct=PASS; killer/mutant=FAIL (observed rejection)
TWO_STATE receipt-e1_sha256: correct=PASS; killer/mutant=FAIL (observed rejection)
TWO_STATE receipt-plan_sha256: correct=PASS; killer/mutant=FAIL (observed rejection)
TWO_STATE exclusive-write: correct=PASS; killer/mutant=FAIL (observed rejection)
TWO_STATE shard-tamper: correct=PASS; killer/mutant=FAIL (observed rejection)
TWO_STATE exact-R3-wording: correct=PASS; killer/mutant=FAIL (observed rejection)
MOCK_FIFTEEN_PASS: 15 x H=1000; real detour collector factory, no factory patch; NOT production smoke
FACTORY_RESOLVED: original draw80 rejects; copied factory draw80..83 and real C path PASS
detour rollouts=1/15
detour rollouts=2/15
detour rollouts=3/15
detour rollouts=4/15
detour rollouts=5/15
detour rollouts=6/15
detour rollouts=7/15
detour rollouts=8/15
detour rollouts=9/15
detour rollouts=10/15
detour rollouts=11/15
detour rollouts=12/15
detour rollouts=13/15
detour rollouts=14/15
detour rollouts=15/15
detour rollouts=1/15
detour rollouts=2/15
detour rollouts=3/15
detour rollouts=4/15
detour rollouts=5/15
detour rollouts=6/15
detour rollouts=7/15
detour rollouts=8/15
detour rollouts=9/15
detour rollouts=10/15
detour rollouts=11/15
detour rollouts=12/15
detour rollouts=13/15
detour rollouts=14/15
detour rollouts=15/15
ok
test_23_partial_failure_resume_and_fail_closed_entry (__main__.DetourTests.test_23_partial_failure_resume_and_fail_closed_entry) ... MOCK_COORDINATOR_PASS: E1 exclusive freeze -> 15 shards -> independent harvest -> resume without rerun; gates patched in test only
detour rollouts=1/15
detour rollouts=2/15
detour rollouts=1/15
detour rollouts=2/15
detour rollouts=3/15
detour rollouts=4/15
detour rollouts=5/15
detour rollouts=6/15
detour rollouts=7/15
detour rollouts=8/15
detour rollouts=9/15
detour rollouts=10/15
detour rollouts=11/15
detour rollouts=12/15
detour rollouts=13/15
detour rollouts=14/15
detour rollouts=15/15
ok
test_24_formal_readout_denominators_and_q_pairs (__main__.DetourTests.test_24_formal_readout_denominators_and_q_pairs) ... ok
test_25_factory_byte_equivalence_and_line_copy (__main__.DetourTests.test_25_factory_byte_equivalence_and_line_copy) ... ok
test_26_factory_equivalence_killers (__main__.DetourTests.test_26_factory_equivalence_killers) ... ok
test_27_three_arms_noise_bytes_and_factory (__main__.DetourTests.test_27_three_arms_noise_bytes_and_factory) ... ok
test_28_cpu_real_short_wiring (__main__.DetourTests.test_28_cpu_real_short_wiring)
Real M9/ckpt + real maze, in memory only; short wiring, never a score/E1 artifact. ... FACTORY_BYTE_EQUIVALENCE_PASS: 2 task/episode pairs x draws 64,71,79; stream/flow, noise bytes, global torch RNG bytes
TWO_STATE factory-arm-B: correct=PASS; killer/mutant=FAIL (observed rejection)
TWO_STATE factory-sigma: correct=PASS; killer/mutant=FAIL (observed rejection)
TWO_STATE factory-noise-state: correct=PASS; killer/mutant=FAIL (observed rejection)
THREE_ARM_NOISE_BYTES_PASS: O3/OF/Q-C same task4 episode4 draw80; 40 per-step samples; flow calls 0/0/10
device: cpu
  teacher 題庫：4096 條（細格步數 p50 26 p90 42 max 62），mix=0.5
設定：seed=33 cons=self ema_m=0.996 K=8 COND=256
stage 1 e_target 目標=recon_ictr ...  w_var=0.0 w_cov=0.0 w_ictr=0.2 sigma=0.05  warmup=500
  e_target match-acc(train batch) nan
/home/cymaxwelllee/Projects/lacot/.venv/lib/python3.11/site-packages/torch/nn/modules/transformer.py:385: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.norm_first was True
  warnings.warn(
stage 2 flow+refine+action ...
  ⭐ 已切到 EMA 影子權重（m=0.999，五模組）
✅ 載入 ckpt_large-stitch_self_K8_c256_ch4_st8000_T128_ep2_gu_eorecon_ictr_tch0.5_emw0.999_wu500_dssoft_norf_cd0.1_bci_s33.pt（跳過訓練，只跑評估）  cfg={'K': 8, 'T_CAP': 128, 'ENC_OBJ': 'recon_ictr', 'LEARNED_REFINE': 0, 'COND_DROP': 0.1}
  decoder 內部點 RMSE 0.0981   打亂 u 之後 1.8960   用到的 u 值 +1.7979   ✓
ok
test_29_float64_full_harvest (__main__.DetourTests.test_29_float64_full_harvest) ... ok
test_30_tail_runtime_two_goals_and_draws (__main__.DetourTests.test_30_tail_runtime_two_goals_and_draws) ... ok
test_31_ruler_threshold_is_live_and_pinned (__main__.DetourTests.test_31_ruler_threshold_is_live_and_pinned) ... ok

----------------------------------------------------------------------
Ran 31 tests in 52.464s

OK
DETOUR_REAL_CPU_PASS: M9+s33/EMA, real maze, Q-C task1 episode10 and O3/F3/OF task4 episode4 draw80 H=40; full validate_row PASS, flow=10/0/10/0, noise bytes paired, delivered u hashes and qualification fields checked; score=false, GPU=false, archive_writes=0
TWO_STATE float32-trace-O3: correct=PASS; killer/mutant=FAIL (observed rejection)
TWO_STATE goal-tau-dtype-O3: correct=PASS; killer/mutant=FAIL (observed rejection)
TWO_STATE float32-trace-F3: correct=PASS; killer/mutant=FAIL (observed rejection)
TWO_STATE float32-trace-OF: correct=PASS; killer/mutant=FAIL (observed rejection)
TWO_STATE goal-tau-dtype-OF: correct=PASS; killer/mutant=FAIL (observed rejection)
TWO_STATE float32-trace-Q-C: correct=PASS; killer/mutant=FAIL (observed rejection)
TWO_STATE tail-static-hash: correct=PASS; killer/mutant=FAIL (observed rejection)
TWO_STATE tail-runtime-qualification: correct=PASS; killer/mutant=FAIL (observed rejection)
TWO_STATE tail-static-hash: correct=PASS; killer/mutant=FAIL (observed rejection)
TWO_STATE tail-runtime-qualification: correct=PASS; killer/mutant=FAIL (observed rejection)
TWO_STATE tail-goal-u: separate draws distinct runtime hashes=PASS; stale first-draw hash=FAIL
TWO_STATE rounded-hardcoded-p10: correct=PASS; killer/mutant=FAIL (observed rejection)
```

### selftest

命令：`CUDA_VISIBLE_DEVICES='' CUBLAS_WORKSPACE_CONFIG=:4096:8 MUJOCO_GL=egl .venv/bin/python -B experiments/_workorders/breakthrough-probe/selftest.py`

exit code：0；raw SHA256：`8e68186b3e8f98c1e4088a8f1a5b12ed6a3cd8e5c3026237d797af2aaa8ec5cd`

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
PROTOCOL_EQUIVALENCE_PASS: 3 x 160 real maze steps, frozen oracle A, actions + XY byte-identical
REAL_CPU_PASS: M9 import, s33/EMA, encode_u false mask, head, decode, four P/Q same-u reruns, fresh-flow seed, legal-w short cap; GPU evidence=false
{"ckpt_sha256": "88180676e1f8d8df22c47bf4eeaf23eba6a73c5e92cacbf522ce3044095af787", "encoded_u_sha256": "fd89a90c7385dfd5c57288d7a2511b188fcb03c867cdf5e23bc0ab72ebb961a9", "encoder_shape": [1, 8, 256], "head_shape": [1, 4, 2], "head_u_hashes": ["fd89a90c7385dfd5c57288d7a2511b188fcb03c867cdf5e23bc0ab72ebb961a9"], "mask_all_false": true, "production_evidence": false, "source_sha256": "276c68fba6c41a05c1e3c2bd5bf4b25c39642d0dbac3ca7aa57cce381056ecfc"}
ok
test_24_move1_108_limit_and_conditional_rows (__main__.ContractTests.test_24_move1_108_limit_and_conditional_rows) ... ok
test_25_short_trace_contract (__main__.ContractTests.test_25_short_trace_contract) ... ok
test_26_d4_exclusive_crossing_symmetric (__main__.ContractTests.test_26_d4_exclusive_crossing_symmetric) ... ok
test_27_m9_pin_and_import_boundary (__main__.ContractTests.test_27_m9_pin_and_import_boundary) ... ok
test_28_full_artifact_conditional_coverage_and_donors (__main__.ContractTests.test_28_full_artifact_conditional_coverage_and_donors) ... ok
test_29_generated_contract_snapshots (__main__.ContractTests.test_29_generated_contract_snapshots) ... ok
test_30_freeze_flags_and_fixed_hash_inventory (fourth_selftest.FourthTests.test_30_freeze_flags_and_fixed_hash_inventory) ... ok
test_31_move1_positive_difference_and_conformance_downgrade (fourth_selftest.FourthTests.test_31_move1_positive_difference_and_conformance_downgrade) ... ok
test_32_reproduction_and_gate_threshold_four_five (fourth_selftest.FourthTests.test_32_reproduction_and_gate_threshold_four_five) ... ok
test_33_harvest_hostile_evidence (fourth_selftest.FourthTests.test_33_harvest_hostile_evidence) ... ok
test_34_historical_A_B_fingerprints (fourth_selftest.FourthTests.test_34_historical_A_B_fingerprints) ... ok
test_35_new_card_and_rule_boundaries (fourth_selftest.FourthTests.test_35_new_card_and_rule_boundaries) ... ok
test_36_combined_exits_from_full_artifacts (fourth_selftest.FourthTests.test_36_combined_exits_from_full_artifacts) ... ok
test_37_stub_fuzz_runloop_noise_switch_target_stuck (fourth_selftest.FourthTests.test_37_stub_fuzz_runloop_noise_switch_target_stuck) ... ok
test_38_runtime_coordinator_sources_gates_shards_heartbeat (fourth_selftest.FourthTests.test_38_runtime_coordinator_sources_gates_shards_heartbeat) ... PROGRESS move=1 rollouts=50/108 task=4 episode=20
PROGRESS move=1 rollouts=100/108 task=4 episode=4
PROGRESS move=1 rollouts=50/108 task=4 episode=18
PROGRESS move=1 rollouts=100/108 task=5 episode=32
ok
test_39_raw_survives_validation_failure (fourth_selftest.FourthTests.test_39_raw_survives_validation_failure) ... ok
test_40_calibration_fail_fast_and_estimators (fourth_selftest.FourthTests.test_40_calibration_fail_fast_and_estimators) ... ok
test_41_remaining_boundary_and_preprocessing_mutants (fourth_selftest.FourthTests.test_41_remaining_boundary_and_preprocessing_mutants) ... ok
test_42_checkpoint_resume_without_rerunning_completed_rows (fourth_selftest.FourthTests.test_42_checkpoint_resume_without_rerunning_completed_rows) ... PROGRESS move=1 rollouts=50/108 task=4 episode=18
PROGRESS move=1 rollouts=100/108 task=5 episode=32
PROGRESS move=1 rollouts=50/108 task=4 episode=18
PROGRESS move=1 rollouts=100/108 task=5 episode=32
ok
test_43_move3_discordance_both_directions_from_traces (fourth_selftest.FourthTests.test_43_move3_discordance_both_directions_from_traces) ... ok

----------------------------------------------------------------------
Ran 44 tests in 16.584s

OK
STATUS: DONE
```

### runtime

命令：`CUDA_VISIBLE_DEVICES='' CUBLAS_WORKSPACE_CONFIG=:4096:8 MUJOCO_GL=egl .venv/bin/python -B experiments/_workorders/breakthrough-probe/runtime_selftest.py`

exit code：0；raw SHA256：`8eb7b26d5c5058cc61769e8123cd70937209924d5484f9548e043ca12bbb4717`

```text
device: cpu
  teacher 題庫：4096 條（細格步數 p50 26 p90 42 max 62），mix=0.5
設定：seed=33 cons=self ema_m=0.996 K=8 COND=256
stage 1 e_target 目標=recon_ictr ...  w_var=0.0 w_cov=0.0 w_ictr=0.2 sigma=0.05  warmup=500
  e_target match-acc(train batch) nan
/home/cymaxwelllee/Projects/lacot/.venv/lib/python3.11/site-packages/torch/nn/modules/transformer.py:385: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.norm_first was True
  warnings.warn(
stage 2 flow+refine+action ...
  ⭐ 已切到 EMA 影子權重（m=0.999，五模組）
✅ 載入 ckpt_large-stitch_self_K8_c256_ch4_st8000_T128_ep2_gu_eorecon_ictr_tch0.5_emw0.999_wu500_dssoft_norf_cd0.1_bci_s33.pt（跳過訓練，只跑評估）  cfg={'K': 8, 'T_CAP': 128, 'ENC_OBJ': 'recon_ictr', 'LEARNED_REFINE': 0, 'COND_DROP': 0.1}
  decoder 內部點 RMSE 0.0981   打亂 u 之後 1.8960   用到的 u 值 +1.7979   ✓
PROTOCOL_EQUIVALENCE_PASS: 3 x 160 real maze steps, frozen oracle A, actions + XY byte-identical
REAL_CPU_PASS: M9 import, s33/EMA, encode_u false mask, head, decode, four P/Q same-u reruns, fresh-flow seed, legal-w short cap; GPU evidence=false
{"ckpt_sha256": "88180676e1f8d8df22c47bf4eeaf23eba6a73c5e92cacbf522ce3044095af787", "encoded_u_sha256": "fd89a90c7385dfd5c57288d7a2511b188fcb03c867cdf5e23bc0ab72ebb961a9", "encoder_shape": [1, 8, 256], "head_shape": [1, 4, 2], "head_u_hashes": ["fd89a90c7385dfd5c57288d7a2511b188fcb03c867cdf5e23bc0ab72ebb961a9"], "mask_all_false": true, "production_evidence": false, "source_sha256": "276c68fba6c41a05c1e3c2bd5bf4b25c39642d0dbac3ca7aa57cce381056ecfc"}
```

### fourth

命令：`CUDA_VISIBLE_DEVICES='' CUBLAS_WORKSPACE_CONFIG=:4096:8 MUJOCO_GL=egl .venv/bin/python -B -m unittest discover -s experiments/_workorders/breakthrough-probe -p fourth_selftest.py -v`

exit code：0；raw SHA256：`f634e812e311acb11c4b95f0f1084c39bbaf674d1f22dc2cc5c9234f77b88ead`

```text
test_30_freeze_flags_and_fixed_hash_inventory (fourth_selftest.FourthTests.test_30_freeze_flags_and_fixed_hash_inventory) ... ok
test_31_move1_positive_difference_and_conformance_downgrade (fourth_selftest.FourthTests.test_31_move1_positive_difference_and_conformance_downgrade) ... ok
test_32_reproduction_and_gate_threshold_four_five (fourth_selftest.FourthTests.test_32_reproduction_and_gate_threshold_four_five) ... ok
test_33_harvest_hostile_evidence (fourth_selftest.FourthTests.test_33_harvest_hostile_evidence) ... ok
test_34_historical_A_B_fingerprints (fourth_selftest.FourthTests.test_34_historical_A_B_fingerprints) ... ok
test_35_new_card_and_rule_boundaries (fourth_selftest.FourthTests.test_35_new_card_and_rule_boundaries) ... ok
test_36_combined_exits_from_full_artifacts (fourth_selftest.FourthTests.test_36_combined_exits_from_full_artifacts) ... ok
test_37_stub_fuzz_runloop_noise_switch_target_stuck (fourth_selftest.FourthTests.test_37_stub_fuzz_runloop_noise_switch_target_stuck) ... ok
test_38_runtime_coordinator_sources_gates_shards_heartbeat (fourth_selftest.FourthTests.test_38_runtime_coordinator_sources_gates_shards_heartbeat) ... /home/cymaxwelllee/Projects/lacot/experiments/_workorders/breakthrough-probe/runtime.py:300: DeprecationWarning: __array_wrap__ must accept context and return_scalar arguments (positionally) in the future. (Deprecated NumPy 2.0)
  points = (points * module.SD_XY_T + module.MU_XY_T).cpu().numpy()
PROGRESS move=1 rollouts=50/108 task=4 episode=20
PROGRESS move=1 rollouts=100/108 task=4 episode=4
PROGRESS move=1 rollouts=50/108 task=4 episode=18
PROGRESS move=1 rollouts=100/108 task=5 episode=32
ok
test_39_raw_survives_validation_failure (fourth_selftest.FourthTests.test_39_raw_survives_validation_failure) ... ok
test_40_calibration_fail_fast_and_estimators (fourth_selftest.FourthTests.test_40_calibration_fail_fast_and_estimators) ... ok
test_41_remaining_boundary_and_preprocessing_mutants (fourth_selftest.FourthTests.test_41_remaining_boundary_and_preprocessing_mutants) ... ok
test_42_checkpoint_resume_without_rerunning_completed_rows (fourth_selftest.FourthTests.test_42_checkpoint_resume_without_rerunning_completed_rows) ... PROGRESS move=1 rollouts=50/108 task=4 episode=18
PROGRESS move=1 rollouts=100/108 task=5 episode=32
PROGRESS move=1 rollouts=50/108 task=4 episode=18
PROGRESS move=1 rollouts=100/108 task=5 episode=32
ok
test_43_move3_discordance_both_directions_from_traces (fourth_selftest.FourthTests.test_43_move3_discordance_both_directions_from_traces) ... ok

----------------------------------------------------------------------
Ran 14 tests in 7.179s

OK
```

STATUS: DONE

## 第四棒驗收與兩態證據
detour 48/48（31 舊＋17 新）、FAIL0／ERROR0／SKIP0；舊 selftest 44/44、fourth discovery 14/14（已含於44，不能再加成58），runtime 的 REAL_CPU_PASS 與 PROTOCOL_EQUIVALENCE_PASS 均通過。四個入口使用 `CUDA_VISIBLE_DEVICES='' CUBLAS_WORKSPACE_CONFIG=:4096:8 MUJOCO_GL=egl`；各 process exit code=0。PB 既有79檔逐一比第三棒 evidence 與本棒開始快照，全同。
- N1：test_32 釘計畫勝／同向例外／皆否／第一子句與三個零向量；pg=0.7071067811865475、0.7071067811865476 在 cp≥p10 但 cp≤cg 時皆 FAIL。只把第二子句換 True，新測試3個 subtest 斷言紅；還原即綠。test_33 用真 M9／s33 EMA／真 maze 經完整 build_e1_table 產 CPU 記憶體測試表，只在測試 patch GPU／hostname 入口閘；共同靜態205格0翻、最大 Δcos=7.2373521398105822e-8。cos 比對上限1e-6只用於回歸測試，资格仍逐格嚴格相等，沒有修改閘值或收割 tolerance。完整 CPU 表存於 evidence 的 tests.real_cpu_e1.fixture；不是正式 E1。
- N2：test_34–46 正確列先收、竄改後用 assertRaisesRegex 逐項拒；資格竄改一律修好 failed_qualification_chunks。覆蓋方向資格重算、D、靜態 u／τ／資格、F3／O3 尾段 runtime 參照、零長度規則與資格繼承、同題 reset、早停 horizon、success 末步與 w 欄。test_37 的 τ 列竄改先被 runtime/static tau hash 擋；另竄改凍結表 τ 以獨立驗 E0 frozen table tau_sha256（正確幾何 τ 無法同時與正確表不同）。兩道各有 mutant 紅燈；沒有繞過正常幾何收割的正例。
- N2(j)：test_47 逐一對 O3 static／tail／goal 的實際 encode_inputs 末點與 target_xy 做斷言，也核對 raw τ 末點／hash；只統計本 row 模型的 encode 呼叫，排除 collector 相容性檢查的另外6次編碼。mutant 讓 encode_u 真收到錯末點，再保持 mock u 輸出不變；收割與 hash 仍通過，唯一新末點斷言紅，證明不是旁支擋下。
- N7：killer 不再有 Exception 預設，所有既有呼叫改為明示例外型別與訊息；本檔全部 assertRaises 都改為 assertRaisesRegex。test_48 驗無關 RuntimeError 不會吞掉、錯誤拒收訊息不會算 PASS；退回 Exception 或不驗訊息各自紅。舊 target／F3-final-g 殺手同步修改 flow_calls，避免先被 flow-call 一致性擋下；量化殺手正確訊息是 provider differs from frozen E1 table。
預檢曾因舊殺手訊息預期不準、模型重複初始化 torch interop threads、編碼觀測混入相容性檢查呼叫而紅；只修測試資料／預期／CPU 模型一次載入重用。預檢 raw 留於 evidence，不將這些紅燈當成產品缺陷或最终 PASS。
所有 mutant 只在記憶體 patch；五支受保護檔從未落盤改壞。19個 mutant，覆蓋17條新測試；每項均先原版綠、改壞斷言紅（ERROR0）、解除 patch 再綠。完整 runner source、逐案三態 raw／SHA 存於 evidence.tests.mutation_audit；可將 runner_source 還原到 /tmp，以 `.venv/bin/python -B /tmp/detour_b4_mutations.py` 在 repo 根目錄重現。下表的 RED 是被測 unittest 的結果碼1；整個 audit process 因成功殺掉 mutant 而 exit0。
| 新測試 | mutant | 原版 | 改壞（斷言 FAIL／ERROR） | 還原 | 紅燈訊息釘子 |
|---|---|---|---|---|---|
| test_32 | `N1-second-clause=True` | GREEN | RED 3/0 | GREEN | `neither` |
| test_33 | `N1-CPU-table-one-static-flip` | GREEN | RED 1/0 | GREEN | `CPU common-static qualification flips` |
| test_34 | `N2-no-runtime-qualification-recalculation` | GREEN | RED 1/0 | GREEN | `Rejected not raised` |
| test_35 | `N2-no-direction-D-recalculation` | GREEN | RED 1/0 | GREEN | `Rejected not raised` |
| test_36 | `N2-no-static-u-check` | GREEN | RED 1/0 | GREEN | `Rejected not raised` |
| test_37 | `N2-no-tau-geometry-check` | GREEN | RED 1/0 | GREEN | `does not match` |
| test_37 | `N2-no-frozen-tau-check` | GREEN | RED 1/0 | GREEN | `Rejected not raised` |
| test_38 | `N2-no-static-qualification-check` | GREEN | RED 1/0 | GREEN | `Rejected not raised` |
| test_39 | `N2-no-F3-runtime-reference` | GREEN | RED 1/0 | GREEN | `Rejected not raised` |
| test_40 | `N2-no-O3-tail-runtime-reference` | GREEN | RED 1/0 | GREEN | `Rejected not raised` |
| test_41 | `N2-no-zero-length-rule` | GREEN | RED 1/0 | GREEN | `Rejected not raised` |
| test_42 | `N2-no-zero-length-qualification-inheritance` | GREEN | RED 1/0 | GREEN | `Rejected not raised` |
| test_43 | `N2-no-paired-reset-check` | GREEN | RED 1/0 | GREEN | `Rejected not raised` |
| test_44 | `N2-no-early-stop-reason-check` | GREEN | RED 1/0 | GREEN | `Rejected not raised` |
| test_45 | `N2-no-success-timing-check` | GREEN | RED 1/0 | GREEN | `Rejected not raised` |
| test_46 | `N2-no-w-field-check` | GREEN | RED 1/0 | GREEN | `Rejected not raised` |
| test_47 | `N2-actual-encode-input-wrong-endpoint` | GREEN | RED 1/0 | GREEN | `O3 captured encode_u endpoint = target` |
| test_48 | `N7-killer-catches-Exception` | GREEN | RED 1/0 | GREEN | `RuntimeError not raised` |
| test_48 | `N7-killer-does-not-check-message` | GREEN | RED 1/0 | GREEN | `AssertionError not raised` |

### 第四棒新舊 SHA

五支受保護檔修前＝修後＝第三棒（逐 byte）。測試／README 改動如下；notes 與 evidence 不自參照 hash，evidence 本體由同名 .sha256 驗證。

| 檔 | 本棒修前 SHA256 | 本棒修後 SHA256 |
|---|---|---|
| detour_selftest.py | 8ee52f7eab0fcc5ec44e100a0b78349870350e1161bf93f4ba6a6355c8fbc349 | 7e4fa888b5746a00569eb45b10144688dd5c3a8fed5669f6601548701b6d3d73 |
| README-detour.md | 47abee6cf90848e1a2c20abc70ff459885675da913f2f0e69f4c16294256938d | 0676cb22991a701e285c12b7441612bc10b41fde4b172cd639e94718e67883ba |
| harness_detour.py | b6da954eea5b4982644c5613cbf3b24b6d6c35ae919b2be582acb804b9a28514 | b6da954eea5b4982644c5613cbf3b24b6d6c35ae919b2be582acb804b9a28514 |
| harvest_detour.py | ad2d50b5ea34c255844457314aadcc3f359bdca2b25433f594c618c2d51ec1fc | ad2d50b5ea34c255844457314aadcc3f359bdca2b25433f594c618c2d51ec1fc |
| detour_plan.py | 81f404af2ab19d0ef4167e19312efa845208a11d4c2ce26d840a4e1bef58a176 | 81f404af2ab19d0ef4167e19312efa845208a11d4c2ce26d840a4e1bef58a176 |
| detour-smoke.sbatch | a39eba0b8d44914f99f776af525d6527ff85635183835fb5c3ece43731591fa9 | a39eba0b8d44914f99f776af525d6527ff85635183835fb5c3ece43731591fa9 |
| detour-formal.sbatch | 004cb996c43b51b0931200fdbafdb735cf05ac9c92f9df9d1fd3345f78234e67 | 004cb996c43b51b0931200fdbafdb735cf05ac9c92f9df9d1fd3345f78234e67 |

### 第四棒限制與 smoke 前程序

GPU 解碼／全長救回率、跨 process GPU u hash、真 Slurm hostname／logs／時限、實際 smoke／formal 全部未驗。CPU 是 REAL_CPU 測試證據，不能替代上述項目；無 BLOCKED 測試。README 已補失敗 smoke 先改名保留再重送；smoke 前 SMOKE_ENABLED=true、PRODUCTION_READY=false，smoke 與 E1 驗過才翻 PRODUCTION_READY；ETA 按每步秒數×1000×支數估，proposed_time_limit_seconds 不保證全長時限。本棒 flags 本體維持雙 false。

### 第四棒原始驗證輸出

全文留痕計數為89條／87種，含唯一 release-False；只解析本節 detour raw，不混入歷史 raw 或 mutation audit。

#### 第四棒 detour

命令：`CUDA_VISIBLE_DEVICES='' CUBLAS_WORKSPACE_CONFIG=:4096:8 MUJOCO_GL=egl .venv/bin/python -B experiments/_workorders/breakthrough-probe/detour_selftest.py`

exit code：0；raw SHA256：`448b80ec13260bbb652f01f4da6072487494e86f3f54ce7235a97f47abf2fe65`

```text
test_01_pins_flags_seeds (__main__.DetourTests.test_01_pins_flags_seeds) ... ok
test_02_plan_machine_only_and_inventory (__main__.DetourTests.test_02_plan_machine_only_and_inventory) ... ok
test_03_table_all_cells_and_roundtrip (__main__.DetourTests.test_03_table_all_cells_and_roundtrip) ... ok
test_04_exact_segment_square (__main__.DetourTests.test_04_exact_segment_square) ... ok
test_05_all_checker_killers_and_boundaries (__main__.DetourTests.test_05_all_checker_killers_and_boundaries) ... ok
test_06_sampled_clearance_mutant (__main__.DetourTests.test_06_sampled_clearance_mutant) ... ok
test_07_unordered_frechet_mutant (__main__.DetourTests.test_07_unordered_frechet_mutant) ... ok
test_08_provider_head_and_noise_forward (__main__.DetourTests.test_08_provider_head_and_noise_forward) ... ok
test_09_goal_zero_wall_script (__main__.DetourTests.test_09_goal_zero_wall_script) ... ok
test_10_zero_inherits_failure_and_no_previous (__main__.DetourTests.test_10_zero_inherits_failure_and_no_previous) ... ok
test_11_provider_flow_mutant (__main__.DetourTests.test_11_provider_flow_mutant) ... ok
test_12_noise_removed_mutant (__main__.DetourTests.test_12_noise_removed_mutant) ... ok
test_13_raw_cache_mutant (__main__.DetourTests.test_13_raw_cache_mutant) ... ok
test_14_targets_flow_and_quantization (__main__.DetourTests.test_14_targets_flow_and_quantization) ... ok
test_15_reject_artifact_matrix (__main__.DetourTests.test_15_reject_artifact_matrix) ... ok
test_16_receipt_and_exclusive_shards (__main__.DetourTests.test_16_receipt_and_exclusive_shards) ... ok
test_17_question_layers_thresholds (__main__.DetourTests.test_17_question_layers_thresholds) ... ok
test_18_failure_partition_norm_and_compliance (__main__.DetourTests.test_18_failure_partition_norm_and_compliance) ... ok
test_19_exact_verdicts_and_exclusivity (__main__.DetourTests.test_19_exact_verdicts_and_exclusivity) ... ok
test_20_full_horizon_fifteen_mock_rollouts (__main__.DetourTests.test_20_full_horizon_fifteen_mock_rollouts) ... ok
test_21_original_factory_rejects_and_detour_passes (__main__.DetourTests.test_21_original_factory_rejects_and_detour_passes) ... ok
test_22_coordinator_freeze_fifteen_shards_resume (__main__.DetourTests.test_22_coordinator_freeze_fifteen_shards_resume) ... TWO_STATE release-False: correct=PASS; killer/mutant=FAIL (observed rejection)
TWO_STATE release-True: correct=PASS; killer/mutant=FAIL (observed rejection)
TWO_STATE plan-inventory: correct=PASS; killer/mutant=FAIL (observed rejection)
TWO_STATE mock-production-reject: correct=PASS; killer/mutant=FAIL (observed rejection)
TWO_STATE tail-in-static-table: correct=PASS; killer/mutant=FAIL (observed rejection)
TWO_STATE E1-final-g: true-tau=PASS; all departure-cell final-g straight lines=FAIL
TWO_STATE E1-reverse: true-tau=PASS; all departure-cell reversed first directions=FAIL
TWO_STATE E1-boundary: p10+1e-7=PASS; p10-1e-7=FAIL
TWO_STATE E1-zero: nonzero=PASS; D/P/G zero=FAIL
TWO_STATE E1-always-pass: correct=PASS; killer/mutant=FAIL (observed rejection)
TWO_STATE zero-no-previous: correct=PASS; killer/mutant=FAIL (observed rejection)
TWO_STATE provider-flow-E0: correct=PASS; killer/mutant=FAIL (observed rejection)
TWO_STATE noise-removed: correct=PASS; killer/mutant=FAIL (observed rejection)
TWO_STATE flow-seed-conditional-validator: correct=PASS; killer/mutant=FAIL (observed rejection)
TWO_STATE M3-c_raw-cache: correct=PASS; killer/mutant=FAIL (observed rejection)
TWO_STATE target-O3: correct=PASS; killer/mutant=FAIL (observed rejection)
TWO_STATE unknown-protocol-harvest-O3: correct=PASS; killer/mutant=FAIL (observed rejection)
TWO_STATE unknown-protocol-worker-O3: correct=PASS; killer/mutant=FAIL (observed rejection)
TWO_STATE target-F3: correct=PASS; killer/mutant=FAIL (observed rejection)
TWO_STATE unknown-protocol-harvest-F3: correct=PASS; killer/mutant=FAIL (observed rejection)
TWO_STATE unknown-protocol-worker-F3: correct=PASS; killer/mutant=FAIL (observed rejection)
TWO_STATE target-OF: correct=PASS; killer/mutant=FAIL (observed rejection)
TWO_STATE unknown-protocol-harvest-OF: correct=PASS; killer/mutant=FAIL (observed rejection)
TWO_STATE unknown-protocol-worker-OF: correct=PASS; killer/mutant=FAIL (observed rejection)
TWO_STATE target-Q-C: correct=PASS; killer/mutant=FAIL (observed rejection)
TWO_STATE unknown-protocol-harvest-Q-C: correct=PASS; killer/mutant=FAIL (observed rejection)
TWO_STATE unknown-protocol-worker-Q-C: correct=PASS; killer/mutant=FAIL (observed rejection)
TWO_STATE target-Q-O3: correct=PASS; killer/mutant=FAIL (observed rejection)
TWO_STATE unknown-protocol-harvest-Q-O3: correct=PASS; killer/mutant=FAIL (observed rejection)
TWO_STATE unknown-protocol-worker-Q-O3: correct=PASS; killer/mutant=FAIL (observed rejection)
TWO_STATE target-Q-F3: correct=PASS; killer/mutant=FAIL (observed rejection)
TWO_STATE unknown-protocol-harvest-Q-F3: correct=PASS; killer/mutant=FAIL (observed rejection)
TWO_STATE unknown-protocol-worker-Q-F3: correct=PASS; killer/mutant=FAIL (observed rejection)
TWO_STATE F3-final-g-target: correct=PASS; killer/mutant=FAIL (observed rejection)
TWO_STATE F3-worker-final-g-condition: correct=PASS; killer/mutant=FAIL (observed rejection)
TWO_STATE F3-call-count: correct=PASS; killer/mutant=FAIL (observed rejection)
TWO_STATE F3-fake-sampling: correct=PASS; killer/mutant=FAIL (observed rejection)
TWO_STATE O3-encoding-endpoint: correct=PASS; killer/mutant=FAIL (observed rejection)
TWO_STATE head-must-equal-frozen-quantization: correct=PASS; killer/mutant=FAIL (observed rejection)
TWO_STATE unknown-arm: correct=PASS; killer/mutant=FAIL (observed rejection)
TWO_STATE missing: correct=PASS; killer/mutant=FAIL (observed rejection)
TWO_STATE duplicate: correct=PASS; killer/mutant=FAIL (observed rejection)
TWO_STATE seed: correct=PASS; killer/mutant=FAIL (observed rejection)
TWO_STATE draw: correct=PASS; killer/mutant=FAIL (observed rejection)
TWO_STATE u-self-certify: correct=PASS; killer/mutant=FAIL (observed rejection)
TWO_STATE missing-qualification: correct=PASS; killer/mutant=FAIL (observed rejection)
TWO_STATE noise-state: correct=PASS; killer/mutant=FAIL (observed rejection)
TWO_STATE missing-chunk: correct=PASS; killer/mutant=FAIL (observed rejection)
TWO_STATE false-success: correct=PASS; killer/mutant=FAIL (observed rejection)
TWO_STATE early-stop: correct=PASS; killer/mutant=FAIL (observed rejection)
TWO_STATE missing-action: correct=PASS; killer/mutant=FAIL (observed rejection)
TWO_STATE receipt-card_sha256: correct=PASS; killer/mutant=FAIL (observed rejection)
TWO_STATE receipt-code_sha256: correct=PASS; killer/mutant=FAIL (observed rejection)
TWO_STATE receipt-collector_sha256: correct=PASS; killer/mutant=FAIL (observed rejection)
TWO_STATE receipt-e1_sha256: correct=PASS; killer/mutant=FAIL (observed rejection)
TWO_STATE receipt-plan_sha256: correct=PASS; killer/mutant=FAIL (observed rejection)
TWO_STATE exclusive-write: correct=PASS; killer/mutant=FAIL (observed rejection)
TWO_STATE shard-tamper: correct=PASS; killer/mutant=FAIL (observed rejection)
TWO_STATE exact-R3-wording: correct=PASS; killer/mutant=FAIL (observed rejection)
MOCK_FIFTEEN_PASS: 15 x H=1000; real detour collector factory, no factory patch; NOT production smoke
FACTORY_RESOLVED: original draw80 rejects; copied factory draw80..83 and real C path PASS
detour rollouts=1/15
detour rollouts=2/15
detour rollouts=3/15
detour rollouts=4/15
detour rollouts=5/15
detour rollouts=6/15
detour rollouts=7/15
detour rollouts=8/15
detour rollouts=9/15
detour rollouts=10/15
detour rollouts=11/15
detour rollouts=12/15
detour rollouts=13/15
detour rollouts=14/15
detour rollouts=15/15
detour rollouts=1/15
detour rollouts=2/15
detour rollouts=3/15
detour rollouts=4/15
detour rollouts=5/15
detour rollouts=6/15
detour rollouts=7/15
detour rollouts=8/15
detour rollouts=9/15
detour rollouts=10/15
detour rollouts=11/15
detour rollouts=12/15
detour rollouts=13/15
detour rollouts=14/15
detour rollouts=15/15
ok
test_23_partial_failure_resume_and_fail_closed_entry (__main__.DetourTests.test_23_partial_failure_resume_and_fail_closed_entry) ... MOCK_COORDINATOR_PASS: E1 exclusive freeze -> 15 shards -> independent harvest -> resume without rerun; gates patched in test only
detour rollouts=1/15
detour rollouts=2/15
detour rollouts=1/15
detour rollouts=2/15
detour rollouts=3/15
detour rollouts=4/15
detour rollouts=5/15
detour rollouts=6/15
detour rollouts=7/15
detour rollouts=8/15
detour rollouts=9/15
detour rollouts=10/15
detour rollouts=11/15
detour rollouts=12/15
detour rollouts=13/15
detour rollouts=14/15
detour rollouts=15/15
ok
test_24_formal_readout_denominators_and_q_pairs (__main__.DetourTests.test_24_formal_readout_denominators_and_q_pairs) ... ok
test_25_factory_byte_equivalence_and_line_copy (__main__.DetourTests.test_25_factory_byte_equivalence_and_line_copy) ... ok
test_26_factory_equivalence_killers (__main__.DetourTests.test_26_factory_equivalence_killers) ... ok
test_27_three_arms_noise_bytes_and_factory (__main__.DetourTests.test_27_three_arms_noise_bytes_and_factory) ... ok
test_28_cpu_real_short_wiring (__main__.DetourTests.test_28_cpu_real_short_wiring)
Real M9/ckpt + real maze, in memory only; short wiring, never a score/E1 artifact. ... FACTORY_BYTE_EQUIVALENCE_PASS: 2 task/episode pairs x draws 64,71,79; stream/flow, noise bytes, global torch RNG bytes
TWO_STATE factory-arm-B: correct=PASS; killer/mutant=FAIL (observed rejection)
TWO_STATE factory-sigma: correct=PASS; killer/mutant=FAIL (observed rejection)
TWO_STATE factory-noise-state: correct=PASS; killer/mutant=FAIL (observed rejection)
THREE_ARM_NOISE_BYTES_PASS: O3/OF/Q-C same task4 episode4 draw80; 40 per-step samples; flow calls 0/0/10
device: cpu
  teacher 題庫：4096 條（細格步數 p50 26 p90 42 max 62），mix=0.5
設定：seed=33 cons=self ema_m=0.996 K=8 COND=256
stage 1 e_target 目標=recon_ictr ...  w_var=0.0 w_cov=0.0 w_ictr=0.2 sigma=0.05  warmup=500
  e_target match-acc(train batch) nan
/home/cymaxwelllee/Projects/lacot/.venv/lib/python3.11/site-packages/torch/nn/modules/transformer.py:385: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.norm_first was True
  warnings.warn(
stage 2 flow+refine+action ...
  ⭐ 已切到 EMA 影子權重（m=0.999，五模組）
✅ 載入 ckpt_large-stitch_self_K8_c256_ch4_st8000_T128_ep2_gu_eorecon_ictr_tch0.5_emw0.999_wu500_dssoft_norf_cd0.1_bci_s33.pt（跳過訓練，只跑評估）  cfg={'K': 8, 'T_CAP': 128, 'ENC_OBJ': 'recon_ictr', 'LEARNED_REFINE': 0, 'COND_DROP': 0.1}
  decoder 內部點 RMSE 0.0981   打亂 u 之後 1.8960   用到的 u 值 +1.7979   ✓
ok
test_29_float64_full_harvest (__main__.DetourTests.test_29_float64_full_harvest) ... ok
test_30_tail_runtime_two_goals_and_draws (__main__.DetourTests.test_30_tail_runtime_two_goals_and_draws) ... ok
test_31_ruler_threshold_is_live_and_pinned (__main__.DetourTests.test_31_ruler_threshold_is_live_and_pinned) ... ok
test_32_direction_second_clause_truth_table (__main__.DetourTests.test_32_direction_second_clause_truth_table) ... ok
test_33_real_cpu_e1_common_static_zero_flips (__main__.DetourTests.test_33_real_cpu_e1_common_static_zero_flips)
Real frozen model; only the GPU/hostname entry guards are patched for this CPU fixture. ... ok
test_34_runtime_qualification_recomputed_not_counted (__main__.DetourTests.test_34_runtime_qualification_recomputed_not_counted) ... ok
test_35_direction_D_recomputed (__main__.DetourTests.test_35_direction_D_recomputed) ... ok
test_36_static_u_frozen_authority (__main__.DetourTests.test_36_static_u_frozen_authority) ... ok
test_37_static_tau_geometry_and_frozen_authority (__main__.DetourTests.test_37_static_tau_geometry_and_frozen_authority) ... ok
test_38_static_qualification_frozen_authority (__main__.DetourTests.test_38_static_qualification_frozen_authority) ... ok
test_39_F3_tail_runtime_reference (__main__.DetourTests.test_39_F3_tail_runtime_reference) ... ok
test_40_O3_tail_runtime_reference (__main__.DetourTests.test_40_O3_tail_runtime_reference) ... ok
test_41_zero_length_rule (__main__.DetourTests.test_41_zero_length_rule) ... ok
test_42_zero_length_keeps_qualification (__main__.DetourTests.test_42_zero_length_keeps_qualification) ... ok
test_43_same_question_reset_fingerprint (__main__.DetourTests.test_43_same_question_reset_fingerprint) ... ok
test_44_early_stop_cannot_claim_horizon (__main__.DetourTests.test_44_early_stop_cannot_claim_horizon) ... ok
test_45_success_only_last_step (__main__.DetourTests.test_45_success_only_last_step) ... ok
test_46_w_field_independently_checked (__main__.DetourTests.test_46_w_field_independently_checked) ... ok
test_47_O3_actual_encoder_endpoint_is_target (__main__.DetourTests.test_47_O3_actual_encoder_endpoint_is_target) ... ok
test_48_killer_requires_expected_exception_and_message (__main__.DetourTests.test_48_killer_requires_expected_exception_and_message) ... ok

----------------------------------------------------------------------
Ran 48 tests in 58.903s

OK
DETOUR_REAL_CPU_PASS: M9+s33/EMA, real maze, Q-C task1 episode10 and O3/F3/OF task4 episode4 draw80 H=40; full validate_row PASS, flow=10/0/10/0, noise bytes paired, delivered u hashes and qualification fields checked; score=false, GPU=false, archive_writes=0
TWO_STATE float32-trace-O3: correct=PASS; killer/mutant=FAIL (observed rejection)
TWO_STATE goal-tau-dtype-O3: correct=PASS; killer/mutant=FAIL (observed rejection)
TWO_STATE float32-trace-F3: correct=PASS; killer/mutant=FAIL (observed rejection)
TWO_STATE float32-trace-OF: correct=PASS; killer/mutant=FAIL (observed rejection)
TWO_STATE goal-tau-dtype-OF: correct=PASS; killer/mutant=FAIL (observed rejection)
TWO_STATE float32-trace-Q-C: correct=PASS; killer/mutant=FAIL (observed rejection)
TWO_STATE tail-static-hash: correct=PASS; killer/mutant=FAIL (observed rejection)
TWO_STATE tail-runtime-qualification: correct=PASS; killer/mutant=FAIL (observed rejection)
TWO_STATE tail-static-hash: correct=PASS; killer/mutant=FAIL (observed rejection)
TWO_STATE tail-runtime-qualification: correct=PASS; killer/mutant=FAIL (observed rejection)
TWO_STATE tail-goal-u: separate draws distinct runtime hashes=PASS; stale first-draw hash=FAIL
TWO_STATE rounded-hardcoded-p10: correct=PASS; killer/mutant=FAIL (observed rejection)
TWO_STATE N1-second-clause: plan-wins/same-direction=PASS; neither/cos45-475/cos45-476/zeros=FAIL
REAL_CPU_E1_COMMON_STATIC_PASS: compared=205 flips=0 max_delta_cos=7.2373521398105822e-08; production_evidence=false GPU=false archive_writes=0
TWO_STATE N2-runtime-qualification-count-repaired: correct=PASS; killer/mutant=FAIL (observed rejection)
TWO_STATE N2-direction-D: correct=PASS; killer/mutant=FAIL (observed rejection)
TWO_STATE N2-static-u: correct=PASS; killer/mutant=FAIL (observed rejection)
TWO_STATE N2-static-tau-geometry: correct=PASS; killer/mutant=FAIL (observed rejection)
TWO_STATE N2-static-tau-frozen: correct=PASS; killer/mutant=FAIL (observed rejection)
TWO_STATE N2-static-qualification-count-repaired: correct=PASS; killer/mutant=FAIL (observed rejection)
TWO_STATE N2-runtime-reference-F3: correct=PASS; killer/mutant=FAIL (observed rejection)
TWO_STATE N2-runtime-reference-O3: correct=PASS; killer/mutant=FAIL (observed rejection)
TWO_STATE N2-zero-length-rule: correct=PASS; killer/mutant=FAIL (observed rejection)
TWO_STATE N2-zero-length-inherited-qualification: correct=PASS; killer/mutant=FAIL (observed rejection)
TWO_STATE N2-unpaired-reset: correct=PASS; killer/mutant=FAIL (observed rejection)
TWO_STATE N2-early-horizon: correct=PASS; killer/mutant=FAIL (observed rejection)
TWO_STATE N2-success-before-last: correct=PASS; killer/mutant=FAIL (observed rejection)
TWO_STATE N2-w-field: correct=PASS; killer/mutant=FAIL (observed rejection)
```

#### 第四棒 selftest

命令：`CUDA_VISIBLE_DEVICES='' CUBLAS_WORKSPACE_CONFIG=:4096:8 MUJOCO_GL=egl .venv/bin/python -B experiments/_workorders/breakthrough-probe/selftest.py`

exit code：0；raw SHA256：`1361401141ee04ef5f291e0b353dc60ff8a4d3fdbe0c7183234609f915c9397f`

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
PROTOCOL_EQUIVALENCE_PASS: 3 x 160 real maze steps, frozen oracle A, actions + XY byte-identical
REAL_CPU_PASS: M9 import, s33/EMA, encode_u false mask, head, decode, four P/Q same-u reruns, fresh-flow seed, legal-w short cap; GPU evidence=false
{"ckpt_sha256": "88180676e1f8d8df22c47bf4eeaf23eba6a73c5e92cacbf522ce3044095af787", "encoded_u_sha256": "fd89a90c7385dfd5c57288d7a2511b188fcb03c867cdf5e23bc0ab72ebb961a9", "encoder_shape": [1, 8, 256], "head_shape": [1, 4, 2], "head_u_hashes": ["fd89a90c7385dfd5c57288d7a2511b188fcb03c867cdf5e23bc0ab72ebb961a9"], "mask_all_false": true, "production_evidence": false, "source_sha256": "276c68fba6c41a05c1e3c2bd5bf4b25c39642d0dbac3ca7aa57cce381056ecfc"}
ok
test_24_move1_108_limit_and_conditional_rows (__main__.ContractTests.test_24_move1_108_limit_and_conditional_rows) ... ok
test_25_short_trace_contract (__main__.ContractTests.test_25_short_trace_contract) ... ok
test_26_d4_exclusive_crossing_symmetric (__main__.ContractTests.test_26_d4_exclusive_crossing_symmetric) ... ok
test_27_m9_pin_and_import_boundary (__main__.ContractTests.test_27_m9_pin_and_import_boundary) ... ok
test_28_full_artifact_conditional_coverage_and_donors (__main__.ContractTests.test_28_full_artifact_conditional_coverage_and_donors) ... ok
test_29_generated_contract_snapshots (__main__.ContractTests.test_29_generated_contract_snapshots) ... ok
test_30_freeze_flags_and_fixed_hash_inventory (fourth_selftest.FourthTests.test_30_freeze_flags_and_fixed_hash_inventory) ... ok
test_31_move1_positive_difference_and_conformance_downgrade (fourth_selftest.FourthTests.test_31_move1_positive_difference_and_conformance_downgrade) ... ok
test_32_reproduction_and_gate_threshold_four_five (fourth_selftest.FourthTests.test_32_reproduction_and_gate_threshold_four_five) ... ok
test_33_harvest_hostile_evidence (fourth_selftest.FourthTests.test_33_harvest_hostile_evidence) ... ok
test_34_historical_A_B_fingerprints (fourth_selftest.FourthTests.test_34_historical_A_B_fingerprints) ... ok
test_35_new_card_and_rule_boundaries (fourth_selftest.FourthTests.test_35_new_card_and_rule_boundaries) ... ok
test_36_combined_exits_from_full_artifacts (fourth_selftest.FourthTests.test_36_combined_exits_from_full_artifacts) ... ok
test_37_stub_fuzz_runloop_noise_switch_target_stuck (fourth_selftest.FourthTests.test_37_stub_fuzz_runloop_noise_switch_target_stuck) ... ok
test_38_runtime_coordinator_sources_gates_shards_heartbeat (fourth_selftest.FourthTests.test_38_runtime_coordinator_sources_gates_shards_heartbeat) ... PROGRESS move=1 rollouts=50/108 task=4 episode=20
PROGRESS move=1 rollouts=100/108 task=4 episode=4
PROGRESS move=1 rollouts=50/108 task=4 episode=18
PROGRESS move=1 rollouts=100/108 task=5 episode=32
ok
test_39_raw_survives_validation_failure (fourth_selftest.FourthTests.test_39_raw_survives_validation_failure) ... ok
test_40_calibration_fail_fast_and_estimators (fourth_selftest.FourthTests.test_40_calibration_fail_fast_and_estimators) ... ok
test_41_remaining_boundary_and_preprocessing_mutants (fourth_selftest.FourthTests.test_41_remaining_boundary_and_preprocessing_mutants) ... ok
test_42_checkpoint_resume_without_rerunning_completed_rows (fourth_selftest.FourthTests.test_42_checkpoint_resume_without_rerunning_completed_rows) ... PROGRESS move=1 rollouts=50/108 task=4 episode=18
PROGRESS move=1 rollouts=100/108 task=5 episode=32
PROGRESS move=1 rollouts=50/108 task=4 episode=18
PROGRESS move=1 rollouts=100/108 task=5 episode=32
ok
test_43_move3_discordance_both_directions_from_traces (fourth_selftest.FourthTests.test_43_move3_discordance_both_directions_from_traces) ... ok

----------------------------------------------------------------------
Ran 44 tests in 16.513s

OK
STATUS: DONE
```

#### 第四棒 runtime

命令：`CUDA_VISIBLE_DEVICES='' CUBLAS_WORKSPACE_CONFIG=:4096:8 MUJOCO_GL=egl .venv/bin/python -B experiments/_workorders/breakthrough-probe/runtime_selftest.py`

exit code：0；raw SHA256：`8eb7b26d5c5058cc61769e8123cd70937209924d5484f9548e043ca12bbb4717`

```text
device: cpu
  teacher 題庫：4096 條（細格步數 p50 26 p90 42 max 62），mix=0.5
設定：seed=33 cons=self ema_m=0.996 K=8 COND=256
stage 1 e_target 目標=recon_ictr ...  w_var=0.0 w_cov=0.0 w_ictr=0.2 sigma=0.05  warmup=500
  e_target match-acc(train batch) nan
/home/cymaxwelllee/Projects/lacot/.venv/lib/python3.11/site-packages/torch/nn/modules/transformer.py:385: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.norm_first was True
  warnings.warn(
stage 2 flow+refine+action ...
  ⭐ 已切到 EMA 影子權重（m=0.999，五模組）
✅ 載入 ckpt_large-stitch_self_K8_c256_ch4_st8000_T128_ep2_gu_eorecon_ictr_tch0.5_emw0.999_wu500_dssoft_norf_cd0.1_bci_s33.pt（跳過訓練，只跑評估）  cfg={'K': 8, 'T_CAP': 128, 'ENC_OBJ': 'recon_ictr', 'LEARNED_REFINE': 0, 'COND_DROP': 0.1}
  decoder 內部點 RMSE 0.0981   打亂 u 之後 1.8960   用到的 u 值 +1.7979   ✓
PROTOCOL_EQUIVALENCE_PASS: 3 x 160 real maze steps, frozen oracle A, actions + XY byte-identical
REAL_CPU_PASS: M9 import, s33/EMA, encode_u false mask, head, decode, four P/Q same-u reruns, fresh-flow seed, legal-w short cap; GPU evidence=false
{"ckpt_sha256": "88180676e1f8d8df22c47bf4eeaf23eba6a73c5e92cacbf522ce3044095af787", "encoded_u_sha256": "fd89a90c7385dfd5c57288d7a2511b188fcb03c867cdf5e23bc0ab72ebb961a9", "encoder_shape": [1, 8, 256], "head_shape": [1, 4, 2], "head_u_hashes": ["fd89a90c7385dfd5c57288d7a2511b188fcb03c867cdf5e23bc0ab72ebb961a9"], "mask_all_false": true, "production_evidence": false, "source_sha256": "276c68fba6c41a05c1e3c2bd5bf4b25c39642d0dbac3ca7aa57cce381056ecfc"}
```

#### 第四棒 fourth

命令：`CUDA_VISIBLE_DEVICES='' CUBLAS_WORKSPACE_CONFIG=:4096:8 MUJOCO_GL=egl .venv/bin/python -B -m unittest discover -s experiments/_workorders/breakthrough-probe -p fourth_selftest.py -v`

exit code：0；raw SHA256：`a8b03f48a81c50e5f50c318a26980da9c886f705ea3a7be8b11e2d7f2ca76a20`

```text
test_30_freeze_flags_and_fixed_hash_inventory (fourth_selftest.FourthTests.test_30_freeze_flags_and_fixed_hash_inventory) ... ok
test_31_move1_positive_difference_and_conformance_downgrade (fourth_selftest.FourthTests.test_31_move1_positive_difference_and_conformance_downgrade) ... ok
test_32_reproduction_and_gate_threshold_four_five (fourth_selftest.FourthTests.test_32_reproduction_and_gate_threshold_four_five) ... ok
test_33_harvest_hostile_evidence (fourth_selftest.FourthTests.test_33_harvest_hostile_evidence) ... ok
test_34_historical_A_B_fingerprints (fourth_selftest.FourthTests.test_34_historical_A_B_fingerprints) ... ok
test_35_new_card_and_rule_boundaries (fourth_selftest.FourthTests.test_35_new_card_and_rule_boundaries) ... ok
test_36_combined_exits_from_full_artifacts (fourth_selftest.FourthTests.test_36_combined_exits_from_full_artifacts) ... ok
test_37_stub_fuzz_runloop_noise_switch_target_stuck (fourth_selftest.FourthTests.test_37_stub_fuzz_runloop_noise_switch_target_stuck) ... ok
test_38_runtime_coordinator_sources_gates_shards_heartbeat (fourth_selftest.FourthTests.test_38_runtime_coordinator_sources_gates_shards_heartbeat) ... /home/cymaxwelllee/Projects/lacot/experiments/_workorders/breakthrough-probe/runtime.py:300: DeprecationWarning: __array_wrap__ must accept context and return_scalar arguments (positionally) in the future. (Deprecated NumPy 2.0)
  points = (points * module.SD_XY_T + module.MU_XY_T).cpu().numpy()
PROGRESS move=1 rollouts=50/108 task=4 episode=20
PROGRESS move=1 rollouts=100/108 task=4 episode=4
PROGRESS move=1 rollouts=50/108 task=4 episode=18
PROGRESS move=1 rollouts=100/108 task=5 episode=32
ok
test_39_raw_survives_validation_failure (fourth_selftest.FourthTests.test_39_raw_survives_validation_failure) ... ok
test_40_calibration_fail_fast_and_estimators (fourth_selftest.FourthTests.test_40_calibration_fail_fast_and_estimators) ... ok
test_41_remaining_boundary_and_preprocessing_mutants (fourth_selftest.FourthTests.test_41_remaining_boundary_and_preprocessing_mutants) ... ok
test_42_checkpoint_resume_without_rerunning_completed_rows (fourth_selftest.FourthTests.test_42_checkpoint_resume_without_rerunning_completed_rows) ... PROGRESS move=1 rollouts=50/108 task=4 episode=18
PROGRESS move=1 rollouts=100/108 task=5 episode=32
PROGRESS move=1 rollouts=50/108 task=4 episode=18
PROGRESS move=1 rollouts=100/108 task=5 episode=32
ok
test_43_move3_discordance_both_directions_from_traces (fourth_selftest.FourthTests.test_43_move3_discordance_both_directions_from_traces) ... ok

----------------------------------------------------------------------
Ran 14 tests in 7.174s

OK
```

#### 第四棒 mutation audit

命令：`.venv/bin/python -B /tmp/detour_b4_mutations.py`（runner source 已嵌入 evidence）

exit code：0；raw SHA256：`f50cbdf60a7178830a0ab1be4b9e8db262bb4e7e3c3acaf76be5277df532a4ba`

```text
MUTATION_TWO_STATE N1-second-clause=True: before=GREEN mutant=RED(3 assertion failures,0 errors) restored=GREEN
MUTATION_TWO_STATE N1-CPU-table-one-static-flip: before=GREEN mutant=RED(1 assertion failures,0 errors) restored=GREEN
MUTATION_TWO_STATE N2-no-runtime-qualification-recalculation: before=GREEN mutant=RED(1 assertion failures,0 errors) restored=GREEN
MUTATION_TWO_STATE N2-no-direction-D-recalculation: before=GREEN mutant=RED(1 assertion failures,0 errors) restored=GREEN
MUTATION_TWO_STATE N2-no-static-u-check: before=GREEN mutant=RED(1 assertion failures,0 errors) restored=GREEN
MUTATION_TWO_STATE N2-no-tau-geometry-check: before=GREEN mutant=RED(1 assertion failures,0 errors) restored=GREEN
MUTATION_TWO_STATE N2-no-frozen-tau-check: before=GREEN mutant=RED(1 assertion failures,0 errors) restored=GREEN
MUTATION_TWO_STATE N2-no-static-qualification-check: before=GREEN mutant=RED(1 assertion failures,0 errors) restored=GREEN
MUTATION_TWO_STATE N2-no-F3-runtime-reference: before=GREEN mutant=RED(1 assertion failures,0 errors) restored=GREEN
MUTATION_TWO_STATE N2-no-O3-tail-runtime-reference: before=GREEN mutant=RED(1 assertion failures,0 errors) restored=GREEN
MUTATION_TWO_STATE N2-no-zero-length-rule: before=GREEN mutant=RED(1 assertion failures,0 errors) restored=GREEN
MUTATION_TWO_STATE N2-no-zero-length-qualification-inheritance: before=GREEN mutant=RED(1 assertion failures,0 errors) restored=GREEN
MUTATION_TWO_STATE N2-no-paired-reset-check: before=GREEN mutant=RED(1 assertion failures,0 errors) restored=GREEN
MUTATION_TWO_STATE N2-no-early-stop-reason-check: before=GREEN mutant=RED(1 assertion failures,0 errors) restored=GREEN
MUTATION_TWO_STATE N2-no-success-timing-check: before=GREEN mutant=RED(1 assertion failures,0 errors) restored=GREEN
MUTATION_TWO_STATE N2-no-w-field-check: before=GREEN mutant=RED(1 assertion failures,0 errors) restored=GREEN
MUTATION_TWO_STATE N2-actual-encode-input-wrong-endpoint: before=GREEN mutant=RED(1 assertion failures,0 errors) restored=GREEN
MUTATION_TWO_STATE N7-killer-catches-Exception: before=GREEN mutant=RED(1 assertion failures,0 errors) restored=GREEN
MUTATION_TWO_STATE N7-killer-does-not-check-message: before=GREEN mutant=RED(1 assertion failures,0 errors) restored=GREEN
MUTATION_AUDIT_PASS: 19 mutants; 17 new tests; files mutated=0 GPU=false archive_writes=0
```

STATUS: DONE
