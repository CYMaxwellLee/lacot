## wallpen2 r1 重啟交付

證明：下限只加在「teacher 列，而且該列自己的目標位移 ≥ F」的樣本上；其他行為逐位元不變。判準＝X1–X4、原 W1–W7／T1–T8／T6d 全過，移除目標門檻的新殺手實際 FAIL。

RESTART: from 5d640811

- 起點 trainer SHA256 `5d640811913b0e349f71489f82c325d4ccf8733fbcbe7a64c0759d03472c28a6`；六個還原檔與 lead 指定的 v2 基準逐位元相同。本次開工另存基準於 `/tmp/wallpen2-r1-restart-baseline-74rzforw/`。原工單已讀；PREREG-wallpen-v2.md §七 v2.1 SHA256 `e8cef8e86ed1447021bcb491be5aea662b6bc502507e192c90ea569ebb34ecd3`。
- `dT_i = ‖traj[i,-1] × SD + MU − (to_xy(s_i) × SD + MU)‖`，與 decoded `d_i` 共用條件起點；目標／起點均 detach。作用列為 `_REAL_W[0]==0` 且 `dT_i≥F`，只對這些列平均 `relu(F-d_i)²`。全真資料或 teacher 目標全短時，回傳可微零。目標首點與 goal g 均不用來判斷。
- trainer 只改下限函式、呼叫接點與活動記錄，新增處標 `wallpen2 r1:`。定標、退路、λ、抽樣與 checkpoint payload 保持既有實作。既有 teacher 描述量保留原子集；每個 floor log 點另記 `active_count`、`active_fraction`（分母＝整個 batch）。sidecar 新增 `floor_active_logs`，含 step 與上述兩欄。
- 前次失敗原文為 `AssertionError: X3 unrelated stream changed: excluded_short_count`。這是測試錯誤：`excluded_short_count` 含當步 decoded `d<F`，允許的參數分岔後不應要求相同。本次只在暖身／只量的前 70 步嚴格比對它；100 步的 batch、teacher eligibility、全域 torch／penalty torch／主 NumPy／teacher NumPy RNG 仍全部嚴格比對。前次結果沒有沿用。
- X2 使用真 CLI subprocess，構造短／長／恰等於 F 的 teacher 目標與真資料列；核對作用列平均、解析梯度、其他列與 prefix 梯度恰零、空集合可微零、起點／目標無梯度、RNG 無消耗。殺手只移除 `dT≥F` 條件，要求同一判準 FAIL。
- X3 讀取並驗 SHA 固定的 v2 基準，以真 C1、S1_FROM、STAGES=2、100 步 stage 2 比較。兩份暫存觀測副本在每次更新後加相同的唯讀 hash 呼叫，再用兩邊未插入觀測的 checkpoint bytes 證明觀測無副作用；不預載、追蹤或猴補 trainer。原 T 測試的既有 harness 照舊。
- W2 的原解析公式／teacher reduction／全真資料測試改提供符合新規則的目標；W3 正常定標與只量測試改用 F=5.65。舊 F=100 在 v2.1 會排除所有目標，不能再用來代表正常定標；小 F 的退路與其殺手照舊。

完整命令（單次原文輸出，沒有串接前次結果）：

```bash
OGBENCH_DATA_DIR=/home/cymaxwelllee/data/ogbench PYTHONDONTWRITEBYTECODE=1 \
/home/cymaxwelllee/Projects/lacot/.venv/bin/python -u \
  experiments/_workorders/wallpen/selftest_wallpen.py \
  > experiments/_workorders/wallpen/selftest-output.txt 2>&1
```

本次固定版本的單次完整自測 **PASS**：`n_tests=21 n_killers=14`，14 個殺手全部實際 FAIL；輸出只有一次開工標頭與一次最終 DONE，沒有拼接先前結果。全部 artifacts 保留於 `/tmp/wallpen-selftest-lzx7dmpl/`。trainer SHA256 `b0b7d80ecbec30f4690f1e9ecd67b5ae9954f454a086a44386f1981f3ac50919` 與執行輸出一致。

- X1／W1／T1：預設 checkpoint SHA 仍為 `637dbddf84afbf744bee2c2814fed4df07d70a917390dad36ee1ced3ca6032e8`；floor 關閉的舊新版 C2 SHA 都為 `3ae14b6f8c810e85e6d7db70ec7b396f82c631fe642823128b0e0af456c548f6`，以上都在這次重新跑出。
- X2：作用列 `[true,true,false,false,true,false,false]`，目標位移 `[9,8,7,6,10,9,7]`，F=8，floor=2；恰等於 F 的列納入，inactive／real／prefix 梯度恰為零，解析梯度、空集合可微零、起點／目標無梯度與 RNG 不變全部 PASS。移除目標門檻的殺手實際 FAIL。
- X3：前 70 步暖身／只量的所有觀測參數 hash 相同；100 步 batch、作用列、全域 torch／penalty torch／主 NumPy／teacher NumPy RNG 全部相同。舊版共記到 57 列 `teacher 且 dT<F 且 d<F`；首次分岔為 zero-based step=70（第 71 次更新），當步有 3 個衝突短目標列。作用列平均與其定標 λ 改變，從這個首次加罰步改變 flow 更新，符合這次唯一的訓練規則變更。
- X3 checkpoint：舊 `9d0505ddfca982199400cdbb65f0f3d49c69bc04a66288c70af53eb6a5d09bb0`；新 `45a43e2610267c38208eba70ff8eaf7a09c82b99342833e0da5b2368099f757e`。兩份觀測副本各自與未插入觀測的版本 bytes 相同；sidecar／log 的 10 個活動記錄逐點核對 PASS。
- W3：小 F 0/50 有效步，λ_floor=λ_flow=`0.1755717484278825`，fallback=true；F=5.65 正常 50/50 有效步，λ_floor=`0.0016192804968316534`，精確等於 κ×meanρ_floor；刪掉退路的殺手得到 λ=0 並實際退出。62 步暖身＋只量的完整 checkpoint tensors／cfg 相同。
- W4–W7／X4：真 C1 的 encoder／decoder 逐位元凍結，兩罰項對 flow 有有限非零梯度、condition 無梯度；C1′ 決定性、H3 schema、fresh loader、旗標拒絕、全部原 T1–T8／T6d 重跑 PASS。

SECOND PASS: 逐項重查目標與 decoded 路徑共用條件起點、≥ 邊界、作用列平均與空集合可微零；其餘 trainer 函式 AST 及 checkpoint 儲存原文與 5d640811 一致。完整 CPU 自測與 14 個殺手全部通過。只改本目錄與保留 /tmp 測試 artifacts，沒有 commit、刪檔、安裝套件、GPU 或 Slurm。

STATUS: DONE

本次完整執行結果見 [selftest-output.txt](selftest-output.txt)，對 v2 基準的變更見 [DIFF-wallpen2-r1.txt](DIFF-wallpen2-r1.txt)。以下為歷史交付，r1 的下限作用列規則以本節為準。

## wallpen2 變更

<!-- wallpen2: 本節是 v2 的現行契約；以下 r1/v1 段落保留為歷史紀錄。 -->

本輪證明 teacher-only 步長下限只對 flow 有梯度、定標不足會退回 λ_flow；S1_FROM 只可與 STAGES=2 的懲罰並用；預設訓練保持 d6d869ff 的 checkpoint bytes。以下 r1/v1 段落是原交付紀錄，本節優先。

- `LACOT_WALLPEN_FLOOR=F`：預設未設或 0 關閉，必須有限且非負。開啟要求 κ>0、STAGES 包含 2、TEACHER_CLEAN=1。S1_FROM 與 κ>0 並用另要求 STAGES **恰為 `2`**；其餘 r1 互斥 assert 保留。
- stage 2 復用牆罰的一次可微 flow 抽樣、同一份 detach condition 及凍住的 decoder 輸出。d 是解碼末點到 condition 起點的原始座標距離；`relu(F-d)^2` 只平均 `_REAL_W[0]==0` 的 teacher 列。全真資料 batch 為可微零，不讓真資料列稀釋平均。
- floor 與 wall 共用前 20% 暖身、50 步只量窗、θ=flow。‖∇floor‖>1e-12 才記 ρ_floor；有效步數≥10 時 λ_floor=κ×meanρ_floor，否則 λ_floor=λ_flow 並印 `wallpen2: floor 定標退路`。完成定標後 λ_floor≤0 或非有限即退出，避免默默停用。窗口之後才加入兩個加權罰項；量測中不加入任何 penalty loss。
- sidecar 新增 `F`、`λ_floor`、`ρ_floor`（全部測量值，跳過為 null）、`floor_valid_steps`、`floor_fallback`。未完成窗口時 λ_floor=null；有效步數仍記實際已測量數。checkpoint 的原始 payload、cfg、state_dict key/shape 不加欄位。
- 每個 stage 2 log 點增加 `floor_pen`、teacher d 的中位數（偶數取中間兩點平均）、teacher d<F 的比例；沒有 teacher 時兩個描述量印 None。開啟 floor 檔名加 `_fl{F:g}`，沿用既有 `_s1from`。

起始 HEAD `ca1a528589e2b7f34eb1b96cc5df3071fcdc4e0a`；原 trainer SHA256 `d6d869ff5e997f319905908d3e220980748c762a9dc8738a3b38c2cee5567247`。實際讀到 PREREG-wallpen-v2.md 的 SHA256 為 `2131461b93bd339d79eec94ec043687a757dc31e86c88f4c08795e0b10309563`，與工單引述的 `43b1cf32…` 不同；§一、§二與工單明列規則一致，本輪遵守工單明列要求。

W1–W6 在 [selftest_wallpen2.py](selftest_wallpen2.py) 中以 subprocess 直接跑 `train_wallpen.py --wallpen-cpu-test=train|floor`；不預載 trainer、不追蹤或猴補其模組。測試 CLI 強制隱藏 CUDA、CPU、單執行緒，跑完整原訓練迴圈；訓練後用 [cpu_probe_wallpen2.py](cpu_probe_wallpen2.py) 執行原始檔內確切的命名、拒絕覆蓋及存檔片段，跳過 rollout，不捏造 rollout 結果。CPU-only 梯度 probe 在真定標步以 autograd.grad 記兩個罰項的 flow norm，驗 decoder frozen、condition 梯度皆 None；不寫進 checkpoint。W7 保留原 T1–T8/T6d 測法與殺手，僅把原 r1 比較基底改為從 HEAD 讀取並驗 d6d869ff，避免從 v2 原文猜測重建。

完整測試命令與原交付相同：

```bash
OGBENCH_DATA_DIR=/home/cymaxwelllee/data/ogbench PYTHONDONTWRITEBYTECODE=1 \
/home/cymaxwelllee/Projects/lacot/.venv/bin/python -u \
  experiments/_workorders/wallpen/selftest_wallpen.py \
  > experiments/_workorders/wallpen/selftest-output.txt 2>&1
```

W1 與 T1 固定預設 SHA `637dbddf84afbf744bee2c2814fed4df07d70a917390dad36ee1ced3ca6032e8`；W2 構造原始座標路徑並檢查 teacher-only mean、短/長公式、全真資料零與真資料梯度零；W3 實際 S1_FROM 微訓練跑小 F／大 F、核對退路／正常定標，另在 62 步（12 暖身+50 只量）核對完整 checkpoint tensor/cfg 不變；W4 真 C1 載入、100 步 stage 2、stage 1 略過且三個 encoder/decoder state_dict 逐位元凍住，兩罰項 flow 梯度有限非零；W5 同 seed 兩次 SHA 相同，換 seed SHA 不同；W6 與實際 H3/G 比對 schema，fresh process 用本 repo 指定 `hsweep/eval_common.py`、seed 33 且不設 WALLPEN_EXPECTED_SEED 載入。三個新增殺手是所有列平均、刪除 floor 退路、STAGES=12+S1_FROM 拒絕；另測六種 floor 無效旗標組合。

交付 [selftest-output.txt](selftest-output.txt) 是一次完整執行的 stdout/stderr（含殺手 FAIL 原文），[DIFF-wallpen2.txt](DIFF-wallpen2.txt) 是對原 d6d869ff trainer／同 HEAD 測試與 README 的完整變更，包含新增測試檔。所有微訓練與失敗副本保留在 /tmp。只執行 CPU，沒有 GPU、Slurm、commit、刪檔或安裝套件。


本輪固定版本的單次完整自測 **PASS**：`n_tests=17 n_killers=13`（原十組檢查＋W1–W7；原十個殺手＋W2/W3/W4 三個殺手）。trainer SHA256 `5d640811913b0e349f71489f82c325d4ccf8733fbcbe7a64c0759d03472c28a6`；原始輸出與交付程式 SHA 一致。最終完整執行的 artifacts 保留於 `/tmp/wallpen-selftest-0psgh1_j/`；較早完整執行另保留於 `/tmp/wallpen-selftest-ajrws3pk/`，未串接進交付輸出。

- W1/T1：預設 SHA 仍為 `637dbddf84afbf744bee2c2814fed4df07d70a917390dad36ee1ced3ca6032e8`；floor 關閉的 C2 舊新版 SHA 都是 `3ae14b6f8c810e85e6d7db70ec7b396f82c631fe642823128b0e0af456c548f6`。
- W3：小 F=1e-6、0/50 有效步，λ_floor=λ_flow=`0.1755717484278825` 且 fallback=true；大 F=100、50/50 有效步，λ_floor=`7.432017578978955e-06` 且精確等於 κ×meanρ_floor。62 步完整 checkpoint tensor/cfg 比對相同。刪掉退路得到 λ_floor=0，實際 RuntimeError FAIL。
- W4：stage 1 跳過，traj_enc/e_pooler/u_dec state_dict 與真 C1 相同；floor 的 flow 梯度 norm 範圍 `2.076258420944214–133.03269958496094`，wall 為 `0.14846475422382355–0.5591681003570557`；decoder frozen、penalty 對 condition 參數的梯度為 None。檔名同時含 `_s1from`、`_fl5.65`；STAGES=12+S1_FROM 的殺手實際 assert FAIL。
- W5：s33 兩次 SHA 都是 `688ecf780f13be91749032e743587f445914990960d063c4d7f1037cdbf0835a`；s34 是 `8e53d06bc59899a741176b1138c009b3811c4049e3da963eff94410ff2fb6b03`。W6 schema 與 H3/G 一致，指定 eval_common fresh loader 成功。W7 原 T1–T8/T6d 與新增旗標拒絕全部 PASS。

SECOND PASS: 重查 teacher-only reduction、50 步只量邊界、正有限 λ_floor 與 S1_FROM 限制；固定版本完整 CPU 自測通過，十三個殺手全部實際 FAIL。未執行 GPU。


這輪只修 T8 測試 harness、補常駐 T6d gradcheck、加 sidecar UTF-8 與旗標安全 assert。證明：旗標開著時訓練行為逐位元不變，全部自測從頭乾淨通過。判準：`selftest-output.txt` 為單次完整執行且以 `STATUS: DONE` 結尾；新 detach 殺手實測 FAIL；預設與上一輪 SHA 相同，CPU C2 舊新版 SHA 相同。

## r1 變更

- **R1／T8**：`load` worker 由 subprocess exec 成 fresh process，在 frozen `load_ckpt` 前不設定 intra/inter-op threads、不執行 torch 平行運算；由唯讀載入器初始化 determinism。其他 worker 保留原 thread 設定。
- **R2／T6d**：從 trainer AST 取出實際 `_wallpen_sample` 函式，極小 `lacot.nf_head.Flow`（token_dim=1、seq_len=3、2 blocks、hidden=4、1 layer/head）、float64，全部 42 個參數張量／572 個 scalar 隨機化，含 `to_params`。只在測試注入明確 z；2D/3D cond 各跑完整 Jacobian 的 gradcheck(z) 與 gradcheck(z,cond,全部 flow 參數)，後者用 `torch.func.functional_call`。history、z、mu 三個 `.detach()` 殺手都先驗前向與 `Flow.sample` 逐位元相同，再要求 gradcheck FAIL。
- **R3**：exclusive-create sidecar 加 `encoding="utf-8"`，JSON 內容、欄位、ckpt 格式與命名不變。
- **R4**：κ>0 額外拒絕 INTENT、FSQ（spec/fit/load）、VQ、LO_W>0、CONT_TRAIN、LOAD_CKPT、S1_FROM、BOOT_DATA。使用唯讀正式 wrapper 的 G/C1/C2lo/C2hi `dryrun` 環境組（1500/8000 步），CPU import 停在模型建構前，確認 assert 不觸發；另外十組不相容設定確認指定 wallpen assert 實際拒絕。
- **行為不變**：新版只增上述 assert 與 sidecar encoding。自測將這兩處還原成暫存舊版，校驗舊版 source SHA256 `418670fd14d1851e35a9cd77461b7bf156550aaa93cec58c8af2ce1660b0ff91` 後跑 C2 κ=1、100+100 步；新版同設定 ckpt SHA 必須相同。pen、定標、teacher、反解計算完全未改，暖身期 pen 仍每步計算。

這張單只交「能正確跑 wallpen 四臂訓練的程式」。證明：兩個新開關都不設時，訓練與 `train_hsweep.py`（H=3）逐位元相同；`LACOT_TEACHER_CLEAN=1` 時，teacher 目標的佔據深度全部 ≤ τ_data，而且主 rng 消耗不變；`LACOT_WALLPEN_KAPPA=κ` 時，兩段懲罰照預註冊的形式與定標規則生效，梯度到得了 flow，更新不到 decoder。判準＝工單 §6 的八個自測全過，而且每個殺手輸入都真的 FAIL 過一次。

## 來源與實作

- 底稿：`../hsweep/train_hsweep.py`，SHA256 `f748e2cb844dd9af2d3d93a624ca2545635953c4ffdbd684cf69fe8b2ceb377e`；工作起點 HEAD `968af5e29235d787905b301738a32ba96153c157`。
- 規格：`/home/cymaxwelllee/Projects/elsa-agent-workspaces/luna/data/fleet-runs/breakthrough-u/wallpen/PREREG-wallpen-v1.md`，SHA256 `d2238aef400f6fb981026640e5c915fc702c55b7d4c9c6e03952cc071d03ca92`；精確 τ_data 與額外 target/g 檢查依本工單。
- [train_wallpen.py](train_wallpen.py) 與 [selftest_wallpen.py](selftest_wallpen.py) 為程式／CPU 自測交付。`train_wallpen.py` 是完整底稿副本，新增處標 `wallpen:`；原訓練、解碼、GeoEnergy 與 ckpt payload 均復用。變更原文見 [DIFF-vs-hsweep.txt](DIFF-vs-hsweep.txt)。
- 可微反解沿 `Flow.sample` 的 block 反序、permutation、2D/3D prefix 規則，用 token list/stack 取代 inverse 的就地寫入；functional padding 保留原 full-K history 切片的 strides，避免 CPU Linear 改選 GEMM 導致誤差。既有 `lacot/nf_head.py` 不變。

## 開關、常數與儲存

| 設定 | 行為 |
|---|---|
| `LACOT_TEACHER_CLEAN` | 預設 0，只接受 0/1。1 時第一次 jitter 用主 rng，最多 16 次重抽用 `default_rng(SEED+1_000_003)`，再退回不平移。檢查整路 128 點（原始座標弧長）、實際 H 前綴 T_CAP 點（原 hsweep 內插）、float32 g。 |
| `LACOT_WALLPEN_KAPPA` | 預設 0，有限非負浮點。非零須 CLEAN=1、AMP=0、LEARNED_REFINE=0、GRPO_W=DIV_W=0，且須有 decoder；另須 INTENT/FSQ/VQ 關、LO_W≤0、CONT_TRAIN=0、LOAD_CKPT/S1_FROM/BOOT_DATA 空。 |
| `LACOT_WALLPEN_STAGES` | 預設 `12`，亦接受 `1`、`2`。正式規格只用 `12`。 |

寫死常數：`WARM_FRAC=0.2`、`CAL_STEPS=50`、`RETRY=16`、τ_data 參考值 `0.05722857`，不接受環境覆蓋。任一新開關啟用時，以 CPU GeoEnergy(res=8) 掃全部 1,005,000 個 OBS 正規化 float32 點，差 >1e-5 即退出；懲罰另建訓練裝置 GeoEnergy。建構保護全域 NumPy／torch RNG 狀態。

pen 對每樣本只平均超過 τ_data 的點之 excess²，再平均 batch。Stage 1 用原始 `_pts1` 與 opt1 全部參數、當步完整 loss；stage 2 用每個 NLL cond 的一份可微樣本與 `l_nf`／flow 參數。stage 2 cond 使用 `flow_cond(cond,anc).detach()`，抽樣用專用裝置 torch.Generator(SEED+2_000_003)，decoder 凍結。flow 抽樣消耗專用 generator，不消耗全域／主資料 RNG。

每段前 floor(0.2×步數) 暖身，接著 50 步只量梯度範數比；兩個 `autograd.grad` 都 retain_graph，‖∇pen‖≤1e-12 記 null。第 50 次量完固定 λ=κ×meanρ，自下一步才加入 loss。量測期連 ×0 的 loss 項也不加。全跳過印「無可定標」，λ=0。未跑完定標或停用的段，sidecar λ=null，旗標 false（表示尚未判定，不表示已成功定標）。

檔名保留 hsweep 規則，於原 `_extra` 後追加 `_tc1`（clean）、`_wp{κ:g}`（κ>0）；啟用新開關且 STAGES≠12 時追加 `_wps{STAGES}`。兩開關關閉時原檔名不變。ckpt 的 key、shape 與 cfg 格式不加入新欄位。sidecar 為 `<完整 ckpt 檔名>.wallpen.json`，欄位為 `τ_data`、`κ`、`STAGES`、`λ_dec`、`λ_flow`、`ρ`（各段全部 50 個值，跳過為 null）、`clean_teacher`（first/retry/fallback）、`no_calibration`（各段旗標）。既有 ckpt 或 sidecar 都拒絕覆蓋；sidecar 使用 exclusive create。

log 額外印 τ_data、每段 λ/ρ_mean/n_used、各 log 點 pen 與 pen/recon_mse 或 pen/l_nf，及每 1000 步 teacher 累計三種計數；原 DIAG 行保持原格式。只量的階段，專用 flow 抽樣仍照每步執行，不碰主 rng 或全域 torch rng。

## CPU 自測與證據

從 repo 根目錄執行：

```bash
OGBENCH_DATA_DIR=/home/cymaxwelllee/data/ogbench \
PYTHONDONTWRITEBYTECODE=1 \
/home/cymaxwelllee/Projects/lacot/.venv/bin/python -u \
  experiments/_workorders/wallpen/selftest_wallpen.py \
  > experiments/_workorders/wallpen/selftest-output.txt 2>&1
```

正式四臂 dryrun 保留 wrapper 的 LACOT 設定與 1500/8000 步數，只用本地 OGBENCH_DATA_DIR、空 CUDA_VISIBLE_DEVICES 與暫存 OUT_DIR；在模型建構前停止，不跑正式訓練。

harness 在 import torch 前固定 `CUDA_VISIBLE_DEVICES=''`、`CUBLAS_WORKSPACE_CONFIG=:4096:8`，禁 bytecode，1 個 CPU intra/inter-op thread（load worker 的設定交由 frozen loader）。清除繼承的 LACOT_*，用工單固定配方與 H=3、batch=64、K=8、D_MODEL=256、T_CAP=128；只有步數設 3/100，LOG_EVERY=10。所有微訓練、殺手副本、ckpt 放 `/tmp/wallpen-selftest-*`，保留而不刪除。

訓練使用原迴圈。載入以 trace exception 停在 rollout 前，然後在原 module globals 執行底稿原文的命名／存在檢查／儲存片段，保留行號與 payload，略過 rollout 輸出。未評估 policy。T8 另外使用唯讀 frozen `eval_common.load_ckpt` 與現算 SHA，僅驗載入。CPU 測試不聲稱重現正式 H3 的 GPU SHA。

一手程式定位：旗標 assert `train_wallpen.py:488`；τ 掃描 `:524`；pen `:544`；可微反解 `:550`；定標 `:576`；teacher 重抽 `:758`；stage 1 接點 `:1329`；stage 2 cond detach／解碼 `:2126`；sidecar `:4112`；T6d `selftest_wallpen.py:228`；fresh load worker `:37`。

完整 stdout/stderr 與殺手失敗 traceback 見 [selftest-output.txt](selftest-output.txt)：本輪從 T1 到最後所有檢查的一次完整執行，沒有串接先前失敗嘗試；舊證據留在 `/tmp/wallpen-r1-original-wzgt83jw/`。最終 T1–T8（含 T6d）、r1 舊新版 C2 等價、R4 assert 全部 PASS。

| 自測 | 結果與一手數值 |
|---|---|
| T1 | PASS：6 批／192 個 teacher 路線選擇逐位元相同；CPU 3＋3 步兩份 ckpt SHA256 同為 `637dbddf84afbf744bee2c2814fed4df07d70a917390dad36ee1ced3ca6032e8`。多一次 torch.rand 的殺手 sha 不同，實測 FAIL。 |
| T2 | PASS：2048 份，整路／target／g 最大深度 `0.0571901798`；first=1727、retry=321、自然 fallback=0；另強制驗證 fallback 分支。第一次直接接受殺手最大深度 `0.0996175185`，實測 FAIL。 |
| T3 | PASS：70 批／2240 個路線選擇，主 rng 狀態、真資料列與路線序號相同；主 rng 重抽殺手實測 FAIL。 |
| T4 | PASS：τ_data=`0.05722856521606445`；res=4 得 `0.1349922717` 並退出，殺手實測 FAIL。 |
| T5 | PASS：20,000 個真資料點 pen=0；env.ij_to_xy 負對照 `(20,16)→(12,16)` pen=`0.0108401309699`，接 100 個自由點 Δ=0；全部點平均殺手 Δ=`0.00267437449656`，實測 FAIL。 |
| T6 | PASS：參考 H3 flow 權重、2D/3D cond，maxΔ 都為 0；flow 有有限非零梯度，decoder／cond 鏈 grad=None，全域 RNG 不變。改用 Flow.sample 殺手實測 FAIL。 |
| T7 | PASS：C1/C2 CPU 100＋100 步，DIAG stp 0–60 逐字相同、≥70 不同；λ_dec=`14.232937380682165`、λ_flow=`5.55985229245679`，兩段均 50/50 ρ 有效且等於 κ×meanρ。共用 τ_data=1e9 的字面殺手兩段各跳過 50/50，λ=0、印無可定標、JSON 有限，正定標判準實測 FAIL。C2 teacher 累計 first=5391、retry=1009、fallback=0。 |
| T8 | PASS：遞迴 key／shape 與 H3 相同、sidecar 欄位／檔名通過、ckpt 與 sidecar 既存都拒絕；frozen `load_ckpt` 成功載入 C2（無評估）。 |
| T6d | PASS：2D/3D 各兩組 gradcheck，共四組；history/z/mu 三個 detach 殺手均前向逐位元一致、Jacobian mismatch 實測 FAIL。 |
| r1 行為等價 | PASS：T1 SHA 與上一輪相同；CPU C2 舊新版 100+100 步 SHA256=`3ae14b6f8c810e85e6d7db70ec7b396f82c631fe642823128b0e0af456c548f6`。 |
| R4 | PASS：正式 G/C1/C2lo/C2hi 環境 dryrun assert 不觸發；十組禁止的設定實際拒絕。 |

`n_tests=10 n_killers=10`：T1–T8 八項、C2 舊新版等價一項、R4 一項；T6d 是 T6 的常駐子測試。原七個殺手加三個 detach 殺手全部實測 FAIL；T8 既存拒絕、R4 不相容拒絕不另算 mutation killers。

SECOND PASS: 最需要仔細看的兩處是 T6d 將確切 sampler AST 綁成 explicit-z／functional parameters，四個完整 Jacobian 檢查與三個 detach 殺手已驗過；T8 由 fresh exec worker 讓 frozen loader 唯一初始化 inter-op threads，實際載入已通過。

## 輸出契約

| 成功路徑必須保留的輸出 | 允許變更的行為 |
|---|---|
| 兩旗標未設：原 make_batch 結果、主 rng、CPU 微訓練 ckpt bytes、檔名及原 log 格式。 | 啟用 clean 時 teacher 的平移／s/g／目標內容可變；真資料列與路線選擇不變。 |
| ckpt 所有 state_dict key／shape、cfg 結構與既有評估載入 API。 | 啟用懲罰且定標完成後，指定 stage 的 loss／模型權重可變；停用段不加 penalty。 |
| C1/C2 stage1 暖身＋定標期原 DIAG 行逐字相同。 | 啟用旗標時增加 wallpen log、檔名後綴、JSON sidecar；既有 sidecar 也拒絕覆蓋。 |
| 全部真資料 pen=0，乾淨 teacher 三組檢查全 ≤τ_data。 | τ_data 與參考差 >1e-5、無效／不相容旗標組合立即失敗；全零定標 λ=0、明示無可定標。 |

## 測試環境差異聲明

CPU 自測涵蓋 float32 完整模型形狀與真資料，以及極小 float64 Flow 的完整 Jacobian，沒有執行 GPU、slurm、正式 1500＋8000 步訓練或 policy 評估。因此仍未驗證：CUDA grid_sample backward 的非決定性／deterministic-algorithm 限制；CUDA kernel、TF32、裝置 generator 與 CPU 的數值／亂數差異；可微反解 32 次 transformer 前傳所需的 GPU 顯存、速度與長程穩定性；GPU 正式預設等價臂能否達到 H3 的完整 ckpt SHA。AMP 在 wallpen 開啟時已 assert 禁用。正常完整腳本沿用底稿的 rollout 段，本交付沒有執行該段。

只寫本目錄與 /tmp，未改底稿／lacot 既有檔案、未碰 results、未 commit、未刪檔、未裝套件。

STATUS: DONE
