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
