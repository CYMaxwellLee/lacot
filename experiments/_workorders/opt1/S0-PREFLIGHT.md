# F5 S0 預檢對抗報告

日期：2026-09-28  
範圍：只讀 `experiments/_workorders/opt1/`、`experiments/scratch_lacot_rollout.py`，未跑 GPU、`sbatch`、`nvidia-smi` 或資料檔。  

## VERDICT: NO-GO

lead 的兩項修訂有效消除了「硬寫 GPU 2」這個自殺格；A 格也沒有發現 `LACOT_*` 拼寫造成的靜默 no-op。但目前仍不能發射：

1. guard 只證明 job 視角看到一張卡，沒有把 `SLURM_JOB_GPUS`／`CUDA_VISIBLE_DEVICES`／UUID 三者核成同一張；compute-app 查詢失敗還會被 `|| true` 吞掉，VRAM 非數字時測試也可能失敗開放。
2. 本站實際 dmon 輸出未在可讀範圍提供，`NR>2`、`$2=sm` 與採樣數判準尚未驗證；現行 awk 也不是卡片宣告的「固定 10 點中位數」判準。
3. card 要求 JSON 有 `ema_w`，但 rollout 的 `out` 沒有寫此欄；`EMA_W` 只進 checkpoint `cfg`。現行 r00 不會因 schema 不符而停。
4. 資料檔的預期 SHA256、byte size、筆數，以及 `T_CAP=128` 是否被 `min(128, MAX_TRAIN_T)` 靜默夾小，尚未驗；磁碟餘量也沒有證據。

## 九格證據

### A. `LACOT_*` 拼寫、型別與合法域

用 `rg` 將 sbatch 中所有鍵與 rollout 的 `os.environ.get` 逐一反查；34/34 都有實際讀取，沒有鍵名不存在而靜默回預設的 no-op。下表的「域」若註明「無顯式 assert」，表示這支腳本只轉型／以 truthiness 使用，沒有替發射者做完整範圍檢查。

| sbatch 鍵／值 | rollout 讀取 | 型別／域核對 | 結果 |
|---|---|---|---|
| `LACOT_AMP=0` | `AMP=int(...)`，`scratch_lacot_rollout.py:304` | int；0 關閉 | PASS，非 no-op |
| `LACOT_BC_INDEP=1` | `BC_INDEP=int(...)`，`:158` | int truthy；1 開獨立 BC | PASS，非 no-op |
| `LACOT_BON_N=0` | `BON_N=int(...)`，`:1579` | `>=0` assert；0 關 BoN | PASS，非 no-op |
| `LACOT_CHUNK=4` | `CHUNK=int(...)`，`:114` | 正整數語義；無顯式 assert，4 合法 | PASS，非 no-op |
| `LACOT_COMPILE=0` | `COMPILE=int(...)`，`:305` | int truthy；0 關閉 | PASS，非 no-op |
| `LACOT_COND=256` | `COND=int(...)`，`:109` | 正維度語義；無顯式 assert，256 合法 | PASS，非 no-op |
| `LACOT_COND_DROP=0.1` | `COND_DROP=float(...)`，`:154` | 目前值在預期 `[0,1]`；腳本未 assert 範圍 | PASS（域護欄偏弱） |
| `LACOT_CONS=self` | `CONS=os.environ.get(...)`，`:722` | `self` 走 self consistency 分支 | PASS，非 no-op |
| `LACOT_CONT_TRAIN=0` | `CONT_TRAIN=int(...)`，`:423` | 0；續訓 assert 不觸發 | PASS，非 no-op |
| `LACOT_DATA_SEED=-1` | `DATA_SEED=int(...)`，`:727` | `-1` 明確回退到 `SEED`，見 `:746` | PASS；不是 no-op |
| `LACOT_DEC_START=soft` | `DEC_START=get(...)`，`:222` | assert 允許空／`hard`／`soft` | PASS，非 no-op |
| `LACOT_DEV_EVAL=0` | `DEV_EVAL=int(...)`，`:127` | 0 關閉開發尺 | PASS，非 no-op |
| `LACOT_DIAG_DUMP=0` | `DIAG_DUMP=get(...) == "1"`，`:3070` | 字串 `0` 關閉 | PASS，非 no-op |
| `LACOT_DIAG_TRAIN=0` | `DIAG_TRAIN=int(...)`，`:736` | 0 關閉 stage-1 診斷 print | PASS；不是 no-op |
| `LACOT_EMA_W=0.999` | `EMA_W=float(...)`，`:1453` | `>0` 建 EMA；目前值在通常 `(0,1)` | PASS，非 no-op |
| `LACOT_ENC_OBJ=recon_ictr` | `ENC_OBJ=get(...)`，`:141` | `:939` 允許 `sg_infonce/recon/recon_ictr` | PASS，非 no-op |
| `LACOT_ENV=pointmaze-large-stitch-v0` | `ENV_NAME=get(...)`，`:41` | 用來拼 data path；檔案存在未驗 | PASS（資料仍 ASK） |
| `LACOT_EVAL_EPISODES=2` | `SEEDS=int(...)`，`:2481` | 每 task 2 集；無正值 assert | PASS（5 task/10 集仍須實跑驗） |
| `LACOT_EVAL_RS=0` | `RS_PRE=[int(...)]`，`:3181` | 得 `[0]`；空 list 會在 `:3184-3185` 停 | PASS，實際只收 R0 |
| `LACOT_FLOW_PROBE=0` | `FLOW_PROBE=int(...)`，`:334` | 0 關閉 | PASS，非 no-op |
| `LACOT_GRPO_W=0` | `GRPO_W=float(...)`，`:285` | `>=0` assert；0 使 GRPO branch 不建 | PASS，非 no-op |
| `LACOT_K=8` | `K=int(...)`，`:108` | 正維度語義；無顯式 assert，8 合法 | PASS，非 no-op |
| `LACOT_LEARNED_REFINE=0` | `LEARNED_REFINE=int(...)`，`:151` | 0 關 refine；也滿足 teacher 的 `:477` 前置條件 | PASS，非 no-op |
| `LACOT_LOAD_CKPT=` | `LOAD_CKPT=get(...)`，`:419` | 空字串＝從零訓練 | PASS，非 no-op |
| `LACOT_LOG_EVERY=1000` | `LOG_EVERY=max(1,int(...))`，`:309` | 正整數；1000 合法 | PASS，非 no-op |
| `LACOT_OUT_DIR=$OUT` | `_OUT_DIR=get(...)`，`:3686` | 字串路徑；用作產物父目錄 | PASS，非 no-op |
| `LACOT_PREREQ=0` | `PREREQ=int(...)`，`:129` | 0 關閉前置診斷 | PASS，非 no-op |
| `LACOT_S1_FROM=` | `S1_FROM=get(...)`，`:426` | 空字串＝不載入／不凍 stage 1 | PASS，非 no-op |
| `LACOT_SEED=33` | `SEED=int(...)`，`:716` | 整數；同時餵 torch 與資料 RNG fallback | PASS，非 no-op |
| `LACOT_STEPS1=1500` | `STEPS1=int(...)`，`:1029` | 正整數；stage 1 實跑 1500 | PASS，非 no-op |
| `LACOT_STEPS2=2000` | `STEPS2=...int(...)`，`:1444`；續訓另讀 `:1448` | 空 ckpt 時實跑 2000；tag 用它 | PASS，非 no-op |
| `LACOT_TCAP=128` | `T_CAP_REQ=int(...)`，`:123`；實際 `T_CAP=min(...)`，`:449` | 請求值 128；若 `MAX_TRAIN_T<128` 會靜默變小 | CONDITIONAL，需資料驗證 |
| `LACOT_TEACHER_MIX=0.5` | `TEACHER_MIX=float(...)`，`:474` | `>0` 建 teacher 題庫；目前與 `LEARNED_REFINE=0` 相容 | PASS，非 no-op |
| `LACOT_WARMUP=500` | `WARMUP=int(...)`，`:1012` | `>0` 對兩段 lr warmup | PASS，非 no-op |

特別點名的七個易誤判鍵均有作用：`DATA_SEED`（`:727,746`）、`EMA_W`（`:1453`）、`STEPS1/STEPS2`（`:1029,1444`）、`GRPO_W`（`:285`）、`DIAG_TRAIN`（`:736,1129`）、`LOG_EVERY`（`:309,1955`）。因此 A 的「拼寫」本身不構成 blocker。

反向對照名片 tag：

| tag 片段 | 實際來源 | 核對 |
|---|---|---|
| `large-stitch_self_K8_c256_ch4` | `ENV/CONS/K/COND/CHUNK` | 全有 env |
| `st2000` | `STEPS2` | 有 env；非 `STEPS1` |
| `T128` | `TCAP` 的實際夾後值 | 只有請求值固定，實際值需驗資料 |
| `ep2` | `EVAL_EPISODES` | 有 env |
| `eorecon_ictr` | `ENC_OBJ` | 有 env |
| `tch0.5` | `TEACHER_MIX` | 有 env |
| `emw0.999` | `EMA_W` | 有 env |
| `wu500` | `WARMUP` | 有 env |
| `dssoft` | `DEC_START` | 有 env |
| `norf` | `LEARNED_REFINE=0` | 有 env |
| `cd0.1` | `COND_DROP` | 有 env |
| `bci` | `BC_INDEP=1` | 有 env |
| `s33` | `TAG_SEED=SEED`，`:1975` | 有 env |

但 tag/schema 並不涵蓋所有會改權重的有效預設：

- `LACOT_ENC_OBJ=recon_ictr` 時，`W_ICTR=0.2`、`ICTR_SIGMA=0.05` 由 `:142-143` 讀取，並在 `:1113-1122` 真正進 loss；sbatch 沒顯式設定，tag、JSON、ckpt cfg 也沒有這兩欄。
- `LACOT_DEC_START=soft` 時，`DEC_START_W=1.0` 由 `:224` 讀取，並在 `:1110-1112` 進 stage-1 loss；同樣不在 sbatch/tag/schema。
- `EMA_M=0.996`（`:723`）在本場景 `CONS=self` 且 `LEARNED_REFINE=0` 下不走 EMA-consistency 分支；`W_VAR/W_COV=0` 也關閉，兩者不是本次的有效漏列分支。

結論：沒有「不存在的鍵＝靜默預設」；但是上述三個有效預設使 provenance 不完整，需在 ASK 中明確裁定「以 Git SHA 預設為凍結值」或補進執行 manifest。

### B. `guard_gate`、dmon、cgroup 與 VRAM

證據在 `s0run.sbatch:30-52`：

- `nvidia-smi -L` 行數必須等於 1（`:30-35`），這對 cgroup 未隔離／多卡誤配是 fail-closed，且已不再寫死 GPU 2。
- 但一張可見卡不等於「Slurm 分配的那一張」：腳本只把 `SLURM_JOB_GPUS`、`CUDA_VISIBLE_DEVICES` 印到 `gpu-assignment.txt`（`:37-39`），沒有比較它們與 `GPU_UUID`。若 cgroup 映射錯卡，數量閘仍會放行。
- compute-app 查詢用 `|| true`（`:43`）。查詢失敗會留下空檔，接著 `grep` 不命中（`:44`），PID 閘可能 fail-open；這不是完整的「查不到就停」。
- `MEMUSED` 由 `:48` 取得後以 `${MEMUSED%%.*}`（`:49`）截去小數。若輸出 `511.9`，會被當成 511 而通過 512 MiB 閘；若輸出 `N/A`／空字串，`[` 的數字比較報錯，但它位於 `if` 條件，`set -e` 不保證因此退出，仍可能繼續。本站 driver 的實際輸出格式未提供。
- dmon 用 `nvidia-smi dmon -s u -c 10`（`:50`），awk 以 `NR>2` 跳兩行表頭，取 `$2` 的整數樣本（`:51`）。標準 dmon 的常見欄位確實是 `gpu sm ...`，但本次沒有本站實際 dmon 輸出，不能把 `$2=sm` 與「兩行表頭」當已驗事實。
- 即使欄位對，現行判準也是 `n>=8 && busy<=2`，不是固定 `n==10`，也不是卡片寫的 10 點中位數 `<5%`；錯誤訊息卻寫「10 樣本」。dmon 失敗雖會被 `|| true` 吞掉，空檔通常會被後面的 `n>=8` 擋住，但部分輸出／欄位漂移仍可能被錯算。
- 每跑前重查（`:59`），但沒有跑中監測；別的工作在一次訓練中途搶卡，r00/r01 仍可能被污染，最多到下一次 pre-run 才被發現。`:75` 的 `gpu-after.csv` 是事後紀錄，不是隔離閘。

判定：B **NO-GO**。lead 的「可見卡數 !=1 停機」是必要但不充分；需補 UUID 分配核對、本站 dmon 實樣，並裁定查詢／解析失敗一律停機。

### C. 產物檔名、`OUT_DIR` 與 hash glob

#### 名稱推導

`scratch_lacot_rollout.py:3681-3682` 的 tag 公式，在 `MAX_TRAIN_T>=128` 時名義上得到：

```text
large-stitch_self_K8_c256_ch4_st2000_T128_ep2_gu_eorecon_ictr_tch0.5_emw0.999_wu500_dssoft_norf_cd0.1_bci_s33
```

但 `T_CAP` 先在 `:449` 做 `min(T_CAP_REQ, MAX_TRAIN_T)`；所以資料最大軌跡若小於 128，實際是 `T<MAX_TRAIN_T>`，不是 card 寫死的 `T128`。JSON 會同時寫 `tcap` 與 `tcap_requested`（`:3144-3146`），這能事後看出問題，但沒有在啟動時阻擋。

每個 `r00`–`r05` 的 `LACOT_OUT_DIR="$OUT"`（`s0run.sbatch:57-73`）會讓腳本 `:3686-3690` 寫：

```text
results/opt1/s0/rXX/rollout_<tag>.json
results/opt1/s0/rXX/ckpt_<tag>.pt
results/opt1/s0/rXX/time.txt
results/opt1/s0/rXX/stdout.log
results/opt1/s0/rXX/stderr.log
results/opt1/s0/rXX/artifacts.sha256
results/opt1/s0/rXX/gpu-after.csv
```

`LACOT_OUT_DIR` 是「產物父目錄」，不是 tag、不是只給 rollout JSON 的開關；checkpoint 也用 `dirname(dst)` 寫入同一個目錄（`:3792`）。因此這一點與 card 的 `rXX` schema 相符。

#### schema 攻擊

- card 要求 JSON 至少有 `ema_w`；實際 `out=dict(...)`（`:3144-3175`）有 `ema_m`，但全檔沒有 `ema_w`。`EMA_W` 只在 checkpoint cfg（`:3827-3831`）落地。故「JSON schema」目前必缺一欄，且 shell 不會因此失敗。
- card 要求 `episodes=10`；程式寫的是 `N_TASKS * SEEDS`（`:3146`），沒有 assert `N_TASKS==5`。環境／ogbench 版本若不是五 task，會產生非 10 但仍可 hash 的結果。
- card 要求 ckpt `cfg`／`ema` 驗證；腳本只保存，沒有在 r00 完成後由 sbatch 檢查它們。

#### glob 行為

`s0run.sbatch:74` 的

```bash
sha256sum "$OUT"/rollout_*.json "$OUT"/ckpt_*.pt > "$OUT/artifacts.sha256"
```

在 Bash 未開 `nullglob` 時，無匹配 glob 會原樣傳給 `sha256sum`，命令報錯；因它不是條件命令，`set -e` 會終止 job。這對缺產物是 fail-closed，但重導向可能先建立一個空的 `artifacts.sha256`。反過來，腳本也沒有檢查「恰好一個 JSON／一個 PT」；若 rXX 事前有 stale 或多份匹配，會把全部一起 hash。`mkdir -p`（`:60`）不會清理舊檔。

判定：C **NO-GO**，主因是 card 要的 `ema_w` 缺失與未做 schema gate；glob 本身不是靜默放行，但缺檔時會以部分產物＋空 hash 清單留下殘骸。

### 4. 機器／卡位是否對

- lead 已驗事實：lady 的 Gres=`gpu:3`、64 CPU、128G RAM，GPU 0/1/2 均為 RTX 2080 Ti；marl2026 佔 1 卡、2 卡閒；torch `2.6.0+cu124` 且 `cuda=True`。
- sbatch 只要求 `--gres=gpu:1 --cpus-per-task=8 --mem=32G`，見 `s0run.sbatch:2-8`；以 lady 的帳面資源看，CPU/RAM 配額未超額，單卡方向正確。原卡的物理 GPU 2 自殺格已移除。
- 未驗的是「這個 job 看到的唯一 UUID 是否真為 Slurm 分配的 UUID」，以及 `CUDA_VISIBLE_DEVICES` 在本站是物理 index、重映射 index 還是 UUID。兩張閒卡的 lead 觀察不能替代 job 內核對。

判定：硬體規格 **PASS（依 lead 證據）**；分配／獨佔 **ASK，未達發射條件**。

### 5. 磁碟餘量

本次按紅線沒有跑 `df`、`du` 或任何無界掃描；工單也沒有提供 `results/opt1/s0` 所在 filesystem 的 free bytes／free inodes。六個 checkpoint、JSON、log、hash 與 GPU 快照的實際大小均未測。

判定：**ASK**，不能用 128G RAM 或 GPU VRAM 推測磁碟足夠。

### 6. 覆蓋面／既有產物

在本 checkout 以 bounded `test`／`rg --files results/opt1/s0` 檢查，`results/opt1/s0` 目前不存在；因此沒有看到會被同名覆蓋的既有 S0 產物。這是乾淨起點的 PASS，不代表腳本本身有清場護欄：它會 `mkdir -p`，不會拒絕先存在的 `rXX` 或 stale glob。

判定：**PASS（截至本次讀取）＋執行前仍須重查**。

### 7. ETA 帳

card 的推導是原 8000 步 0.5–2 小時按四分之一縮放為 7.5–30 分，再加固定 stage 1、初始化、eval/IO，估 20–60 分／跑；六跑即 120–360 分（2–6 小時），低於 8 小時 limit，這個算式作為粗估合理但不是實測。

反方壓力：`LACOT_EVAL_RS=0` 並不只做一個 10 集呼叫；主流程固定依序跑 `bc`、`null_u`、`shuf`、`R0`（`:3412-3417`），每個 `rollout` 都是 5 task × 2 seed。因此一次 run 至少是四個 10 集 eval arm，card 的「10 集 rollout」容易低估固定 eval 成本。`r00` 也是完整訓練＋eval＋寫 checkpoint，不是短 smoke。

另外，sbatch 沒有在 r00 後檢查 time 再決定是否進 r01；若 r00 超過估算，後五跑仍會照跑直到 8 小時被切斷。

判定：**AMBER／ASK**。只有拿到 r00 `time.txt` 後，按 `6×r00` 加固定 overhead 並確認小於 8 小時，才可把 20–60 分當現場 ETA。

### 8. 依賴與外部前置

程式 import `torch`、`numpy`、`ogbench` 及 repo 內 LaCoT 模組，見 `scratch_lacot_rollout.py:8-16`；eval 以 `ogbench.make_env_and_datasets(..., dataset_dir=OGB_DATA)` 執行，見 `:2478-2481`。sbatch 另依賴：

- `/archive/cymaxwelllee/LaCoT/.venv/bin/python`、外部 NPZ data、`ogbench`、MuJoCo `osmesa`；
- `/usr/bin/time`、`nvidia-smi`、`awk`、`grep`、`sha256sum`、`git`；
- Slurm `admin/it/great-mage` 權限與 lady nodelist/GRES/cgroup 配置。

腳本只用 `test -x "$PY"`、`test -f "$DATA"` 與 torch CUDA assert（`:27-29`）；它沒有驗資料筆數、檔案大小、預期 hash，也沒有在 shell 端先 import `ogbench`／建立 MuJoCo env。先遣的 torch 驗證不能推出其餘依賴均已驗。

判定：**ASK／未閉合**。

### 9. r00 暖機兼 smoke 能否蓋死點

設計上 r00 會走與 r01–r05 完全相同的 `guard_gate`、1500+2000 訓練、四個 eval arm、JSON/PT 寫入、hash 與 `gpu-after.csv`（`s0run.sbatch:54-75`），所以能抓：Python/匯入、CUDA、資料找不到、shape/loss 炸掉、rollout 初始化、checkpoint 寫入、缺檔 glob 等「跑掛」問題；暖機統計按 card 丟棄也合理。

它抓不到或不會自動擋：

- JSON 缺 `ema_w`、`episodes` 非 10、`tcap` 被夾小、ckpt cfg 缺 `EMA_W`／`ema` 等「量錯／量了不可信」問題；
- dmon 欄位錯位、compute-app query fail-open、VRAM 小數截斷；
- 一次 run 中途發生的卡爭用；
- r00 完成後沒有人工 gate，腳本直接進 r01。

判定：**PARTIAL PASS**。r00 是好的 crash smoke，但不是 schema／隔離／可信度 smoke；必須人工驗 r00 產物後才能放行 r01–r05。現行腳本沒有這個 interlock。

## ASK 清單（未查不到就不猜）

1. 請提供一次本站 job 視角的 `SLURM_JOB_GPUS`、`CUDA_VISIBLE_DEVICES`、`nvidia-smi -L`、UUID 對照，以及完整 `nvidia-smi dmon -s u -c 10` stdout；確認本站 `$2` 確為 `sm`、表頭確為兩行、資料確有 10 筆。
2. 請裁定 guard 的 fail policy：`--query-compute-apps`／dmon／memory query 任一失敗是否一律停；目前 `:43`、`:50` 吞錯，`:49` 對小數／`N/A` 不安全。
3. 在 lady 驗 data NPZ 的存在、預期 SHA256、byte size、row count；並確認 `MAX_TRAIN_T>=128`，否則本件不是 card 凍結的 T128。
4. 提供 `results/opt1/s0` 所在 filesystem 的 `df -h` 與 inode 餘量；本預檢沒有執行 df。
5. 決定 `ema_w` schema：要把 `EMA_W=0.999` 寫入 rollout JSON／manifest，或明確改 card 以 checkpoint cfg 為唯一來源；在此之前 r00 不能算符合 card。
6. 明確凍結並記錄 `W_ICTR=0.2`、`ICTR_SIGMA=0.05`、`DEC_START_W=1.0`；目前三者依 Git SHA 預設運作，但不在 sbatch/tag/JSON/cfg。
7. r00 完成後先人工核對 JSON、ckpt cfg/`ema`、恰一份 JSON/PT、`tcap=128`、`episodes=10` 與 `time.txt`，再決定是否進 r01；不要讓 loop 自動把 smoke 失敗帶入 N=5。
8. 確認同一個 `/archive` venv 實際能 import `ogbench`、啟動 `MUJOCO_GL=osmesa`、使用 `/usr/bin/time`，以及提交帳號／partition／qos 權限；本預檢未跨出 repo 讀取這些外部狀態。

## 自我懷疑（非空）

- 我沒有取得本站 dmon 實樣，也沒有執行 `nvidia-smi`／Slurm；對 `NR>2` 與 `$2=sm` 的判讀只能說符合常見格式，不能冒充本站已驗。
- `results/opt1/s0` 在本 checkout 不存在，不等於提交時共享 filesystem 一定沒有別的 job 先建立同名目錄。
- `W_ICTR`、`ICTR_SIGMA`、`DEC_START_W` 可能是作者有意依 commit 預設凍結；我把它們標成 provenance/ASK，不把它們誤判成拼寫 no-op。
- 20–60 分／跑的線性外推可能在訓練占主導時成立，但一次 run 實際有四個 10 集 eval arm；沒有 r00 wall 就不能替它背書。
- lead 的 3 卡實測與 torch 驗證是有效先遣證據，但仍不能替代本 job 的 UUID 映射、cgroup 可見性、資料與磁碟證據。

STATUS: DONE
