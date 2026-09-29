# ucontrast2 發射單 v1：①預檢對抗報告

日期：2026-09-29

受檢：`U2-LAUNCH-CARD.md`、六支 ucontrast2 sbatch、ucontrast2 `run.py`、
`../ucontrast1/README.md`、`REWORK-ORDER.md`，並以題示的 ⓪／lead 已驗事實作
外部事實錨。本查唯讀；沒有 Slurm／SSH／jasmine，因此不把本機靜態結果冒充成
GPU 或排程實測。

## 總判定

**VERDICT: NO-GO（卡片與鏈序契約尚未閉合）**

核心 launcher 的 gate、array 範圍、正式輸出根目錄和資源申請大致對；但目前
至少有四個上機前要關的缺口：

1. 發射單的 `n_tasks`／`n_draws` 計數與 ucontrast1 schema 及 `run.py` 的硬驗收不一致。
2. 卡上宣告的「select → smoke → formal」不是 Slurm dependency；正式主跑不驗 smoke，
   s35 smoke 的 A index 在 select 前可以直接跑。
3. 卡上的 CPU select 指令被裸換行拆成無效命令，且「zeldajr 或 jasmine」與 archive
   可見性說法互相矛盾。
4. 卡片宣稱「別無其他寫入、不寫 NFS home」，但六支 sbatch 都在 `$HOME` worktree
   以沒有 `-B`／`PYTHONDONTWRITEBYTECODE` 的 Python 啟動，存在 `.pyc`／`__pycache__`
   隱藏寫入。

以下 PASS 只表示靜態契約已對上；不替代 jasmine smoke。

## FLEET-LAUNCH ①八格

| 格 | 判定 | 結論 |
|---|---|---|
| 1. 目標、凍結輸入、gate | PASS（沿用 lead 事實） | `validate_inputs()=READY`、provenance SHA、flag-off SHA 和五個 OUT 不存在按題示採信；`run.py:68-95` 仍會重驗 gate。 |
| 2. 節點、資料、資源 | PASS-with-risk | 六支均指向 ada-lite／jasmine／gpu:1／8 CPU／16G／12h；archive、CUDA/MuJoCo 與 Slurm 實際可用性未在本機驗。 |
| 3. 六支 sbatch 展開 | PASS | array、job name、mode、outdir 逐支相符；見下表。 |
| 4. CLI／schema 契約 | **NO-GO** | flags 本身大多正確，但 select 命令斷行、`--calibration-root` default 陷阱及結果計數都需修。 |
| 5. 寫入、exclusive、log | **NO-GO（寫入面）／PASS（結果隔離）** | `run.py` 確實做 exclusive directory/file；但未禁 Python bytecode，違反卡片的「只寫 archive」。logs pattern 不撞 `main-33906`。 |
| 6. 鏈序與依賴 | **NO-GO** | 沒有 `--dependency`，且正式 launcher 不檢查 smoke receipt；s35 smoke A 不檢查 selection。 |
| 7. smoke 覆蓋面 | PASS-with-residual | 卡片誠實指出 n64 smoke 不走 derived route，但把 draws 寫錯；64-draw derived route 仍是首次 GPU 執行。 |
| 8. ETA × OOM／死點 | PASS-with-residual | 12h 對 6.2h 有約 1.9× 牆鐘餘量；n64 derived full GPU、CUDA peak、逾時／搶占續跑仍未閉合。 |

## 1. 六支 sbatch 與鏈序逐字對照

| 腳本 | job／array | launcher 實參 | outdir |
|---|---|---|---|
| `s35-mutant.sbatch` | `ucontrast2-s35-mutant`／單格 | `s35 --s35-stage mutant`，無 `--index` | `/archive/cymaxwelllee/ucontrast1/s35-calibration` |
| `s35-sweep.sbatch` | `ucontrast2-s35-sweep`／`0-3` | `s35 --s35-stage sweep --index 0..3` | 同上 |
| `s35-smoke.sbatch` | `ucontrast2-s35-smoke`／`0-1` | `s35 --smoke --index 0..1` | `/archive/cymaxwelllee/ucontrast1/smoke2-s35` |
| `s35.sbatch` | `ucontrast2-s35`／`0-1` | `s35 --index 0..1` | `/archive/cymaxwelllee/ucontrast1/s35` |
| `n64-smoke.sbatch` | `ucontrast2-n64-smoke`／`0-1` | `n64 --smoke --index 0..1` | `/archive/cymaxwelllee/ucontrast1/smoke2-n64` |
| `n64.sbatch` | `ucontrast2-n64`／`0-1` | `n64 --index 0..1` | `/archive/cymaxwelllee/ucontrast1/n64` |

證據：六支檔案的 `#SBATCH` 與 exec 行；例如正式 n64 在
`n64.sbatch:2-17`，正式 s35 在 `s35.sbatch:2-17`，校準 sweep 在
`s35-sweep.sbatch:2-19`。本機 `bash -n experiments/_workorders/ucontrast2/*.sbatch`
全數通過；`run.py`、`rollout_n64.py`、`analyze.py` 也以 `python3 -B` AST 解析通過。

job name 和 output pattern 分離良好：

- 正式／smoke array 用 `%A_%a`，mutant 用 `%j`。
- 既有 `/archive/cymaxwelllee/ucontrast1/logs/` 目錄按 lead 事實已存在，Slurm
  可在 body 前開 output file。
- 9/28 的 `main-33906` 殘留不會被新腳本命中；新檔名前綴是 `n64-`、`s35-`、
  `s35-smoke-`、`s35-sweep-` 或 `s35-mutant-`。這一點 PASS，僅仍待上機確認權限。

## 2. CLI 契約與 default 陷阱

`run.py --help` 顯示的 choices 與六支實參相容：`mode` 只有 `n64/s35`，
`--s35-stage` 只有 `mutant/sweep/select`，`--outdir` required，`--index` 是 int，
`--smoke` 是 flag。

實際效果如下：

- `n64-smoke`：`draws=2`、`episodes=5`、`questions=None`，走原
  `experiments/scratch_lacot_rollout.py`。
- `n64` 正式：`draws=64`、`episodes=40`、載入 50 題 manifest，才走
  `rollout_n64.py` derived source。
- `s35-mutant`：`index=None` 合法，`args.outdir` 被當 calibration root。
- `s35-sweep`：index `0` 是 A/σ0，`1..3` 是 B/σ `.05/.10/.20`；先驗
  `s35_mutant_gate`。
- `s35 --s35-stage select`：`run.py:262-276` 把 **`args.outdir`** 當 calibration
  root；此 branch 不使用 `--calibration-root`。所以正確的 select 入口必須把
  `--outdir` 寫成 `/archive/cymaxwelllee/ucontrast1/s35-calibration`。
- s35 formal／smoke 才用 `--calibration-root`；六支 sbatch 沒傳，依靠
  `run.py:18`／`:359` 的固定 default `/archive/cymaxwelllee/ucontrast1/s35-calibration`。
  在本靶固定路徑下是一致的，但 operator 若只改 `--outdir` 而忘了傳 root，會讀錯地方。
- `run.py:178-179` 將 n64 固定映射到 seed 33、s35 固定映射到 seed 35；卡片只泛稱
  「s33/s35 ckpt SHA 由 gate 釘」，沒有明寫 n64＝s33。這與 50 題來自 s33 題池及
  現有 gate 是一致推論，但屬重要的未明示輸入；若意圖確實如此，應在卡片補一句，
  否則應把 checkpoint／seed 變成明確且受 gate 的參數。

### 必改 A：卡上的 select 命令不是可貼上的 shell 命令

`U2-LAUNCH-CARD.md:34-36` 實際是：

```text
cd ~/Projects/lacot-u-frozen && <venv-python> experiments/_workorders/
ucontrast2/run.py s35 --s35-stage select --outdir /archive/cymaxwelllee/
ucontrast1/s35 --calibration-root /archive/cymaxwelllee/ucontrast1/s35-calibration
```

三行間沒有 `\`；第一個 path 被切斷，第三行會被 shell 當成另一個命令。
而且 `<venv-python>` 仍是 placeholder。照此貼上，select 會先以錯誤的
`--outdir /archive/cymaxwelllee/` 尋找校準產物，然後再執行不存在的命令，不能產生
`sigma-selection.json`。

**改法：**把 `U2-LAUNCH-CARD.md:33-37` 改成只在 jasmine 執行的一行完整命令，例：

```bash
cd "$HOME/Projects/lacot-u-frozen" && /archive/cymaxwelllee/LaCoT/.venv/bin/python -B experiments/_workorders/ucontrast2/run.py s35 --s35-stage select --outdir /archive/cymaxwelllee/ucontrast1/s35-calibration
```

卡片目前同時寫「zeldajr 或 jasmine 均可」和「select 在 jasmine 跑」；按題示
archive／venv 可見性，應刪除 zeldajr 選項，或另給 zeldajr 能讀同一校準 root 的
明確絕對路徑。不能兩種說法並存。

### 必改 B：卡上的結果計數把欄位含義寫錯

`ucontrast1/README.md:30` 明定 `n_tasks` 是 `(task, episode)` 題對數，不是
official task 數；`collector.py:129` 將 `n_draws` 寫成所有 raw draw rows。
ucontrast2 launcher 又在 `run.py:234-238` 硬驗 `n_draws == count * draws`。
所以 `U2-LAUNCH-CARD.md:51-53` 的 `n_draws=64/8` 和 smoke `n_tasks=5` 都不對：

| stage | `n_tasks` | `draws_per_task` | 實際 `n_draws` |
|---|---:|---:|---:|
| n64 formal | 50 | 64 | 3200 |
| s35 formal | 200 | 8 | 1600 |
| n64 smoke | 25 | 2 | 50 |
| s35 smoke | 25 | 8 | 200 |
| s35 mutant／每個 sweep bin | 25 | 8 | 200 |

`run.py:187-190` 直接給出 draws／episodes；s35 calibration 的
`run.py:325-330` 直接硬驗 `25/200/8`。**改法：**卡片把 `n_draws` 改成上表，
另列 `draws_per_task`；`U2-LAUNCH-CARD.md:69-70` 的「smoke=8 draws」也改成
「n64 smoke=2 draws、s35 smoke=8 draws」。目前這不是結果程式 bug，而是發射／收割
驗收契約 bug；不修會誤報成功或失敗。

## 3. exclusive write、覆蓋與 logs

### 已閉合的部分

- 正式／smoke 的 `launch()` 在 `run.py:162-169` 先建立 base，再以
  `outdir.mkdir(exist_ok=False)` 建立 A 或 B；同 arm 重跑在 GPU 啟動前拒絕。
- `ucontrast1/run.py:42-46` 的 `write()` 和 `collector.py:146-150` 的結果寫入
  都使用 `open("x")`；preflight、result、verified 都不是無聲覆蓋。
- s35 mutant 用同一 reserve path；s35 sweep 每個 label 用
  `mkdir(..., exist_ok=False)`，結果檔仍由 `open("x")` 保護。

因此 exclusive write 不是單純「相信 OUT 不存在」；launcher 自己有拒絕邏輯。
代價是 partial／failed OUT 也會卡住同 arm 的直接重跑，需人工 quarantine 或新目錄。

### 必改 C：卡片宣稱的寫入面不成立

卡片 `U2-LAUNCH-CARD.md:55-58` 說只有 archive 的六個 OUT／calibration／logs，
並且「不寫 NFS home」。但六支 sbatch 都在 `:12` `cd "$HOME/Projects/lacot-u-frozen"`，
在 `:15-16` 用 `python -u` 啟動；沒有 `-B`，也沒有
`PYTHONDONTWRITEBYTECODE=1`。`run.py` 會 import u1/u2 Python modules，child rollout
也會 import repo modules，故在乾淨或 stale 的 frozen worktree 可能建立／更新
`__pycache__/*.pyc`。這是 archive 之外的 NFS home 寫入，即使不碰科學結果也違反卡片
的寫入聲明。

**改法二選一：**

1. 六支 sbatch 加 `export PYTHONDONTWRITEBYTECODE=1`（這會傳給 `run.py` 的 child
   subprocess），並可保留 `python -u`；或同時把 parent／child 都明確改成 `-B`。
2. 若主人允許 incidental bytecode，改卡片把 `__pycache__` 明列為例外，不再宣稱
   「別無其他寫入」。

## 4. 鏈序、select 路徑與誰在 enforce

### 路徑契約本身

這一段在「正確執行命令」前提下是對的：

- sweep 的四個結果在 `s35-calibration/sweep/{A,B-sig0.05,B-sig0.1,B-sig0.2}`。
- `s35_selection()`（`run.py:117-144`）要求四個絕對 result path、每個 result 的
  preflight／verified SHA、s35 checkpoint、pooled oracle@8 winner；select 寫出
  `sigma-selection.json`。
- formal s35 A/B 由 `run.py:181-185` 讀 default `--calibration-root`，並把
  selection SHA 寫入 preflight；B 的 σ 是 selection winner，不是硬寫 `.05`。
- n64 不依賴 select，B=.05 是本臂預註冊固定值。

### 鏈序沒有被 Slurm 強制

六支 sbatch 都沒有 `--dependency`。程式內的實際 gate 是：

- `s35-sweep`：`run.py:293` 要先有 s35 mutant gate；這一段會 fail closed。
- CPU select：`run.py:265-276` 要 mutant gate 和四個 verified sweep；這一段會 fail closed。
- s35 formal：A/B 會讀 s35 selection；這一段會 fail closed。
- **s35 smoke A（array index 0）**：`run.py:184` 明確寫 `not args.smoke`，所以 smoke
  時 A 不呼叫 `s35_selection()`；select 前可以直接建立 `smoke2-s35/A`。
- **s35 smoke B（index 1）**：在 `run.py:181-183` 的 σ 計算中讀 selection；select
  不存在時在 reserve OUT 前 BLOCKED。
- **n64 formal 和 s35 formal 都不檢查 smoke2 目錄的 verified receipt**。只要 n64
  的 input gate／s35 的 selection 到位，formal 可以早於 smoke 啟動。

因此「smoke 收割過後才 formal」目前是人肉順序，不是 code／Slurm gate。

### select 前提交 s35 smoke 的具體失敗形

同一個 `s35-smoke.sbatch --array=0-1` 可能出現 A 成功、B BLOCKED。A 留下
`smoke2-s35/A` 後，select 完成再整批重提，A 會因 `exist_ok=False` 被拒；B 才可能
成功。這是資料不覆蓋的好處，但也是鏈序錯誤後的 stranded partial state。

**改法：**

1. `run.py` 讓 s35 smoke 的 A、B 都先呼叫同一個 `s35_selection()`；或在 smoke
   sbatch 前置檢查 selection，避免半個 array 先跑。
2. 正式 n64/s35 在 `reserve_outdir()` 前驗對應 smoke2 的兩個 `verified.json`、
   preflight／result SHA、mode／stage／schema；或者明確放一個由收割官建立的 sentinel
   並由 launcher 驗證。若不想加 code，就必須把卡上的「smoke 後」降級為人肉提示，
   不能宣稱已 enforce。
3. select 的 root 入口統一：要嘛 select 也使用 `--calibration-root`，要嘛卡片與
   README 永遠只寫 `--outdir <calibration-root>`；不要留下目前的雙參數歧義。

## 5. smoke 盲區的獨立評估

卡片 `U2-LAUNCH-CARD.md:69-72` 的核心聲明是誠實的：
`run.py:190-191` 在 n64 smoke 令 `questions=None`，所以 source 是原 frozen
rollout；只有正式 n64 才選 `rollout_n64.py`，並由 `run.py:216-221` 記 derived
source SHA。這表示正式 n64 是該 derived／`exec(compile(...))` 路徑的首次 GPU 執行，
不能把 n64 smoke 的 PASS 解讀成覆蓋了 derived route。

### 緩解是否足夠

有實質緩解，但不足以把風險寫成已閉合：

- `rollout_n64.py:17-53` 會驗原 source PIN、三個唯一替換錨和 DERIVED_PIN。
- formal 完成後 `run.py:234-259` 會重驗題數、draw shape、題單、oracle、schema 尾三欄
  和 verified SHA。
- 題示的先前 CPU／derived 證據可降低純 Python 轉換與抽樣錯誤風險。
- 仍未覆蓋 GPU 上的首次 derived route、其 CUDA／MuJoCo import／執行交互、64 draws
  的長時間記憶體和只在最後寫檔的故障面。

所以本格判定為「聲明誠實、緩解部分足夠、殘餘風險中高」。若要閉合，需增加真正
走 derived route 的小規模 GPU smoke／gate；若不改 code，至少在 harvest 判讀中把它
列為正式首航風險，不把 n64-smoke PASS 升格成全路徑 PASS。

## 6. ETA × 死點與 mem 帳

採題示 lead／9/28 實測的 `6.9 s/rollout` 與 `MaxRSS≈1.22 GiB`，不重驗：

| stage（每臂） | rollouts | 推算 elapsed | `--time` 餘量 |
|---|---:|---:|---:|
| n64 smoke | 25×2=50 | 約 5.8 分 | 11h54m |
| s35 smoke／mutant／一個 sweep bin | 25×8=200 | 約 23 分 | 11h37m |
| s35 formal | 200×8=1600 | 約 3.07h（卡寫約 3.1h） | 約 3.9× |
| n64 formal | 50×64=3200 | 約 6.13h（卡寫約 6.2h） | 約 1.96× |

卡片的 formal array 各兩臂，jasmine 三張 4060Ti 足以讓兩臂並行。校準鏈則不能
把四格 sweep 都當成同時完成：mutant 約 23 分後，四格在三張卡上至少是兩波，理想
牆鐘約 46 分，還不含 queue／I/O；「四格同時 23 分」只適用至少四張可用 GPU。

memory ledger：每格申請 `--mem=16G`；這是 host RAM，不是 CUDA VRAM。相對已知
1600-rollout／臂 MaxRSS 1.22 GiB，表面 headroom 約 13.1×。n64 collector 保留
3200 rows／臂，若 row 記憶體近似線性，粗略約 2.44 GiB，仍約有 6.6× host-RAM
餘量；但這個 n64／derived／GPU 峰值沒有實測，不能把它寫成 OOM 已排除。GPU OOM
也不會由 `--mem=16G` 保證。

真正的死點是：rollout loop 完成後才在 `collector.py:134-150` 寫 result；逾時、OOM、
被搶占或 process crash 會留下 preflight／partial OUT 而沒有可收割 result。因為
`reserve_outdir()` 會拒絕既有 arm，沒有 checkpoint／續跑機制，n64 約 6 小時的失敗
成本是真實殘餘風險。這不是本次 launch card 已解決的問題，應在 ledger 中標成
「失敗即換新 OUT／人工調查」，不能自動重提原 arm。

## 必改清單（NO-GO 附改哪裡）

1. **改 `U2-LAUNCH-CARD.md:33-37`**：select 改成 jasmine 上的一行完整 python 命令，
   填實際 venv，刪除 zeldajr／jasmine 矛盾。
2. **改 `U2-LAUNCH-CARD.md:51-53,69-70`**：使用實際 `n_tasks`、`n_draws`、
   `draws_per_task` 表；n64 smoke 明寫 2 draws、25 題對／50 rows。
3. **改 `run.py` smoke／formal gate**：s35 smoke A/B 都驗 selection；formal 兩臂
   驗 smoke2 的兩個 verified receipts，或提供等價 sentinel／dependency。否則刪掉
   卡上的強制鏈序聲明。
4. **改六支 sbatch 的啟動環境**：加 `PYTHONDONTWRITEBYTECODE=1`（最簡單且會傳給
   child），或明確允許并記錄 frozen worktree 的 bytecode 寫入。
5. **改 `U2-LAUNCH-CARD.md:22-23` 附近**：明寫「n64 uses s33 checkpoint／seed=33，
   s35 uses s35 checkpoint／seed=35」，把目前的程式推論升格為發射單契約。

## 殘餘風險

- n64 smoke 即使 PASS，也沒有跑 derived route；正式 64 draws 是首次 GPU derived 執行。
- GPU CUDA／MuJoCo 的逐位元行為、實際 GPU memory peak、archive 寫入和 Slurm 啟動仍未
  由本查驗證。
- s35 粗掃每格只有 25 題；winner σ 有抽樣誤差，但選擇規則本身已由 gate 釘住。
- 逾時／OOM／搶占只會得到 partial OUT；目前沒有 checkpoint／續跑。
- `--calibration-root` 的固定 default 只在本靶絕對路徑不變時安全；換節點／換 archive
  mount 需要明確傳參並同步 analyze 路徑。

## 自我懷疑

- 我接受題示「ada-lite 12h、jasmine 三卡 idle、archive 5.2T、五個 OUT 不存在、
  frozen `validate_inputs()=READY`」為 lead 已驗事實，沒有在無 Slurm／SSH 的沙盒重驗。
- `.pyc` 是由「沒有 `-B`／沒有 `PYTHONDONTWRITEBYTECODE` 且在 `$HOME` 執行」推出的
  結構性風險；實際是否因 frozen tree 已有有效 bytecode 而沒有更新，需在 jasmine
  觀察，不能反過來當成已無寫入。
- ETA 使用題示的 6.9 秒／rollout 線性外推；沒有把 queue、模型載入、I/O、GPU clock
  波動或 derived route 的新峰值藏進 PASS。
- 卡片 select 的換行若作者原意只是排版折行，操作者仍可能手動修復；但在 fenced code
  block 中沒有 continuation marker，唯讀預檢必須按可貼上命令判定為不安全。
- 本報告審的是發射／執行契約，不重新判 ucontrast1 collector 的科學正確性，也不把
  9/28 結果當成 ucontrast2 的新結果。

STATUS: DONE
