# EXP1 裝置資源四查（⓪裝置資源官）

日期：2026-09-28。範圍僅為 `EXP1-LAUNCH-CARD.md`、
`experiments/refine_v3/train_offline.py`、`lacot/refine_models.py`、
`lacot/refine_quality.py`。本查唯讀；未跑 GPU、`sbatch`、訓練或資料處理。

## 結論摘要

| 查項 | 結果 | 可否直接交① |
|---|---|---|
| 1. 裝置／CPU | **PASS（但 8 核不是最小申請）** | 可；見執行緒 caveat |
| 2. 產物寫入面／隱藏寫入 | **NO-GO** | 不可；發射單命令仍指向 `results/` |
| 3. NFS 紅線 | **NO-GO** | 不可；`results/` 是明列違例，`scratch` 未定址 |
| 4. 機器身分／GPU 儀器 | **NO-GO（提交腳本查不到）** | 不可；code 側無 GPU 儀器，但未驗到實際 sbatch |

## 1. code 實際裝置對 sbatch 申請：**PASS（資源相容；有多申請風險）**

已驗事實：受跑鏈是 CPU-only，沒有預設 `cuda` 的 device 參數雷。

- `train_offline.py:62` 執行 `torch.set_num_threads(args.threads)`；
  `:157` 的 `--threads` 預設是 `1`，`:159-160` 只允許 `1..8`。
- `refine_models.py:394-399` 的 `train_models` 明確檢查
  `record["state"].device.type != "cpu"` 即拒絕，錯誤訊息為
  `this first-batch entrypoint is CPU-only; GPU needs F5`。
- `train_offline.py:74` 只讀 `npz`，未把資料或模型搬到 CUDA；`:80`、`:114`
  也把執行摘要標為 `device: "cpu"`。
- `train_offline.py:35` checkpoint 載入固定 `map_location="cpu"`。
- 三個靶檔的命中結果：

  ```text
  $ rg -n 'torch\.device|cuda|mps|xpu|DataLoader|num_workers|set_num_interop_threads' \
      experiments/refine_v3/train_offline.py lacot/refine_models.py lacot/refine_quality.py
  experiments/refine_v3/train_offline.py:62:    torch.set_num_threads(args.threads)
  ```

  `torch.device`、`cuda`、`mps`、`xpu`、`DataLoader`、`num_workers`、
  `set_num_interop_threads` 均無命中。`refine_quality.py` 的
  `device=pred.device`（`:223`）及 `device=state.device`（`:411`、`:466`）
  只是跟隨呼叫者 tensor；本受跑入口由上述 CPU 檢查封住，並非 CUDA 預設。
- `train_models` 的模型迴圈在 `refine_models.py:411-434` 逐模型、逐 step
  直接取 tensor；沒有 DataLoader 或 worker fan-out。

對 `--cpus-per-task=8` 的判定：已知 code 管理的 intra-op threads 預設只有 1，
所以 8 核足夠，且按發射單現有命令沒有傳 `--threads 8` 時屬於多申請（不是少申請）。
若另行傳 `--threads 8`，上限剛好對齊 8 核。code 沒有設定 inter-op thread pool，
因此無法僅靠靜態檔案保證作業系統層的所有瞬時執行緒數；這是 caveat，不是 GPU
觸發證據。

## 2. 存放位置與隱藏寫入：**NO-GO**

發射單互相矛盾：

- `EXP1-LAUNCH-CARD.md:23-25` 明定正式寫入面是 jasmine 本機
  `/archive/cymaxwelllee/refine_v3/exp1/`，且禁止 repo `results/`。
- 但同卡 `:9-10` 的指令骨架把 `--out` 寫成
  `results/refine_v3/exp1/<domain>-s<seed>/`。這是相對 repo/home 的產物路徑，
  若照此骨架發射即違反主人裁示；必須以 `/archive/...` 的絕對路徑取代後才可放行。

受跑 code 的寫入面只有呼叫者傳入的 `args.out`：

- `train_offline.py:122-130` 建立 `out`，寫 `report.json`、`metadata.json`、
  `oracle.pt`；`:149` 顯示 `--out` 是 required，沒有預設 `out`。
- `train_offline.py:26-31` 的 `torch.save` 也只吃呼叫者傳入的 checkpoint path。
- `refine_models.py:121-126`、`:143-170` 的檔案操作是讀 dataset／讀
  `DATA-MANIFEST.json`；`refine_quality.py` 沒有檔案寫入。
- 受查三檔沒有 `cache`、`tmp`、`temp`、`TemporaryFile`、`DataLoader` 或其他
  顯式隱藏輸出路徑命中。Python import 可能由執行環境產生 `__pycache__`，不在
  此 code 的顯式寫入契約內，未在本查中假設它不存在。

因此：若 `--out=/archive/cymaxwelllee/refine_v3/exp1/...`，code 寫入面符合裁示；
但目前發射單仍同時保留 `results/...` 骨架，查項整體 NO-GO。

## 3. NFS 紅線：**NO-GO**

按主人規則分類發射單中可辨識的路徑：

| 發射單文字 | 類別 | 讀／寫判定 |
|---|---|---|
| `experiments/refine_v3/train_offline.py`（`:9`） | repo／NFS home 上的 code | 讀 code 可；非產物寫入 |
| `results/refine_v3/exp1/...`（`:10`、`:24` 所稱 repo results） | repo／NFS home | **產物寫入禁止；明確違例** |
| `/archive/cymaxwelllee/refine_v3/exp1/`（`:23-24`） | jasmine 本機 `/archive` | 產物讀寫允許 |
| `DATA-MANIFEST.json`（`:15`、`:26`） | repo／NFS home | code 只讀 manifest，允許 |
| `venv`、`npz`、jasmine 側 `/archive`（`:26`） | 具體絕對路徑未給全 | 依賴可查讀；無法驗證實際 mount／路徑 |
| `scratch`（`:29`） | 未定義位置 | smoke 產物寫入位置查不到，不能視為本機 |

code 方面已驗 `DATA_MANIFEST` 是由 `refine_models.py:26` 建出的 repo 路徑，
並在 `:146` 讀取；dataset 由 `:123-126`、`:152` 讀取。只有
`train_offline.py:122-130` 寫 `args.out`。所以 code 沒有額外把重讀寫產物寫回
repo/NFS 的隱藏路徑，但發射命令的 `results/` 與 smoke 的未定址 `scratch` 仍使本查
不能放行。

## 4. 機器身分與 GPU 儀器：**NO-GO（實際提交腳本缺失）**

- 發射單 `EXP1-LAUNCH-CARD.md:17-18` 指定 jasmine CPU、`⛔ 不申請 gres`、
  `--cpus-per-task=8`；此與 code 的 CPU-only gate 相符。
- 在發射單與三個受查 code 的 bounded search 中，`nvidia-smi`、`nvidia`、
  `cuda`、GPU 初始化／儀器呼叫均無命中；因此 code 側沒有要求在 jasmine 跑 GPU
  儀器。
- 靶範圍內沒有 EXP1 的 `.sbatch`／提交腳本，只有卡上文字「兩 sbatch 分發」
  （`:19-20`）。所以無法驗證實際 sbatch 是否殘留 `nvidia-smi` 或其他 GPU
  儀器，也無法從本靶確認 zeldajr=AMD 的提交機沒有被誤用。不得以「未提供」冒充
  「已驗無」。提交腳本補齊並通過同一 grep 前，維持 NO-GO。

## 給①預檢對抗官的已驗事實清單

1. 受跑入口 `train_offline.py` 沒有 `--device` 參數；CPU 路徑由
   `torch.set_num_threads`、CPU tensor、`train_models` 的 CPU gate 組成，沒有
   CUDA 預設雷。
2. 目前入口預設 `--threads 1`、上限 8；沒有 DataLoader workers。故申請 8 核
   足夠但按現有命令偏多；未設定 inter-op pool，總 OS thread 數不能由靜態碼完全封頂。
3. 受跑 code 的正式寫入只有 `args.out/report.json`、`metadata.json`、`oracle.pt`；
   沒有顯式 cache/tmp/default-out 寫入。輸入 dataset 與 manifest 是讀取。
4. 發射單 `:9-10` 的 `results/...` 與 `:23-25` 的本機 `/archive/...` 互斥；
   這是目前真正的存放/NFS NO-GO，不能照 `results/` 骨架發射。
5. `scratch` 沒有位置定義；實際 EXP1 sbatch 沒在靶內，GPU 儀器與 zeldajr 提交
   端禁用狀態尚未驗證。

## 自我懷疑

- 我沒有實際執行 `sbatch`、登入 jasmine/zeldajr、檢查 mount table，故不能把
  `/archive` 的本機性或提交腳本內容當成已驗事實。
- `torch.set_num_threads(1)` 只直接約束 intra-op；因 code 沒有
  `set_num_interop_threads`，不能宣稱整個程序永遠只用一條 OS thread。
- `scratch` 可能由未提供的 wrapper 解析成本機 scratch，也可能落到 NFS；靶內無法
  判定，故刻意保留 NO-GO。
- `refine_quality.py` 的部分工具會跟隨輸入 tensor device；本結論只適用於本卡指定
  的 `train_offline.py` 受跑鏈，不外推到未指定的 GPU caller。

STATUS: DONE
