# EXP1 預檢對抗報告（F5 ①）

日期：2026-09-28。靶為修正版 `EXP1-LAUNCH-CARD.md`、`exp1.sbatch`、
`EXP1-DEVICE-CHECK.md` 與 `experiments/refine_v3/train_offline.py`。本查唯讀；未跑
訓練、未提交 `sbatch`、未讀取正式 NPZ。⓪官報告列出的 CPU-only、無 `--device`、顯式
輸出只有 `args.out` 與 `/archive` 紅線，作為已驗事實沿用。

**VERDICT: NO-GO**

旗標、六格 array 展開、`--out`、`--threads 8`、輸出檔與 stale 防覆蓋已對上；
但 ant 的正式記憶體峰值沒有 jasmine 節點對照，`300000` 也未證明等於 train 全量，
且 Slurm log 目錄／目標 venv 的節點端前置條件未驗。`-t 240` 暫不砍；先補下列
ASK 與 2k smoke 才能改 GO。

## 九格

| 格 | 判定 | 證據與攻擊結論 |
|---|---|---|
| A1 逐 token 對 argparse | **PASS（附依賴 ASK）** | `exp1.sbatch:31-33` 的 `--domain --data --split-seed --seed --threads --steps --max-windows --hidden --out` 全在 `train_offline.py:146-157` 宣告。`domain` 是 `pointmaze/ant` choices；`threads=8` 通過 `1..8`；`steps=20000` 為正整數；`max-windows=300000` 為正整數；`hidden=256` 是可交給 MLP 的整數；未傳的 `batch-size`、`t-cap` 使用合法預設 `64/128`。兩個正式 data basename 與 manifest 的 pointmaze/ant-medium 項相符。`bash -n exp1.sbatch` PASS。`PY` 可執行與 data 檔存在仍只在節點上的腳本 runtime `test -x/test -f` 驗。 |
| A2 array 映射 | **PASS** | 腳本公式是 `seed=id%3`、`domain=id/3`（`exp1.sbatch:12,16-20`），手動展開為：`0→(0,pointmaze)`、`1→(1,pointmaze)`、`2→(2,pointmaze)`、`3→(0,ant)`、`4→(1,ant)`、`5→(2,ant)`；每格的 `OUT=/archive/.../exp1/<domain>-s<seed>` 不碰撞，與卡上 2 domain × 3 seed 對齊。 |
| A3 `-m` module import | **ASK（非 marker 直接 NO-GO）** | `experiments/__init__.py` 與 `experiments/refine_v3/__init__.py` 都不存在；但 Python 3 namespace package 是合法的，從 repo root 做 `importlib.util.find_spec('experiments.refine_v3.train_offline')` 已解析到 `experiments/refine_v3/train_offline.py`，所以「無 `__init__.py` 必然炸」不成立。完整 `python3 -B -m ... --help` 在本機只因 `/usr/bin/python3` 缺 `torch` 而停在 import，未進 `run()`；不能冒充 jasmine 目標 venv 已驗。須用腳本指定的 `/archive/cymaxwelllee/LaCoT/.venv/bin/python` 做同一 `--help` 或 smoke，確認依賴與 `cd /home/.../Projects/lacot` 的實際 module import。 |
| B1 CPU 時與 ETA | **ASK（暫不砍步數）** | 已存 toy evidence：200 steps、hidden64、1 thread、三個 model group 的 elapsed 為 pointmaze `2.455–2.489 s`、ant `3.679–3.735 s`。只按 steps 線性外推，20k 是 `245–374 s`（約 4.1–6.2 min/job）；hidden 256 的 MLP 參數量約為 hidden64 的 4.6–6.2 倍，僅作量級推算約 20–40 min/job，真資料建窗、full-batch calibration/evaluate 另加未知成本。`train_models` 確實對 world/obs-only/controller 各跑 `steps`，每一組 ensemble=3（`refine_models.py:404-407,411-433`），即每 job 3W+3obs+3A。`-t 240` 是每 task 240 分鐘，從現有 evidence 看不必縮；正式 GO 仍需 2k real-data smoke 的 wall/peak RSS。 |
| B2 300k windows / RAM | **NO-GO（待 RAM ASK）** | `build_records` 在每個 split 各做 `min(max_windows,total)`（`refine_models.py:217-261`），不是全程共用的 300k。對 ant `D=29,A=8,T_CAP=128`，每窗僅計已成形的 float32/bool/int64 record：`4D + 4(32A) + 4(32D) + 32 + 4(2T_CAP) + 3×8 = 5,932 B`，300k 為 `1.780 GB = 1.657 GiB`；三個 split 全達 cap 的 record lower bound 已是 `5.339 GB = 4.972 GiB`，尚未算 NPZ、Python per-window list、`np.stack`/`torch.tensor` 暫存與 allocator。更危險的是 `OfflineCalibrator.fit`／`evaluate` 沒有 batching：h=32 的單一 ensemble prediction `[3,B,32,29]` 在 B=300k 就約 `3.341 GB = 3.111 GiB`；evaluate 同時保留 world/obs-only/shuffled 三份約 `9.334 GiB`，還有 residual/errors。已有工作單只給約 `3000×197≈59.1 萬` raw candidates 的量級；依 `split_episodes` 的 1/5 calibration、1/5 test、3/5 train（`refine_models.py:182-188`），`300000` 很可能仍截斷 train，不能把它寫成「train 全量」。jasmine 的 `RealMemory`/partition memory 查詢因 Slurm socket `Operation not permitted` 無法取得，腳本也沒有 `--mem`；在 RAM 未對照前不放行。 |
| B3 walltime / queue / log 前置 | **ASK** | `#SBATCH -p ada-lite -w jasmine --cpus-per-task=8 -t 240 --array=0-5`（`exp1.sbatch:2-8`）與 CPU-only code 相容；但 queue 狀態未驗，且 card 說「先 pointmaze 後 ant、兩 sbatch」，腳本本身卻是一個 `0-5` array。若順序是硬條件，應由 lead 先送 `--array=0-2`，完成後再送 `--array=3-5`。另，Slurm `-o /archive/.../logs/exp1-%A_%a.out` 會在 script body 前開檔；`mkdir -p /archive/.../logs` 在 `exp1.sbatch:25` 太晚，log parent 必須在提交前已存在。`/archive` 5.2T 依題示沿用，不重查 df；這不替代 log directory 與 queue ASK。 |
| C1 產物清單 | **PASS-with-note** | `run()` 明確寫 `report.json`、`metadata.json`（`train_offline.py:122-129`），再寫 `oracle.pt`（`:130`）。`report.json` 內含 `readings`、`qualification_blocks`、`split_hash` 等 summary；沒有獨立的 `readings.json` 或 `qualification_blocks.json`。卡上的「readings json＋checkpoint＋qualification_blocks」只有在收割器把前兩欄視為 report 內欄位時才對；若收割器要求三個獨立檔，需先改口徑。 |
| C2 stale / smoke | **PASS stale；NO-GO smoke 尚缺** | `exp1.sbatch:22-24` 對非空 `OUT` 以 exit 24 停止；程式又以 open mode `"x"` 寫 report/metadata，故不覆蓋既有 evidence。代價是任何中途失敗留下的非空 partial OUT 也會拒絕重跑，必須人工 quarantine/換新 seed-output；這是安全行為，不可當完成。卡只說 smoke `--steps 2000` 另發，靶內沒有 smoke sbatch。建議 lead 在 jasmine 預建 logs/smoke 後，以新目錄跑：`mkdir -p /archive/cymaxwelllee/refine_v3/logs /archive/cymaxwelllee/refine_v3/smoke && /usr/bin/time -v /archive/cymaxwelllee/LaCoT/.venv/bin/python -B -u -m experiments.refine_v3.train_offline --domain pointmaze --data /archive/cymaxwelllee/data/ogbench/pointmaze-large-stitch-v0.npz --split-seed 1729 --seed 0 --threads 8 --steps 2000 --max-windows 300000 --hidden 256 --out /archive/cymaxwelllee/refine_v3/smoke/pointmaze-s0`；再以 ant/seed0 重複一次才覆蓋 ant RAM。此命令由 lead 執行，本查未執行。 |
| C3 qualification / 預註冊 | **PASS-with-ASK** | `experiments/refine_v3/README.md:12-22` 與卡 `:5-8` 對上 L1/L2 修訂：W held-out NMSE `≤0.9×obs-only`、每 h/子空間 paired-CI 下界 `>0`、Aω action NMSE `≤0.1` 留作資格；PI 85–95% 與 data validity ≥90% 改為 health readings、不擋資格。`qualification_reasons`（`refine_quality.py:300-323`）也移除了兩個 health gate，但保留 generated-candidate validity/nominal+alternative coverage 各 ≥80% 的授權 gate；本入口 `evaluate()` 將兩者設為 `None`（`refine_quality.py:254-255`），所以本次 report 的 `qualification_blocks` 預期仍會列「generated ... missing」，不能宣稱 final authorization。另 `require_controller` 參數目前未影響 Aω check；這與「ant A 僅 diagnostic」的實作註記需由 lead 明定口徑。 |

## 覆蓋面、依賴與已修帳

- 覆蓋面是 6 array task：每個 domain/seed 都跑 Wφ ensemble=3、obs-only ensemble=3、Aω/controller ensemble=3；`--seed` 改模型/評估 RNG，`--split-seed 1729` 固定三 seed 的 episode split，完成錨可用三份 `split_hash` 比對。ant 的 Aω 仍是 diagnostic，不能由「有輸出」推成 ant R>0 deployable。
- 三個首查 NO-GO 的修正痕跡已找到：正式 `--out` 為 `/archive/.../exp1/...`（`exp1.sbatch:21,33`）、`--threads 8` 已傳、smoke 路徑已在卡定為 `/archive/cymaxwelllee/refine_v3/smoke/`，且 `--max-windows 300000`、`--hidden 256` 已進正式命令（`exp1.sbatch:29-33`）。正式 sbatch 本身不寫 smoke，故不能把卡上定址當成 smoke 已存在。
- manifest 實際已有 `antmaze-medium-stitch-v0.npz` 與 SHA256 `b3fa34...d71e99`、`obs_dim=29/act_dim=8`（`DATA-MANIFEST.json:16-21`），但 launch card `:19-20` 仍寫「manifest 目前只有 large」。這是文件 stale；以 code loader 讀到的 manifest 為執行依據前，請 lead 更新卡或明確鎖定 manifest 版本。manifest 自身仍標示 local copy、未對 upstream 驗證。
- code 的正式寫入面與 ⓪官事實一致：輸入 NPZ/manifest 讀取，輸出只經 `args.out`；但 sbatch 未加 `-B`，Python import 可能在 repo source 旁產生/更新 `__pycache__`。若「不寫 NFS」包含 incidental bytecode，請在節點命令加 `PYTHONDONTWRITEBYTECODE=1` 或 `-B`，目前不把它假報成零寫入。

## ASK（改 GO 前）

1. lead 提供 jasmine `RealMemory`、ada-lite 的可用/預設 memory 與 2k real-data smoke 的 `/usr/bin/time -v` peak RSS；以 ant 為準，並決定 `300000` 是有意的 cap 還是需依 train 實際窗數改成全量。若保留 cap，卡上的「train 全量」文字要改。
2. 提交前建立 `/archive/cymaxwelllee/refine_v3/logs/`；用目標 `/archive/.../.venv/bin/python -B -m ... --help` 或上述 smoke 證明 torch、namespace module、依賴版本均可用。確認是否採 pointmaze/ant 兩段 array 提交。
3. 確認 report 內 `readings`/`qualification_blocks` 是收割契約，並確認 ant Aω 是要作資格格還是只作 diagnostic；同步修正卡上 medium manifest 的舊句。

## 自我懷疑

- 我沒有登入 jasmine、沒有讀 mount table、沒有計算正式 NPZ 的真實 episode/窗數、沒有跑 `sbatch` 或訓練；RAM 數字是源碼形狀的 lower bound，不是峰值 RSS。
- toy elapsed 來自 hidden64/1 thread/小資料；hidden256、8 threads、正式 data 的速度不能由線性外推保證，20–40 分鐘只是排程量級，不是測量。
- `find_spec` 只驗 namespace 的路徑解析；本機完整 import 因缺 `torch` 未完成，故沒有把它寫成目標 venv PASS。
- manifest 的 medium hash 是檔案內既存記錄，我沒有重算或讀 `/archive`；卡與 manifest 的文字矛盾仍需 lead 收口。
- 本報告把 queue、`/archive` 5.2T 與 ⓪官 CPU/GPU 事實按題示沿用；Slurm 控制器拒絕本地 socket，不能從本工作站補出 jasmine 的即時狀態。

STATUS: DONE
