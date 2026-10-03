把發射單 v8 的改動與 S2 兩條審查線的必改項落進現有 detour harness，讓 v8 的 smoke（E1 首方向表＋15 支）能上機；判準＝`detour_selftest.py` 全 PASS（含 float64 真環境列級收割、O3／F3 目標欄、E1 正負對照）、舊三個 selftest 原文不動 PASS、PB 既有 79 檔 hash 不變，每個新閘的殺手輸入實際 FAIL。

本機驗收與限制見 [IMPL-NOTES-detour.md](IMPL-NOTES-detour.md)。本棒不跑 GPU、不連 jasmine、不送 sbatch、不 commit；兩個 detour release flags 維持 False。CPU 短程與 mock 不是正式成績，GPU E1／smoke／formal 仍 UNVERIFIED-GPU。

## 協定與本機驗證

`detour-u-v8` 正式題單 408 支：O3／F3／OF 各 120；Q-C／Q-O3／Q-F3 共 48。smoke 是 main 三臂各 task4／5／2 一題，加 Q 三臂各 task1／3 一題，共 15 支。episode 從原機器題庫產生，不手抄；`detour_plan.py --out NEW_PATH` 採 exclusive create。

O3 每 chunk 編碼往前 3 步；F3 同規則供近目標、每 chunk 真呼叫一次 `flow.sample`；OF 保留整條路＋最終 goal 的 v5 O 行為。逐 chunk 記 w、target_xy、cond_target_xy、h，O3 另記 tau_endpoint／tau_dtype／tau hash。goal 尾段只用 runtime 資格，沒有跨 draw 快取。F3 解碼末距只描述，不能拿來拒收。

```bash
.venv/bin/python -B experiments/_workorders/breakthrough-probe/detour_selftest.py
```

自測含真 M9／s33 EMA／迷宮的 CPU H=40 完整 `validate_row` 收割，以及完整 E1 的 CPU 測試表，與凍結先量版共同 205 靜態格須 0 翻格。CPU 表只在測試記憶體中產生；測試明示 patch GPU／hostname 入口閘，不能作正式 E1 或成績。契約、trapset、`detour-horizon-ruler.json`、`detour-e1-v6-h3.json` 都是來源逐 byte 副本並 pin SHA；未改來源量尺或路線。

## 給 lead 的凍結與發射資訊（本棒未執行）

- 凍結樹須保留 repo 相對結構：PB 新舊依賴、原 M9、collector、`lacot/` 套件及 `results/*.pt`（包含 runtime.CKPT 指定的 s33 checkpoint）。只複製 PB 目錄無法載入模型。
- smoke 與 formal 必須使用**同一凍結根路徑**、相同 interpreter、同一 `DETOUR_DATASET_DIR`。provenance 比較包括 source／checkpoint／dataset 的絕對路徑；內容相同但換路徑仍會拒收。執行前須在 jasmine 本機準備完整 dataset 與相依套件。
- 真入口要求 `socket.gethostname().split('.')[0] == 'jasmine'`、確定性 CUDA 設定、release flag。實際 Slurm hostname 尚未在本棒查驗。
- **`sbatch --output` 的父目錄須在提交前建立**：`/archive/cymaxwelllee/breakthrough1/detour-u/`。Slurm 開 stdout 早於 Python 建 outdir，腳本內建目錄來不及。
- 兩份 sbatch 第一個位置參數是凍結根目錄；`DETOUR_PYTHON` 與 `DETOUR_DATASET_DIR` 指定 jasmine 本機 interpreter／資料。
- **時限用命令列 `sbatch --time` 覆蓋，⛔ 不改檔**。ETA 先按各 task／arm 的實測「每步秒數 × 1000 × 支數」加總，再留餘裕；smoke 樣本可能提早成功，不能把按短樣本時間外推的 `proposed_time_limit_seconds` 當保證。`.sbatch` 與 `detour_selftest.py` 屬於 code hash；smoke 產表後改檔會使 E1 表失效。
- smoke 前在凍結樹的 `detour-flags.json` 設 `SMOKE_ENABLED=true`、`PRODUCTION_READY=false`；smoke 完整跑完且 E1／raw／receipt 驗過後，才翻 `PRODUCTION_READY=true` 供 formal 使用。本棒本機 flags 仍都是 false。
- smoke 中途失敗需重送時，先將 `/archive/cymaxwelllee/breakthrough1/detour-u/smoke` **改名**為未使用的保留目錄（例如 `smoke-failed-YYYYMMDD-HHMMSS`），確認原 `smoke` 路徑已空出，再重送 smoke。**不得刪除失敗目錄**；保留 E1、raw、checkpoint 與 shards 供查核。新 smoke 會重新產表、重新跑 15 支，不能把失敗樣本混作完成的 smoke。

smoke 先生成 `e1-table-v8.json` 與 SHA，再跑 15 支。`cells` 僅含 w≠g 的 O3 靜態格；`of_cells` 保留 OF 的 v5 描述表。門檻從量尺的 real cos_4 取 p10，不寫死 0.814。`cpu_comparison` 列出與 CPU 先量版共同靜態格的 PASS／FAIL 差異；`critical_cells` 列路線上首步背離格。w=g 尾段的實際 noisy goal 只在各 rollout 的 runtime chunk 記錄資格。

「首方向合格」不代表整段可執行。檢查器記錄直衝近目標切牆角仍 PASS 的已知限制；淨空與 Fréchet 留作描述。成功不受 E1 資格篩掉，主判不再刪整題或整層分母。

formal 讀 `/archive/cymaxwelllee/breakthrough1/detour-u/smoke/e1-table-v8.json`。續跑使用新 `DETOUR_OUTDIR=.../formal-r2` 與 `DETOUR_RESUME_FROM=.../formal`；先驗原 receipt、表與 shards，再重用完成列。跨 process GPU 重編碼 u hash 是否完全相同仍待 smoke→formal 真機驗證；不放寬 hash 比較。

## 獨立收割

必須使用相同凍結程式樹與 E1 表：

```bash
python -B experiments/_workorders/breakthrough-probe/harvest_detour.py \
  --artifact /archive/cymaxwelllee/breakthrough1/detour-u/formal/raw.json \
  --e1-table /archive/cymaxwelllee/breakthrough1/detour-u/formal/e1-table-v8.json \
  --out /archive/cymaxwelllee/breakthrough1/detour-u/formal/harvest-detour.json
```

收割以 float64 重建 trace，initial／goal 與 goal τ 使用記錄的原 dtype；不靠 tolerance 掩蓋精度損失。三層各自輸出 O3／F3 落格、互斥 R1–R5 逐字判語與 Q 加註；失敗三分與順從度只描述；OF 報成功率及同題同 draw 的 d_h。smoke 不套正式四 draw 題級判斷。
