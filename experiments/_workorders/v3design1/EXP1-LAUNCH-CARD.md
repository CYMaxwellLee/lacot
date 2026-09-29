# 實驗 1 發射單（草稿；F5 預檢靶）— Wφ/Aω 教師資格（正式資料）

_lead=ルナ 2026-09-28。mod-oracle 批過兩輪（S2 FAIL(8)→回鍋 M1-M3/G1-G2/L1-L2 修訂全落、24 tests）。_

- **在證明什麼**（DESIGN-v3 (d) 實驗 1、判準修訂版）：Wφ 看得出實際動作後果、Aω 能教同計畫
  動作。資格判準〔預註冊、L1/L2 修訂後〕：W held-out NMSE ≤ obs-only 0.9 倍且配對改善 CI
  下界>0（分 h=4/16/32、分子空間）；Aω action NMSE ≤ .1；PI 覆蓋與 validity 為健康報告欄
  不擋資格。shuffled-action/constant 對照隨批。
- **指令骨架**〔⓪官首航修正 v2：--out 改 /archive 絕對路徑（原 results/ 骨架與寫入面
  段自相矛盾、⓪官抓到）＋補 --threads 8（code 預設 1、8 核申請才不是多要）〕：
  `experiments/refine_v3/train_offline.py --domain {pointmaze|ant} --data <npz>
  --split-seed 1729 --seed {0,1,2} --threads 8
  --out /archive/cymaxwelllee/refine_v3/exp1/<domain>-s<seed>/`
  正式步數〔拍、預檢官攻〕：W/A 各 20000 steps、batch 64（對齊 shat probe 慣例的量級；
  toy 的 200 steps 僅接線驗證）。窗數上限〔①官修訂後 lead 裁〕：**300k cap（per-split
  min(cap,total)）**——非 train 全量（真 train 窗 ~35 萬級會被截）；300k 對資格判定
  統計力綽綽有餘、且 evaluate 無 batching 下 RAM 峰值受控（①官帳 ~20GB 下界、
  jasmine 128G/119G free 實測），夠用就停不追全量。
  smoke 的 scratch 定址：`/archive/cymaxwelllee/refine_v3/smoke/`（⓪官指出原「scratch」
  未定義——現定為 jasmine 本機此路徑）。sbatch 用單一 array 0-5 一次發（卡上原
  「兩 sbatch 分發」非硬序、取消）；--mem=48G；python -B 不寫 bytecode 進 NFS。
  收割口徑：readings 與 qualification_blocks 是 report.json 內欄位（非獨立檔）；
  ant 的 Aω=diagnostic 格不作資格（DESIGN 原文）；本輪 generated-candidate 兩 gate
  預期 missing（無真候選、非 final authorization）。
- **資料**：pointmaze=pointmaze-large-stitch-v0（manifest 9add335e…）；ant=antmaze-
  **medium**-stitch-v0（字典線/C1/C2/16pp 包絡全帳所在域；manifest 已補 b3fa34ee…、
  lead 於 jasmine 親算 SHA256——①官指出的卡片 stale 句已修）。large 筆保留不用。
- **機器**：jasmine CPU（資料在其 /archive、模型小 MLP、CPU-only code——本次真的是 CPU 活，
  sbatch ⛔ 不申請 gres、--cpus-per-task=8；餓核 insight 的正面應用）。
- **ETA**〔拍〕：(3W+3obs+3A)×2 domain×20k steps CPU ≈ 2-4 小時；先 pointmaze 後 ant 兩
  sbatch 分發，各 -t 240。
- **預期輸出**：每 domain×seed 一組 readings json＋checkpoint＋qualification_blocks；
  完成錨=readings 的 split hash 一致（三 seed 同）＋per-h per-子空間 NMSE 表齊。
- **寫入面**〔主人 2026-09-28 儲存裁示的首個應用〕：jasmine 本機
  `/archive/cymaxwelllee/refine_v3/exp1/`（⛔ 不寫 NFS 共享 home 的 repo results/；
  summary 級 json 收割後由 lead 拉回 workorders 入帳）。
- **依賴**：manifest（含 medium 補項）、venv、npz——jasmine 側 /archive 路徑 lead 已多次實證。
- **判讀規則**（跑完才用）：ant 若只 gait 過=只標 gait 安全診斷、xy/yaw 不過則停 goal
  selector；pointmaze Aω 不過則停 refine、selector 可續（DESIGN (d) 原文）。
- **F5 鏈**：本卡過①預檢 → smoke（單 seed 縮步數 2k、寫 scratch）→ 正式（寫 ckpt ⇒
  ルナ親手）→ ②點火 → ③哨兵（>30 分掛）→ ④收割 → 資格判讀。
