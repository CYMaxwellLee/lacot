# 發射單 ×2（patchrun1；2026-09-28）— F5 預檢靶

_lead=ルナ。過夜隊列 33805/33809 的補洞。判準沿用原工單（不開新刀），僅修執行層。_

## 單 A：seedrep2 — G17 高容量格補跑（G32K32 ×4）

- **在證明什麼**：G17「λ差±SD」12 顆中的高容量四顆（G32K32 兩 λ × seeds 80001/80002）。
  9/12 空白格②的方向翻轉（λs 高容量−）指著這格 —— 是 12 顆裡最承重的。判準=同 33805
  原工單（λ1−λ0 配對差、per-seed），本單不改判準只補執行。
- **指令**：`sbatch data/fleet-runs/patchrun1/seedrep2.sbatch`（提交機=zeldajr、跑=jasmine）
- **與 33805 的 diff（僅四處）**：-J、`-t 90→300`、-o rep-→rep2-、`--array=8-11`、python 加 `-u`。
  CELLS/SEEDS 映射與輸出路徑逐字不動。
- **預期輸出**：`/archive/cymaxwelllee/diag-g16/seedrep/s8000{1,2}/results/G32K32_lam{0p0,1p0}.json`
  ×4 ＋ 同名 ckpt/*.pt ×4。JSON schema 同 G16K16 完成格（metrics.{G,K,recon_a_val,recon_s_val,
  any_group_collapsed,...}、config.steps=50000）。log 有 log-every=1000 的進度行（-u 保證可見）。
- **預期計數**：4 json + 4 pt；每 json metrics.steps 反映 50000。
- **ETA**：⚠️ 未知格 —— G16K16 實測 wall 4599.9s；G32K32 昨天 90 分零輸出（無 -u、buffered
  丟光）＝速度未量過。**smoke 先行**：`--steps 500` 同 code path 量每千步秒數 → 外推 50k；
  外推 >280 分則停下改單（不硬塞 300 分）。
- **寫入面**：seedrep/s8000{1,2}/{ckpt,results}/G32K32_*（G32K32 檔名既有目錄中不存在、
  不蓋 K32/G16K16 完成品）＋ logs/rep2-*（新 pattern 不蓋 rep-*）。
- **依賴**：diag-g16 部署 code、data npz、venv —— 三者 33805 完成的 8 顆已實證可用。
- **lead 已驗事實**〔ルナ 2026-09-28 02:1x UTC ssh 親驗〕：jasmine squeue 空；
  s80001/s80002 的 results/ 現況=K32+G16K16 各兩 λ（無 G32K32 殘檔）；TIMEOUT log 僅
  slurmstepd CANCELLED 一行=無 partial 產物；`--steps` argparse 存在（default 50000）。

## 單 B：shat3 — G15 修正重排（G16K16 格、新基準）

- **在證明什麼**：G15 obs-only 對照臂 —— λ0 的碼在 ŝ 預測上有沒有超出狀態的增量資訊。
  判準=程式內建預註冊（code_info_margin=0.9、suspect_ratio_threshold=2.0 預設沿用、
  ratio_to_lambda1≤1.5 級距），本單不改判準。
- **死因帳（33809）**：兩格訓練段正常跑完（19999 步、mse 收斂），死在 `shat_probe.py:197`
  讀參照 json 的 `recon_s_val` —— 舊 summary 格式沒有這個量。
- **修法**：參照與受測 ckpt 都換 seedrep 新基準（s80001 同切分同 seed 配對）。附帶修掉
  33809 的口徑混淆（舊版=舊 ckpt 配新切分 probe 混搭）。
- **指令**：`sbatch data/fleet-runs/patchrun1/shat3.sbatch`
- **預期輸出**：`diag-g16/shat3-results/` 下 tag=bcodec_shat_gk16_obsonly_srep80001 的
  summary json＋stdout「=== 結果 ===」段（err_real/ref_a/err_bad/err_obs_only 四行＋判定行）。
- **預期計數**：1 json；stdout 判定行=四種 verdict 之一（不預測哪種）。
- **ETA**：33809 實測訓練段 3.5-6.5 分 CPU ⇒ 全程 ~10 分內；-t 120 裕量大。
- **寫入面**：shat3-results/（新目錄）＋ logs/shat3-*。零覆蓋。
- **依賴＋lead 已驗事實**：參照 json 的 metrics.recon_s_val=.0520、metrics.G=16、metrics.K=16
  〔ルナ親撈〕；probe :198 assert 需要的 G/K 齊；probe 的 split-seed/val-frac 預設 42/0.1 與
  seedrep 訓練側預設一致〔兩邊 argparse 親讀〕；資料檔與切分函數對齊已由 33809 訓練段實證
  （它死在訓練之後）。
- **未驗格（標明）**：新 seedrep ckpt 內 cfg dict 結構（probe :98 讀 cfg["G"/"K"/"D"]）——
  低風險（同一支 bcodec_gk_train.py 存的、33809 舊 ckpt 同構跑通過），smoke 即驗。
- **G32K32 格不在本單**：等單 A 完成後同款再排。

## F5 鏈位置

本卡=①預檢對抗官的靶。GO 後：smoke（單 A `--steps 500`＋單 B 直接全跑=本身 10 分級）
→ 正式發射（寫 ckpt ⇒ ルナ親手 sbatch）→ ②點火 → ③哨兵（單 A 掛、單 B 免）→ ④收割。
