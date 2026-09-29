# 真跑那份 code 的證據段（jasmine:/archive/cymaxwelllee/diag-g16；ルナ 2026-09-28 ssh 親撈）

⚠️ 預檢官注意：跑的 code 是 jasmine 本機部署副本（zeldajr 讀不到）。本檔為關鍵段落
快照；zeldajr 的 ~/Projects/lacot repo 份僅供參考、可能與部署份漂移。

## bcodec_gk_train.py argparse（部署份 :46-64）

```
--g {16,32} required / --k {16,32} required / --lam-s {0.0,1.0} required
--seg-len 4 / --steps 50000 / --batch 512 / --lr 3e-4 / --beta 0.25
--d None(沿用 group_dim=8) / --hidden 512 / --decay 0.99 / --dead-steps 500
--seed None / --split-seed 42 / --val-frac 0.1
--data-dir $OGBENCH_DATA_DIR 或 /home/cymaxwelllee/.ogbench/data
--out-dir <script dir> / --threads 8 / --log-every 1000
```
資料檔＝`os.path.join(args.data_dir, f"{bc.DATASET_NAME}.npz")`（:75），不存在會
印「⛔ 找不到資料檔」並退出（:76-77）。

## shat_probe.py argparse（部署份 :68-86）

```
--ckpt required / --lam1-ref-json required / --data-path DATA_PATH_DEFAULT
--split-seed 42（⛔ 必須跟 bcodec_v0 訓練同顆）/ --val-frac 0.1（同上）
--hidden 256[拍] / --steps 20000[拍] / --batch 512 / --lr 3e-4 / --seed 0
--bad-anchor-seed 20260913 / --suspect-ratio-threshold 2.0 / --code-info-margin 0.9
--tag required / --out-dir <script>/results
```
ckpt 載入讀 `cfg["G"], cfg["K"], cfg["D"]`（:98）；參照 json 經
`lam1_doc.get("metrics", lam1_doc)` 後讀 `recon_s_val`（:196-197）並
assert `lam1_metrics["G"]==G and ["K"]==K`（:198）。

## 參照 json 實測鍵（seedrep/s80001/results/G16K16_lam1p0.json 的 metrics）

G=16, K=16, D=8(有), lam_s=1.0, steps=50000, recon_a_val=.0324,
recon_s_val=.0520, any_group_collapsed=False, mean_active_codes=16.0
（G/K/recon_s_val 三鍵齊 ⇒ :197-198 兩關過得去）

## 33805 完成格效能錨

G16K16 lam0p0 s80001：wall=4599.9s、log 尾「=== done G16K16_lam0p0 (wall 4599.9s) ===」
＋saved json/pt 兩行。K32 wall≈3432s（train_seconds）。
TIMEOUT 四顆（task 8-11）log 全檔僅一行 slurmstepd CANCELLED ⇒ 無任何訓練輸出
（昨版無 -u、buffered）。

## jasmine 現況（02:1x UTC）

squeue 空；seedrep/s8000{1,2}/ 各有 {ckpt,results}/{K32,G16K16}_lam{0p0,1p0} 完成品、
無 G32K32 殘檔。
