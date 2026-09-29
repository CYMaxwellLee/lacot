#!/bin/bash
#SBATCH -J shat-obsonly2
#SBATCH -p ada-lite
#SBATCH -w jasmine
#SBATCH --cpus-per-task=8
#SBATCH -t 120
#SBATCH -o /archive/cymaxwelllee/diag-g16/seedrep/logs/shat3-%A.out
# G15 修正重排（2026-09-28）：33809 兩格死因 = shat_probe.py:197 KeyError 'recon_s_val'
# —— 舊參照（bcodec_v0/results/*_lam1p0_summary.json）根本沒有「λ1 自帶頭 ŝ 誤差」欄。
# 修法＝參照與受測 ckpt 都換 G17 seedrep 新基準（同一切分、同 seed 配對，s80001）：
#   ckpt   = seedrep/s80001/ckpt/G16K16_lam0p0.pt          （9/27 夜剛出爐）
#   參照   = seedrep/s80001/results/G16K16_lam1p0.json      （metrics.recon_s_val=.0520 實測在、帶 G/K 供 :198 assert）
# 順帶把 33809 的口徑混淆修掉（舊版=舊 ckpt 配新切分 probe）。
# 口徑：全新基準（新軌跡切分），與 9/26 segment 切分舊數字不直接比（33809 原註記沿用）。
# 本單只跑 G16K16 格；G32K32 格等 seedrep2 補跑完成後同款再排。
# split-seed/val-frac 用程式預設 42/0.1 ＝ seedrep 訓練同款（--split-seed 訓練側也是預設 42）。
# v2（F5 預檢修正）：加 set -euo pipefail；部署版 probe 不寫 _probe.npz（ssh grep 驗過）⇒ 預期計數 1 json 不變。
set -euo pipefail
D=/archive/cymaxwelllee/diag-g16
mkdir -p $D/shat3-results
cd $D
env CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= \
/archive/cymaxwelllee/LaCoT/.venv/bin/python3 -u experiments/bcodec_shat_probe/shat_probe.py \
  --ckpt /archive/cymaxwelllee/diag-g16/seedrep/s80001/ckpt/G16K16_lam0p0.pt \
  --lam1-ref-json /archive/cymaxwelllee/diag-g16/seedrep/s80001/results/G16K16_lam1p0.json \
  --tag bcodec_shat_gk16_obsonly_srep80001 \
  --data-path /archive/cymaxwelllee/data/ogbench/antmaze-medium-stitch-v0.npz \
  --out-dir $D/shat3-results
