#!/bin/bash
#SBATCH -J shat-obsonly3
#SBATCH -p ada-lite
#SBATCH -w jasmine
#SBATCH --cpus-per-task=8
#SBATCH -t 120
#SBATCH -o /archive/cymaxwelllee/diag-g16/seedrep/logs/shat4-%A.out
# G15 的 G32K32 格（2026-09-28 下午）：shat3（G16K16 格）同款、僅換受測物與參照——
# seedrep2 補跑完成後條件齊：ckpt=seedrep/s80001/ckpt/G32K32_lam0p0.pt、
# 參照=seedrep/s80001/results/G32K32_lam1p0.json（metrics.recon_s_val=.0242、G/K=32 供 assert）。
# 判準/口徑與 shat3 完全同（新基準、程式內建預註冊門檻）。
set -euo pipefail
D=/archive/cymaxwelllee/diag-g16
mkdir -p $D/shat3-results
cd $D
env CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= \
/archive/cymaxwelllee/LaCoT/.venv/bin/python3 -u experiments/bcodec_shat_probe/shat_probe.py \
  --ckpt /archive/cymaxwelllee/diag-g16/seedrep/s80001/ckpt/G32K32_lam0p0.pt \
  --lam1-ref-json /archive/cymaxwelllee/diag-g16/seedrep/s80001/results/G32K32_lam1p0.json \
  --tag bcodec_shat_gk32_obsonly_srep80001 \
  --data-path /archive/cymaxwelllee/data/ogbench/antmaze-medium-stitch-v0.npz \
  --out-dir $D/shat3-results
