#!/bin/bash
#SBATCH -J g17-smoke
#SBATCH -p ada-lite
#SBATCH -w jasmine
#SBATCH --cpus-per-task=8
#SBATCH -t 20
#SBATCH -o /archive/cymaxwelllee/diag-g16/seedrep/logs/smoke0928-%A.out
# F5 smoke（單 A 專用；預檢官指正後另做入口）：同 code path、--steps 1500（≥log-every 1000
# ⇒ 至少一個千步進度點，可量每千步秒數外推 50k ETA）、獨立 out-dir 不碰正式 seedrep/s8000x。
# 兼驗：CPU-only 下 --cpus-per-task=8 的實際吞吐、資料載入固定成本、G32 argparse choices、存檔段。
set -euo pipefail
OUT=/archive/cymaxwelllee/diag-g16/seedrep/smoke0928
mkdir -p $OUT
cd /archive/cymaxwelllee/diag-g16
PY=/archive/cymaxwelllee/LaCoT/.venv/bin/python3
exec $PY -u experiments/bcodec_v0/bcodec_gk_train.py --g 32 --k 32 --lam-s 0.0 --seed 80001 \
  --steps 1500 --data-dir /archive/cymaxwelllee/data/ogbench --out-dir $OUT
