#!/bin/bash
#SBATCH --job-name=breakthrough-move3
#SBATCH --partition=ada-lite
#SBATCH --nodelist=jasmine
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/archive/cymaxwelllee/breakthrough1/move3-%j.out
# Release remains closed. v6.3 fixes this allocation at 120 minutes; GPU ETA is pending.
# vFinal must set measured time/quota and prepare the archive directory before sbatch.
set -euo pipefail
cd /home/cymaxwelllee/Projects/lacot
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONHASHSEED=0
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OGBENCH_DATA_DIR=/archive/cymaxwelllee/data/ogbench MUJOCO_GL=osmesa
probe_tmp=$(mktemp -d /tmp/breakthrough-move3.XXXXXXXX)
export TMPDIR="$probe_tmp" MPLCONFIGDIR="$probe_tmp/matplotlib"
trap 'rm -rf -- "$probe_tmp"' EXIT
: "${PROBE_CALIBRATION:?Set PROBE_CALIBRATION to the reviewed calibration JSON}"
# 續跑掛點（vFinal 終審修 2）：PROBE_OUTDIR＋PROBE_RESUME_FROM 由 sbatch --export 傳入
# ＝續跑沿用本腳本全部 Slurm 資源與環境；未設時走首發預設。
outdir="${PROBE_OUTDIR:-/archive/cymaxwelllee/breakthrough1/move3}"
resume_args=()
[ -n "${PROBE_RESUME_FROM:-}" ] && resume_args=(--resume-from "$PROBE_RESUME_FROM")
/archive/cymaxwelllee/LaCoT/.venv/bin/python -u \
  experiments/_workorders/breakthrough-probe/harness_move3.py \
  --outdir "$outdir" "${resume_args[@]}" \
  --calibration "$PROBE_CALIBRATION"
