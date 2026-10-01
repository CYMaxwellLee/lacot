#!/bin/bash
#SBATCH --job-name=breakthrough-move1
#SBATCH --partition=ada-lite
#SBATCH --nodelist=jasmine
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/archive/cymaxwelllee/breakthrough1/move1-%j.out
# vFinal must set measured time/quota and prepare the archive directory before sbatch.
set -euo pipefail
cd /home/cymaxwelllee/Projects/lacot
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONHASHSEED=0
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OGBENCH_DATA_DIR=/archive/cymaxwelllee/data/ogbench MUJOCO_GL=osmesa
probe_tmp=$(mktemp -d /tmp/breakthrough-move1.XXXXXXXX)
export TMPDIR="$probe_tmp" MPLCONFIGDIR="$probe_tmp/matplotlib"
trap 'rm -rf -- "$probe_tmp"' EXIT
: "${PROBE_CALIBRATION:?Set PROBE_CALIBRATION to the reviewed calibration JSON}"
# 續跑掛點（vFinal 終審修 2）：同 move3.sbatch
outdir="${PROBE_OUTDIR:-/archive/cymaxwelllee/breakthrough1/move1}"
resume_args=()
[ -n "${PROBE_RESUME_FROM:-}" ] && resume_args=(--resume-from "$PROBE_RESUME_FROM")
/archive/cymaxwelllee/LaCoT/.venv/bin/python -u \
  experiments/_workorders/breakthrough-probe/harness_move1.py \
  --outdir "$outdir" "${resume_args[@]}" \
  --calibration "$PROBE_CALIBRATION"
