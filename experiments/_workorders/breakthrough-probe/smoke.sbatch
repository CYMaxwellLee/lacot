#!/bin/bash
#SBATCH --job-name=breakthrough-smoke
#SBATCH --partition=ada-lite
#SBATCH --nodelist=jasmine
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/archive/cymaxwelllee/breakthrough1/smoke-%j.out
# BLOCKED: not launch-ready. 30 minutes is a placeholder, NOT a measured ETA.
# vFinal must set measured time/quota and prepare the archive directory before sbatch.
set -euo pipefail
cd /home/cymaxwelllee/Projects/lacot
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONHASHSEED=0
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OGBENCH_DATA_DIR=/archive/cymaxwelllee/data/ogbench MUJOCO_GL=osmesa
probe_tmp=$(mktemp -d /tmp/breakthrough-smoke.XXXXXXXX)
export TMPDIR="$probe_tmp" MPLCONFIGDIR="$probe_tmp/matplotlib"
trap 'rm -rf -- "$probe_tmp"' EXIT
: "${PROBE_CALIBRATION:?Set PROBE_CALIBRATION to the reviewed calibration JSON}"
/archive/cymaxwelllee/LaCoT/.venv/bin/python -u \
  experiments/_workorders/breakthrough-probe/smoke.py \
  --outdir /archive/cymaxwelllee/breakthrough1/smoke \
  --calibration "$PROBE_CALIBRATION"
