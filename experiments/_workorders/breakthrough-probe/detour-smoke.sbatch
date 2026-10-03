#!/bin/bash
#SBATCH --job-name=detour-smoke
#SBATCH --partition=ada-lite
#SBATCH --nodelist=jasmine
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/archive/cymaxwelllee/breakthrough1/detour-u/smoke-%j.out
set -euo pipefail
: "${1:?First argument must be the frozen program root}"
cd -- "$1"
: "${DETOUR_PYTHON:?Set interpreter path on jasmine}"
: "${DETOUR_DATASET_DIR:?Set jasmine local dataset directory}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONHASHSEED=0
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 MUJOCO_GL=osmesa
export OGBENCH_DATA_DIR="$DETOUR_DATASET_DIR"
probe_tmp=$(mktemp -d /tmp/detour-smoke.XXXXXXXX)
export TMPDIR="$probe_tmp" MPLCONFIGDIR="$probe_tmp/matplotlib"
trap 'rm -rf -- "$probe_tmp"' EXIT
# E1 generator and fifteen draws share one model, device and deterministic configuration.
# Lead-approved copied factory keeps flow_seed verification and accepts detour draws.
"$DETOUR_PYTHON" -B -u experiments/_workorders/breakthrough-probe/harness_detour.py \
  --smoke --outdir /archive/cymaxwelllee/breakthrough1/detour-u/smoke \
  --dataset-dir "$DETOUR_DATASET_DIR"
