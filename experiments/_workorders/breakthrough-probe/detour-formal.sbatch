#!/bin/bash
#SBATCH --job-name=detour-formal
#SBATCH --partition=ada-lite
#SBATCH --nodelist=jasmine
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/archive/cymaxwelllee/breakthrough1/detour-u/formal-%j.out
# 時限用命令列 sbatch --time 覆蓋，⛔ 不改檔
# --time 數值由 lead 按 smoke 加權 ETA ×2 設定。
set -euo pipefail
: "${1:?First argument must be the frozen program root}"
cd -- "$1"
: "${DETOUR_PYTHON:?Set interpreter path on jasmine}"
: "${DETOUR_DATASET_DIR:?Set jasmine local dataset directory}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONHASHSEED=0
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 MUJOCO_GL=osmesa
export OGBENCH_DATA_DIR="$DETOUR_DATASET_DIR"
probe_tmp=$(mktemp -d /tmp/detour-formal.XXXXXXXX)
export TMPDIR="$probe_tmp" MPLCONFIGDIR="$probe_tmp/matplotlib"
trap 'rm -rf -- "$probe_tmp"' EXIT
outdir="${DETOUR_OUTDIR:-/archive/cymaxwelllee/breakthrough1/detour-u/formal}"
resume_args=()
if [ -n "${DETOUR_RESUME_FROM:-}" ]; then
  : "${DETOUR_OUTDIR:?Resume requires a fresh formal-r2 output directory}"
  resume_args=(--resume-from "$DETOUR_RESUME_FROM")
fi
"$DETOUR_PYTHON" -B -u experiments/_workorders/breakthrough-probe/harness_detour.py \
  --outdir "$outdir" --dataset-dir "$DETOUR_DATASET_DIR" "${resume_args[@]}" \
  --e1-table /archive/cymaxwelllee/breakthrough1/detour-u/smoke/e1-table-v8.json
