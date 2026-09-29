#!/bin/bash
#SBATCH --job-name=ucontrast2-s35
#SBATCH --array=0-1
#SBATCH --partition=ada-lite
#SBATCH --nodelist=jasmine
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=/archive/cymaxwelllee/ucontrast1/logs/s35-%A_%a.out
set -euo pipefail
cd "$HOME/Projects/lacot-u-frozen"
export OGBENCH_DATA_DIR=/archive/cymaxwelllee/data/ogbench MUJOCO_GL=osmesa PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
exec /archive/cymaxwelllee/LaCoT/.venv/bin/python -u \
  experiments/_workorders/ucontrast2/run.py s35 \
  --index "$SLURM_ARRAY_TASK_ID" --outdir /archive/cymaxwelllee/ucontrast1/s35
