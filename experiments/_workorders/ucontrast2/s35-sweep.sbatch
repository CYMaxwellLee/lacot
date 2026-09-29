#!/bin/bash
#SBATCH --job-name=ucontrast2-s35-sweep
#SBATCH --array=0-3
#SBATCH --partition=ada-lite
#SBATCH --nodelist=jasmine
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=/archive/cymaxwelllee/ucontrast1/logs/s35-sweep-%A_%a.out
set -euo pipefail
cd "$HOME/Projects/lacot-u-frozen"
# 0=A sigma 0; 1/2/3=B sigma 0.05/0.10/0.20; ep 40-44 x 8 draws.
export OGBENCH_DATA_DIR=/archive/cymaxwelllee/data/ogbench MUJOCO_GL=osmesa PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
exec /archive/cymaxwelllee/LaCoT/.venv/bin/python -u \
  experiments/_workorders/ucontrast2/run.py s35 --s35-stage sweep \
  --index "$SLURM_ARRAY_TASK_ID" \
  --outdir /archive/cymaxwelllee/ucontrast1/s35-calibration
