#!/bin/bash
#SBATCH --job-name=ucontrast-flagoff
#SBATCH --partition=ada-lite
#SBATCH --nodelist=jasmine
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --output=experiments/_workorders/ucontrast1/smoke/flagoff-%j.out
set -euo pipefail
cd "$HOME/Projects/lacot"
UCONTRAST_PY=/archive/cymaxwelllee/LaCoT/.venv/bin/python
UCONTRAST_DATA=/archive/cymaxwelllee/data/ogbench
test -n "${SLURM_JOB_ID:-}" || { echo 'BLOCKED: sbatch allocation required'; exit 2; }
test -x "$UCONTRAST_PY" || { echo "BLOCKED: missing venv $UCONTRAST_PY"; exit 2; }
test -f "$UCONTRAST_DATA/pointmaze-large-stitch-v0.npz" || { echo 'BLOCKED: missing dataset'; exit 2; }
export OGBENCH_DATA_DIR="$UCONTRAST_DATA" MUJOCO_GL=osmesa PYTHONUNBUFFERED=1
exec "$UCONTRAST_PY" -u experiments/_workorders/ucontrast1/gen_flagoff_proof.py \
  --tasks '1:3,2:7,4:11' --device cuda \
  --out experiments/_workorders/ucontrast1/smoke/flagoff-proof.json
