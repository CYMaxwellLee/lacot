#!/bin/bash
# Common deployment skeleton from experiments/j_signal_trial/run_三臂.sbatch.
set -euo pipefail
cd "$HOME/Projects/lacot"
UCONTRAST_ARCHIVE=/archive/cymaxwelllee
UCONTRAST_PY="$UCONTRAST_ARCHIVE/LaCoT/.venv/bin/python"
UCONTRAST_DATA="$UCONTRAST_ARCHIVE/data/ogbench"
test -n "${SLURM_JOB_ID:-}" || { echo 'BLOCKED: sbatch allocation required'; exit 2; }
test -x "$UCONTRAST_PY" || { echo "BLOCKED: missing venv $UCONTRAST_PY"; exit 2; }
test -f "$UCONTRAST_DATA/pointmaze-large-stitch-v0.npz" || { echo 'BLOCKED: missing dataset'; exit 2; }
export OGBENCH_DATA_DIR="$UCONTRAST_DATA" MUJOCO_GL=osmesa PYTHONUNBUFFERED=1
exec "$UCONTRAST_PY" -u experiments/_workorders/ucontrast1/run.py "$@"
