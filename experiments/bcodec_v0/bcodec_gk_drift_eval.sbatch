#!/usr/bin/env bash
#SBATCH -J bcodec-gk-eval
#SBATCH -p admin
#SBATCH -A it
#SBATCH -q great-mage
#SBATCH --nodelist=zeldajr
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH -o /home/cymaxwelllee/Projects/lacot/experiments/bcodec_v0/logs/%x-%j.out
set -eu
PY=/home/cymaxwelllee/Projects/lacot/.venv/bin/python
HERE=/home/cymaxwelllee/Projects/lacot/experiments/bcodec_v0
env OGBENCH_DATA_DIR=/home/cymaxwelllee/.ogbench/data \
    CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= MUJOCO_GL=osmesa \
    "$PY" -u "$HERE/bcodec_gk_drift_eval.py" "$@"
