#!/usr/bin/env bash
#SBATCH -J robust-dec-driftcmp
#SBATCH -p admin
#SBATCH -A it
#SBATCH -q great-mage
#SBATCH --nodelist=zeldajr
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH -o /home/cymaxwelllee/Projects/lacot/experiments/robust_decoder/logs/%x-%j.out
# 加測：開環漂移增長率 sigma=0 vs 最好顆。⛔ CPU-only：不掛 --gres、不碰 GPU。
set -eu
PY=/home/cymaxwelllee/Projects/lacot/.venv/bin/python
HERE=/home/cymaxwelllee/Projects/lacot/experiments/robust_decoder

echo "host: $(hostname)"
"$PY" --version

env OGBENCH_DATA_DIR=/home/cymaxwelllee/.ogbench/data \
    CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= MUJOCO_GL=osmesa \
    "$PY" -u "$HERE/drift_compare.py" "$@"
