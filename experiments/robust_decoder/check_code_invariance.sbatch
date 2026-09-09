#!/usr/bin/env bash
#SBATCH -J robust-dec-codecheck
#SBATCH -p admin
#SBATCH -A it
#SBATCH -q great-mage
#SBATCH --nodelist=zeldajr
#SBATCH --cpus-per-task=4
#SBATCH --time=00:15:00
#SBATCH -o /home/cymaxwelllee/Projects/lacot/experiments/robust_decoder/logs/%x-%j.out
# 驗收 #2：code 不變斷言。⛔ CPU-only：不掛 --gres、不碰 GPU。
set -eu
PY=/home/cymaxwelllee/Projects/lacot/.venv/bin/python
HERE=/home/cymaxwelllee/Projects/lacot/experiments/robust_decoder

echo "host: $(hostname)"
"$PY" --version

env OGBENCH_DATA_DIR=/home/cymaxwelllee/.ogbench/data CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= \
    "$PY" -u "$HERE/check_code_invariance.py" "$@"
