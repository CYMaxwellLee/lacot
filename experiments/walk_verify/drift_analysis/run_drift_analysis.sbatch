#!/usr/bin/env bash
#SBATCH -J drift-analysis
#SBATCH -p admin
#SBATCH -A it
#SBATCH -q great-mage
#SBATCH --nodelist=zeldajr
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH -o /home/cymaxwelllee/Projects/lacot/experiments/walk_verify/drift_analysis/logs/%x-%j.out
# 定量拆解「完美教材開環播放為什麼還是只有 .554」—— 量化損失 vs 開環執行漂移。
# ⛔ CPU-only：不掛 --gres、不碰 GPU。只讀 OGBENCH_DATA_DIR，只寫本目錄 results/。
set -eu
PY=/home/cymaxwelllee/Projects/lacot/.venv/bin/python
HERE=/home/cymaxwelllee/Projects/lacot/experiments/walk_verify/drift_analysis

echo "host: $(hostname)"
echo "python: $PY"
"$PY" --version

env OGBENCH_DATA_DIR=/home/cymaxwelllee/.ogbench/data \
    CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= MUJOCO_GL=osmesa \
    "$PY" -u "$HERE/run_drift_analysis.py" "$@"
