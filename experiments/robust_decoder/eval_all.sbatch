#!/usr/bin/env bash
#SBATCH -J robust-dec-eval
#SBATCH -p admin
#SBATCH -A it
#SBATCH -q great-mage
#SBATCH --nodelist=zeldajr
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH -o /home/cymaxwelllee/Projects/lacot/experiments/robust_decoder/logs/%x-%j.out
# 驗收 #1 sanity / #3 主表 / #5 gif —— schedule 版 200 題教材接力，5 顆並排。
# ⛔ CPU-only：不掛 --gres、不碰 GPU。只讀 OGBENCH_DATA_DIR，只寫本目錄 results/。
set -eu
PY=/home/cymaxwelllee/Projects/lacot/.venv/bin/python
HERE=/home/cymaxwelllee/Projects/lacot/experiments/robust_decoder

echo "host: $(hostname)"
"$PY" --version

env OGBENCH_DATA_DIR=/home/cymaxwelllee/.ogbench/data \
    CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= MUJOCO_GL=osmesa \
    "$PY" -u "$HERE/eval_all.py" "$@"
