#!/usr/bin/env bash
#SBATCH -J bcodec-gk-train
#SBATCH -p admin
#SBATCH -A it
#SBATCH -q great-mage
#SBATCH --nodelist=zeldajr
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH -o /home/cymaxwelllee/Projects/lacot/experiments/bcodec_v0/logs/%x-%j.out
# 行為字典 v0 補跑：G×K 平行頭版（同容量對照）。資源宣告同 bcodec_train.sbatch。
# ⛔ CPU-only：不掛 --gres、不碰 GPU。
set -eu
PY=/home/cymaxwelllee/Projects/lacot/.venv/bin/python
HERE=/home/cymaxwelllee/Projects/lacot/experiments/bcodec_v0

echo "host: $(hostname)"
echo "python: $PY"
"$PY" --version

env OGBENCH_DATA_DIR=/home/cymaxwelllee/.ogbench/data \
    CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= MUJOCO_GL=osmesa \
    "$PY" -u "$HERE/bcodec_gk_train.py" "$@"
