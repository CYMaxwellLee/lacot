#!/usr/bin/env bash
#SBATCH -J bcodec-drift-eval
#SBATCH -p admin
#SBATCH -A it
#SBATCH -q great-mage
#SBATCH --nodelist=zeldajr
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH -o /home/cymaxwelllee/Projects/lacot/experiments/bcodec_v0/logs/%x-%j.out
# 行為字典 v0 關1 量測——資源宣告照抄 run_drift_analysis_gkdict.sbatch（昨天 job
# 28033，COMPLETED）。⛔ CPU-only：不掛 --gres、不碰 GPU。
set -eu
PY=/home/cymaxwelllee/Projects/lacot/.venv/bin/python
HERE=/home/cymaxwelllee/Projects/lacot/experiments/bcodec_v0

echo "host: $(hostname)"
echo "python: $PY"
"$PY" --version

env OGBENCH_DATA_DIR=/home/cymaxwelllee/.ogbench/data \
    CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= MUJOCO_GL=osmesa \
    "$PY" -u "$HERE/bcodec_drift_eval.py" "$@"
