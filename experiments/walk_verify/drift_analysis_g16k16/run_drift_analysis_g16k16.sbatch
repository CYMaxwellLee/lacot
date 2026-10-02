#!/usr/bin/env bash
#SBATCH -J drift-analysis-g16k16
#SBATCH -p admin
#SBATCH -A it
#SBATCH -q great-mage
#SBATCH --nodelist=zeldajr
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH -o /home/cymaxwelllee/Projects/lacot/experiments/walk_verify/drift_analysis_g16k16/logs/%x-%j.out
# G16K16 單段保真度（E_indep，同 K32 定義）——工單 2026-09-11，見 run_drift_analysis_g16k16.py 檔頭。
# ⛔ CPU-only：不掛 --gres、不碰 GPU。只讀 OGBENCH_DATA_DIR，只寫本目錄 results/。
# 資源宣告照抄 ../drift_analysis/run_drift_analysis.sbatch（同一款 CPU-only MuJoCo 量測）。
set -eu
PY=/home/cymaxwelllee/Projects/lacot/.venv/bin/python
HERE=/home/cymaxwelllee/Projects/lacot/experiments/walk_verify/drift_analysis_g16k16

echo "host: $(hostname)"
echo "python: $PY"
"$PY" --version

env OGBENCH_DATA_DIR=/home/cymaxwelllee/.ogbench/data \
    CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= MUJOCO_GL=osmesa \
    "$PY" -u "$HERE/run_drift_analysis_g16k16.py" "$@"
