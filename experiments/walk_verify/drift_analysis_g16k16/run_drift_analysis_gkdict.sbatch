#!/usr/bin/env bash
#SBATCH -J drift-analysis-gkdict
#SBATCH -p admin
#SBATCH -A it
#SBATCH -q great-mage
#SBATCH --nodelist=zeldajr
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH -o /home/cymaxwelllee/Projects/lacot/experiments/walk_verify/drift_analysis_g16k16/logs/%x-%j.out
# 通用版 gk_scan 字典單段保真度量測——追加工單 2026-09-11（G32_K32），見
# run_drift_analysis_gkdict.py 檔頭。資源宣告照抄同目錄 run_drift_analysis_g16k16.sbatch
# （同一款 CPU-only MuJoCo 量測，今天稍早已用過）。
# ⛔ CPU-only：不掛 --gres、不碰 GPU。
set -eu
PY=/home/cymaxwelllee/Projects/lacot/.venv/bin/python
HERE=/home/cymaxwelllee/Projects/lacot/experiments/walk_verify/drift_analysis_g16k16

echo "host: $(hostname)"
echo "python: $PY"
"$PY" --version

env OGBENCH_DATA_DIR=/home/cymaxwelllee/.ogbench/data \
    CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= MUJOCO_GL=osmesa \
    "$PY" -u "$HERE/run_drift_analysis_gkdict.py" "$@"
