#!/bin/bash
# 跑一次 success-rate rollout（scratch_lacot_rollout.py）。
# 環境變數：LACOT_SEED、LACOT_CONS（self|ema）、LACOT_EMA_M、LACOT_EVAL_EPISODES
set -euo pipefail
PROJ="$HOME/Projects/lacot"
ARCHIVE="/archive/cymaxwelllee"
[ -x "$ARCHIVE/LaCoT/.venv/bin/python" ] || { echo "❌ $(hostname): 沒有本機 venv" >&2; exit 2; }
DATA="$ARCHIVE/data/ogbench"
[ -f "$DATA/pointmaze-medium-navigate-v0.npz" ] || { echo "❌ $(hostname): 沒有本機資料" >&2; exit 3; }
echo "=== $(hostname) | seed=${LACOT_SEED:-0} cons=${LACOT_CONS:-self} ==="
nvidia-smi --query-gpu=index,name --format=csv,noheader | sed 's/^/  GPU /'
cd "$PROJ"
exec env OGBENCH_DATA_DIR="$DATA" MUJOCO_GL=osmesa PYTHONUNBUFFERED=1 \
    "$ARCHIVE/LaCoT/.venv/bin/python" -u experiments/scratch_lacot_rollout.py
