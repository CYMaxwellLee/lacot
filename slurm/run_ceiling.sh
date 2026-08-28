#!/bin/bash
# 「絕對對的 e_target 天花板」實驗（exp_etarget_ceiling.py）。主人 2026-08-24 裁示。
# 環境變數：LACOT_ENV、LACOT_SEED、LACOT_STEPS1/2、LACOT_EVAL_EPISODES、LACOT_EVAL_MAXH、
#           LACOT_OFFLINE_BATCHES
set -euo pipefail
PROJ="$HOME/Projects/lacot"; ARCHIVE="/archive/cymaxwelllee"
[ -x "$ARCHIVE/LaCoT/.venv/bin/python" ] || { echo "❌ $(hostname): 沒有本機 venv" >&2; exit 2; }
DATA="$ARCHIVE/data/ogbench"
ENVN="${LACOT_ENV:-pointmaze-medium-navigate-v0}"
[ -f "$DATA/$ENVN.npz" ] || { echo "❌ $(hostname): 沒有本機資料 $DATA/$ENVN.npz" >&2; exit 3; }
echo "=== $(hostname) | env=$ENVN seed=${LACOT_SEED:-0} steps2=${LACOT_STEPS2:-12000} ==="
nvidia-smi --query-gpu=index,name --format=csv,noheader | sed 's/^/  GPU /'
cd "$PROJ"
exec env OGBENCH_DATA_DIR="$DATA" MUJOCO_GL=osmesa PYTHONUNBUFFERED=1 \
    "$ARCHIVE/LaCoT/.venv/bin/python" -u experiments/exp_etarget_ceiling.py
