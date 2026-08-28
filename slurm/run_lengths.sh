#!/bin/bash
# 量軌跡/goal/oracle 路徑長度（experiments/measure_lengths.py）。主人 2026-08-24 交辦。
set -euo pipefail
PROJ="$HOME/Projects/lacot"; ARCHIVE="/archive/cymaxwelllee"
[ -x "$ARCHIVE/LaCoT/.venv/bin/python" ] || { echo "❌ $(hostname): 沒有本機 venv" >&2; exit 2; }
DATA="$ARCHIVE/data/ogbench"
[ -f "$DATA/pointmaze-medium-navigate-v0.npz" ] || { echo "❌ $(hostname): 沒有本機資料" >&2; exit 3; }
cd "$PROJ"
exec env OGBENCH_DATA_DIR="$DATA" MUJOCO_GL=osmesa PYTHONUNBUFFERED=1 \
    "$ARCHIVE/LaCoT/.venv/bin/python" -u experiments/measure_lengths.py
