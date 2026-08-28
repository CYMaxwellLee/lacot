#!/bin/bash
# 對照 smoke：跑【改動之前】那一版主線，用來驗新版帶舊參數時行為沒變。
# ⛔ 暫時性的，驗完就刪。
set -euo pipefail
PROJ="$HOME/Projects/lacot"; ARCHIVE="/archive/cymaxwelllee"
DATA="$ARCHIVE/data/ogbench"
[ -f "$DATA/${LACOT_ENV:-pointmaze-medium-stitch-v0}.npz" ] || { echo "❌ $(hostname): 沒資料" >&2; exit 3; }
cd "$PROJ"
exec env OGBENCH_DATA_DIR="$DATA" MUJOCO_GL=osmesa PYTHONUNBUFFERED=1 \
    "$ARCHIVE/LaCoT/.venv/bin/python" -u .smokeref/scratch_lacot_rollout_before.py
