#!/bin/bash
set -euo pipefail
PROJ="$HOME/Projects/lacot"; ARCHIVE="/archive/cymaxwelllee"
DATA="$ARCHIVE/data/ogbench"
[ -f "$DATA/${LACOT_ENV:-pointmaze-medium-stitch-v0}.npz" ] || { echo "❌ $(hostname): 沒資料" >&2; exit 3; }
cd "$PROJ"
exec env OGBENCH_DATA_DIR="$DATA" MUJOCO_GL=osmesa PYTHONUNBUFFERED=1 \
    "$ARCHIVE/LaCoT/.venv/bin/python" -u experiments/exp_span_gap.py
