#!/bin/bash
# 載入 checkpoint 換 u 探針，⛔ 不重訓。環境變數：LACOT_CKPT、LACOT_ENV、LACOT_R
set -euo pipefail
PROJ="$HOME/Projects/lacot"; ARCHIVE="/archive/cymaxwelllee"
[ -x "$ARCHIVE/LaCoT/.venv/bin/python" ] || { echo "❌ $(hostname): 沒有本機 venv" >&2; exit 2; }
cd "$PROJ"
exec env OGBENCH_DATA_DIR="$ARCHIVE/data/ogbench" MUJOCO_GL=osmesa PYTHONUNBUFFERED=1 \
    "$ARCHIVE/LaCoT/.venv/bin/python" -u experiments/probe_u.py
