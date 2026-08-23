#!/bin/bash
# 跑 exp_three_fronts.py 的一格。由 submit_fronts.sh 提交。
#
# 配置（主人 2026-08-23 裁示）：
#   code -> ~/Projects/lacot（走 home，每台掛得到）
#   venv/資料 -> 該台自己的 /archive，⛔ 不走 NFS（內網實測 ~7MB/s）
#
# 環境變數：MODE（要跑哪一格）、LACOT_SEED、LACOT_STEPS1/2
set -euo pipefail

PROJ="$HOME/Projects/lacot"
ARCHIVE="/archive/cymaxwelllee"

if [ -x "$ARCHIVE/LaCoT/.venv/bin/python" ]; then
    PY="$ARCHIVE/LaCoT/.venv/bin/python"
else
    # ⛔ 故意不 fallback 到 home 的 NFS venv：那會安靜地變慢而看起來正常。
    echo "❌ $(hostname): 本機沒有 venv（$ARCHIVE/LaCoT/.venv）" >&2
    exit 2
fi

DATA="$ARCHIVE/data/ogbench"
if [ ! -f "$DATA/pointmaze-medium-navigate-v0.npz" ]; then
    echo "❌ $(hostname): 本機沒有 ogbench 資料（$DATA）" >&2
    exit 3
fi

echo "=== $(hostname) | MODE=${MODE:-?} seed=${LACOT_SEED:-0} ==="
nvidia-smi --query-gpu=index,name --format=csv,noheader | sed 's/^/  GPU /'

cd "$PROJ"
exec env OGBENCH_DATA_DIR="$DATA" PYTHONUNBUFFERED=1 "$PY" -u experiments/exp_three_fronts.py
