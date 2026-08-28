#!/bin/bash
# u 裡到底裝了多少路線資訊 —— 直接量（experiments/exp_decode_probe.py）。
# 🚨 主人 2026-08-28 核可：「這禮拜已經測無數次了，最後一次，測完不要忘記」
# 環境變數：LACOT_DP_MODE(u|sg|ae)、LACOT_K、LACOT_DP_CKPT、LACOT_SEED、LACOT_DP_STEPS
set -euo pipefail
PROJ="$HOME/Projects/lacot"
ARCHIVE="/archive/cymaxwelllee"
ENV_NAME="${LACOT_ENV:-pointmaze-medium-stitch-v0}"
[ -x "$ARCHIVE/LaCoT/.venv/bin/python" ] || { echo "❌ $(hostname): 沒有本機 venv" >&2; exit 2; }
DATA="$ARCHIVE/data/ogbench"
[ -f "$DATA/$ENV_NAME.npz" ] || { echo "❌ $(hostname): 沒有本機資料 $ENV_NAME" >&2; exit 3; }
# ⭐ mode=u 一定要 ckpt，⛔ 缺了就停 —— 不然它會靜默地跑成一個沒意義的配置
if [ "${LACOT_DP_MODE:-u}" = "u" ]; then
  [ -n "${LACOT_DP_CKPT:-}" ] || { echo "❌ mode=u 但沒給 LACOT_DP_CKPT" >&2; exit 4; }
  [ -f "$PROJ/${LACOT_DP_CKPT}" ] || { echo "❌ 找不到 ckpt: $PROJ/${LACOT_DP_CKPT}" >&2; exit 5; }
fi
echo "=== $(hostname) | mode=${LACOT_DP_MODE:-u} K=${LACOT_K:-4} seed=${LACOT_SEED:-0} env=$ENV_NAME ==="
nvidia-smi --query-gpu=index,name --format=csv,noheader | sed 's/^/  GPU /'
cd "$PROJ"
exec env OGBENCH_DATA_DIR="$DATA" PYTHONUNBUFFERED=1 \
    "$ARCHIVE/LaCoT/.venv/bin/python" -u experiments/exp_decode_probe.py
