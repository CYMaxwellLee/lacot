#!/bin/bash
# 跑一個 anti-collapse 變體。由 submit_variants.sh 以 job array 提交。
#
# 配置遵循主人 2026-08-23 的裁示：
#   開發檔案（code）  -> ~/Projects/lacot   一份，走 home（每台都掛得到）
#   大東西（venv/資料）-> 該台自己的 /archive，⛔ 不走 NFS（內網實測只有 ~7MB/s）
#
# 環境變數：
#   VARIANT   要跑的變體名（self0 / self05 / ... / byol / barlow / ema_var / sigreg）
#   LACOT_*   透傳給實驗腳本
set -euo pipefail

PROJ="$HOME/Projects/lacot"
ARCHIVE="/archive/cymaxwelllee"

# venv：優先用本機 /archive 的（快），沒有才退回 home 那份（慢但能跑）
# ⚠️ 2026-08-23：目錄統一叫 LaCoT（主人裁示）。這裡曾經寫成小寫 lacot，
#    於是 fallback 一路掉到 home 那份 NFS venv —— job 跑得動但慢，
#    正好違反「大東西不走 NFS」。改名要掃全部引用處。
if [ -x "$ARCHIVE/LaCoT/.venv/bin/python" ]; then
    PY="$ARCHIVE/LaCoT/.venv/bin/python"
    VENV_SRC="本機 /archive/LaCoT"
else
    # ⛔ 故意不 fallback 到 home 的 NFS venv：那會安靜地變慢而看起來正常。
    #    寧可讓 job 失敗、把問題暴露出來。
    echo "❌ $(hostname): 本機沒有 venv（$ARCHIVE/LaCoT/.venv）" >&2
    echo "   ⛔ 不 fallback 到 home 的 NFS 版本 —— 那會安靜地拖慢整批 job。" >&2
    exit 2
fi

# 資料：一定要本機的，⛔ 不接受 NFS 版本
DATA="$ARCHIVE/data/ogbench"
if [ ! -f "$DATA/pointmaze-medium-navigate-v0.npz" ]; then
    echo "❌ $(hostname): 本機沒有 ogbench 資料（$DATA）" >&2
    exit 3
fi

echo "=== $(hostname) | variant=${VARIANT:-?} | venv=$VENV_SRC ==="
nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader | sed 's/^/  GPU /'
echo

cd "$PROJ"
exec env OGBENCH_DATA_DIR="$DATA" PYTHONUNBUFFERED=1 \
    "$PY" -u experiments/exp_anticollapse_one.py
