#!/bin/bash
#SBATCH -J g17-seedrep2
#SBATCH -p ada-lite
#SBATCH -w jasmine
#SBATCH --cpus-per-task=8
#SBATCH -t 90
#SBATCH --array=8-11
#SBATCH -o /archive/cymaxwelllee/diag-g16/seedrep/logs/rep2-%A_%a.out
# G17 補跑（2026-09-28、F5 預檢 NO-GO 修正後 v2）：G32K32 四顆 —— 33805 task 8-11 TIMEOUT。
# 死因帳（預檢官修訂版）：訓練 code 是 CPU-only（bcodec_common.py:159/:296 寫死 device cpu、
# set_num_threads(8)），昨版 --gres=gpu:1 白佔卡且【沒申請 CPU 核】⇒ 8 threads 擠在預設核數上
# ＝ G32K32 90 分跑不完的主嫌；加上無 python -u（buffered log 零輸出）＝死因不可見。
# v2 修正：去 --gres、加 --cpus-per-task=8（33809 同款有先例）、加 -u、set -euo pipefail。
# v3：-t 90 —— smoke 33851 實測 8 核下 1500 步 66.5s ⇒ 50k 外推 ~36 分、×2.5 裕量
# （lam_s=1.0 臂多 state loss 項，留裕）。CELLS/SEEDS 映射與輸出路徑逐字不動。
set -euo pipefail
CELLS=("vq 32 0.0" "vq 32 1.0" "gk 16 0.0" "gk 16 1.0" "gk 32 0.0" "gk 32 1.0")
SEEDS=(80001 80002)
CELL=${CELLS[$((SLURM_ARRAY_TASK_ID / 2))]}
SEED=${SEEDS[$((SLURM_ARRAY_TASK_ID % 2))]}
read KIND K LAM <<< "$CELL"
OUT=/archive/cymaxwelllee/diag-g16/seedrep/s$SEED
mkdir -p $OUT
cd /archive/cymaxwelllee/diag-g16
PY=/archive/cymaxwelllee/LaCoT/.venv/bin/python3
if [ "$KIND" = "gk" ]; then
  exec $PY -u experiments/bcodec_v0/bcodec_gk_train.py --g $K --k $K --lam-s $LAM --seed $SEED \
    --data-dir /archive/cymaxwelllee/data/ogbench --out-dir $OUT
else
  exec $PY -u experiments/bcodec_v0/bcodec_train.py --k $K --lam-s $LAM --seed $SEED \
    --data-dir /archive/cymaxwelllee/data/ogbench --out-dir $OUT
fi
