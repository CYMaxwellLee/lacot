#!/bin/bash
#SBATCH -J v3-exp1
#SBATCH -p ada-lite
#SBATCH -w jasmine
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH -t 240
#SBATCH --array=0-5
#SBATCH -o /archive/cymaxwelllee/refine_v3/logs/exp1-%A_%a.out
# v3 預檢修正：--mem=48G（smoke 實測 ant peak RSS 15.7GB＋餘量；48G→24G 收斂讓三顆並行）；
# python -B（不寫 __pycache__ 進 NFS repo）；logs/ 已由 lead 提交前建好（-o 開檔早於 body）。
# v3 實驗 1：教師資格正式訓練（Wφ/Aω/obs-only ×3 seeds ×2 domains）。
# ⓪裝置官已驗：受跑鏈 CPU-only（train_models 有 CPU gate）、無 GPU 儀器、無 gres。
# 產物一律 jasmine 本機 /archive（主人 2026-09-28 儲存裁示）。
# array 映射：id%3=seed(0/1/2)、id/3=domain(0=pointmaze,1=ant)。
# 步數 20000〔拍〕由 smoke 段驗證吞吐後沿用或收斂（smoke=--steps 2000 版另發）。
set -euo pipefail
cd /home/cymaxwelllee/Projects/lacot
SEED=$((SLURM_ARRAY_TASK_ID % 3))
case $((SLURM_ARRAY_TASK_ID / 3)) in
  0) DOMAIN=pointmaze; DATA=/archive/cymaxwelllee/data/ogbench/pointmaze-large-stitch-v0.npz ;;
  1) DOMAIN=ant;       DATA=/archive/cymaxwelllee/data/ogbench/antmaze-medium-stitch-v0.npz ;;
esac
OUT=/archive/cymaxwelllee/refine_v3/exp1/${DOMAIN}-s${SEED}
if [ -d "$OUT" ] && [ -n "$(ls -A "$OUT" 2>/dev/null)" ]; then
  echo "$OUT 已有既存檔案（stale），停止" >&2; exit 24
fi
mkdir -p "$OUT" /archive/cymaxwelllee/refine_v3/logs
PY=/archive/cymaxwelllee/LaCoT/.venv/bin/python
test -x "$PY"
test -f "$DATA"
# --max-windows 300000=解除 1024 toy 上限（發射單「train 全量」；G17 帳 segments ~250k 量級）
# --hidden 256〔拍〕=正式容量，對齊 shat_probe 慣例（toy 的 64 僅接線）
exec "$PY" -B -u -m experiments.refine_v3.train_offline \
  --domain "$DOMAIN" --data "$DATA" --split-seed 1729 --seed "$SEED" \
  --threads 8 --steps 20000 --max-windows 300000 --hidden 256 --out "$OUT"
