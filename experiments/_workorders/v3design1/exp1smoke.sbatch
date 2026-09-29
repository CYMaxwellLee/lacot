#!/bin/bash
#SBATCH -J v3-exp1-smoke
#SBATCH -p ada-lite
#SBATCH -w jasmine
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 40
#SBATCH --array=0-1
#SBATCH -o /archive/cymaxwelllee/refine_v3/logs/exp1smoke-%A_%a.out
# 實驗 1 smoke（F5 鏈；①官 C2 建議命令的 sbatch 版）：--steps 2000、同正式 code path、
# /usr/bin/time -v 量 peak RSS（ant 格=RAM 判準）；out 到 smoke/ 不碰正式。
set -euo pipefail
cd /home/cymaxwelllee/Projects/lacot
case $SLURM_ARRAY_TASK_ID in
  0) DOMAIN=pointmaze; DATA=/archive/cymaxwelllee/data/ogbench/pointmaze-large-stitch-v0.npz ;;
  1) DOMAIN=ant;       DATA=/archive/cymaxwelllee/data/ogbench/antmaze-medium-stitch-v0.npz ;;
esac
OUT=/archive/cymaxwelllee/refine_v3/smoke/${DOMAIN}-s0
if [ -d "$OUT" ] && [ -n "$(ls -A "$OUT" 2>/dev/null)" ]; then
  echo "$OUT 已有既存檔案，停止" >&2; exit 24
fi
mkdir -p "$OUT"
PY=/archive/cymaxwelllee/LaCoT/.venv/bin/python
test -x "$PY"
"$PY" -B -c "import experiments.refine_v3.train_offline as m; print('module OK', m.__file__)"
exec /usr/bin/time -v "$PY" -B -u -m experiments.refine_v3.train_offline \
  --domain "$DOMAIN" --data "$DATA" --split-seed 1729 --seed 0 \
  --threads 8 --steps 2000 --max-windows 300000 --hidden 256 --out "$OUT"
