#!/usr/bin/env bash
#SBATCH -p admin
#SBATCH -A it
#SBATCH -q great-mage
#SBATCH --nodelist=zeldajr
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH -o /home/cymaxwelllee/Projects/lacot/experiments/bcodec_shat_probe/logs/%x-%j.out
# 工單2（主人核准施工計畫，2026-09-13）：ŝ 外掛頭——λ0 練好的行為字典，碼裡是否
# 已藏有該段預期到達狀態 ŝ 的資訊。CPU-only（跟 bcodec_v0 的訓練/eval 同一慣例、
# 同一顆 venv）。
# ⛔ 不改動任何既有檔案。2026-09-13 新增。
set -eu
HERE=/home/cymaxwelllee/Projects/lacot/experiments/bcodec_shat_probe
PY=/home/cymaxwelllee/Projects/lacot/.venv/bin/python
CKPT=${CKPT:?must set CKPT}
LAM1_REF=${LAM1_REF:?must set LAM1_REF}
TAG=${TAG:?must set TAG}
STEPS=${STEPS:-20000}
HIDDEN=${HIDDEN:-256}

echo "host: $(hostname)"
echo "python: $PY"
"$PY" --version

env OGBENCH_DATA_DIR=/home/cymaxwelllee/.ogbench/data \
    CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= MUJOCO_GL=osmesa \
    "$PY" -u "$HERE/shat_probe.py" \
    --ckpt "$CKPT" --lam1-ref-json "$LAM1_REF" --tag "$TAG" --steps "$STEPS" --hidden "$HIDDEN" \
    --out-dir "$HERE/results"
