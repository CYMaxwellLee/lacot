#!/usr/bin/env bash
#SBATCH -J robust-dec-train
#SBATCH -p admin
#SBATCH -A it
#SBATCH -q great-mage
#SBATCH --nodelist=zeldajr
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH -o /home/cymaxwelllee/Projects/lacot/experiments/robust_decoder/logs/%x-%j.out
# robust decoder fine-tune（decoder 吃「偏移的姿態」）—— 一顆 = 一個 sigma-xy 檔位。
# ⛔ GPU 訓練：照範本 header（p2_eval_vqoracle.sbatch）—— admin/it/great-mage/zeldajr/gpu:1。
# 用法：sbatch train_robust_decoder.sbatch <sigma_xy> <tag> [額外 python 參數...]
set -eu
PY=/home/cymaxwelllee/venvs/lacot-rocm/bin/python
HERE=/home/cymaxwelllee/Projects/lacot/experiments/robust_decoder

SIGMA_XY=${1:?用法: sbatch train_robust_decoder.sbatch <sigma_xy> <tag>}
TAG=${2:?用法: sbatch train_robust_decoder.sbatch <sigma_xy> <tag>}
shift 2 || true

echo "host: $(hostname)"
echo "python: $PY"
"$PY" --version
echo "sigma_xy=$SIGMA_XY tag=$TAG extra_args=$*"

# ⚠️ 不手動設 CUDA_VISIBLE_DEVICES / HIP_VISIBLE_DEVICES —— 這是 GPU job，
# 可見性交給 slurm 的 --gres=gpu:1 配置（跟 p2_eval_vqoracle.sbatch 同款，
# 那支範本本來就不手動設這兩個變數）。
echo "rocm-smi 摘要："; rocm-smi --showid 2>&1 | head -20 || true

env OGBENCH_DATA_DIR=/home/cymaxwelllee/.ogbench/data \
    "$PY" -u "$HERE/train_robust_decoder.py" --sigma-xy "$SIGMA_XY" --tag "$TAG" \
        --steps 20000 --device cuda "$@"
