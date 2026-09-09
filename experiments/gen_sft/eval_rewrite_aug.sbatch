#!/usr/bin/env bash
#SBATCH -p admin
#SBATCH -A it
#SBATCH -q great-mage
#SBATCH --nodelist=zeldajr
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH -o /home/cymaxwelllee/Projects/lacot-rewrite/experiments/gen_sft/logs/%x-%j.out
# rewrite-v1 步2：鐘形定讞 + 半路起點增強 —— 六顆 eval（各自 R grid 由 R_GRID 指定）。
# CPU-only（mujoco 物理 + 小 transformer 生成），不掛 --gres=gpu。
# ⛔ 新檔（v3/aug 系列），不改動既有 eval_rewrite.sbatch。
set -eu
WT=/home/cymaxwelllee/Projects/lacot-rewrite
PY=/home/cymaxwelllee/venvs/lacot-rocm/bin/python
GENSFT_CKPT=${GENSFT_CKPT:?must set GENSFT_CKPT}
R_GRID=${R_GRID:-8,inf}
N_TRAJ=${N_TRAJ:-200}
TAG=${TAG:-rewrite_eval_aug}
EXTRA=${EXTRA:-}

cd "$WT"
echo "worktree: $(git -C "$WT" rev-parse --abbrev-ref HEAD) $(git -C "$WT" rev-parse HEAD)"
echo "node: $(hostname)  gensft_ckpt=$GENSFT_CKPT  r_grid=$R_GRID  n_traj=$N_TRAJ  tag=$TAG"

env OGBENCH_DATA_DIR=/home/cymaxwelllee/.ogbench/data MUJOCO_GL=osmesa \
    CUDA_VISIBLE_DEVICES="" HIP_VISIBLE_DEVICES="" \
    "$PY" -u experiments/gen_sft/eval_rewrite_aug.py \
    --gensft-ckpt "$GENSFT_CKPT" --r-grid "$R_GRID" --n-traj "$N_TRAJ" --workers 8 --tag "$TAG" \
    --out-dir experiments/gen_sft/results --gif-dir experiments/gen_sft/results/gifs_rewrite_aug $EXTRA
