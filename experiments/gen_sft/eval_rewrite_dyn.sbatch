#!/usr/bin/env bash
#SBATCH -p admin
#SBATCH -A it
#SBATCH -q great-mage
#SBATCH --nodelist=zeldajr
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH -o /home/cymaxwelllee/Projects/lacot-dyn/experiments/gen_sft/logs/%x-%j.out
# 主人裁示①②：dyn 六顆 eval —— 直接沿用既有 eval_rewrite_aug.py（不改一行，機制
# 100% 沿用：eval_rewrite_aug.py 只看 --gensft-ckpt 路徑，不管訓練語料是 aug 還是
# dyn，本檔只是新的 sbatch 呼叫殼，換 WT + 換 ckpt 路徑）。
# CPU-only（mujoco 物理 + 小 transformer 生成），不掛 --gres=gpu。
# ⛔ 新檔（dyn 系列），不改動既有 eval_rewrite_aug.py / eval_rewrite_aug.sbatch。
# ⚠️ 抄自 eval_rewrite_aug.sbatch 時原檔 WT=lacot-rewrite（隊友施工目錄）——已改成
#    lacot-dyn，提交前有重新核對，不碰隊友那邊。
set -eu
WT=/home/cymaxwelllee/Projects/lacot-dyn
PY=/home/cymaxwelllee/venvs/lacot-rocm/bin/python
GENSFT_CKPT=${GENSFT_CKPT:?must set GENSFT_CKPT}
R_GRID=${R_GRID:-8,inf}
N_TRAJ=${N_TRAJ:-200}
TAG=${TAG:-rewrite_eval_dyn}
EXTRA=${EXTRA:-}

cd "$WT"
echo "worktree: $(git -C "$WT" rev-parse --abbrev-ref HEAD) $(git -C "$WT" rev-parse HEAD)"
echo "node: $(hostname)  gensft_ckpt=$GENSFT_CKPT  r_grid=$R_GRID  n_traj=$N_TRAJ  tag=$TAG"

env OGBENCH_DATA_DIR=/home/cymaxwelllee/.ogbench/data MUJOCO_GL=osmesa \
    CUDA_VISIBLE_DEVICES="" HIP_VISIBLE_DEVICES="" \
    "$PY" -u experiments/gen_sft/eval_rewrite_aug.py \
    --gensft-ckpt "$GENSFT_CKPT" --r-grid "$R_GRID" --n-traj "$N_TRAJ" --workers 8 --tag "$TAG" \
    --out-dir experiments/gen_sft/results --gif-dir experiments/gen_sft/results/gifs_rewrite_dyn $EXTRA
