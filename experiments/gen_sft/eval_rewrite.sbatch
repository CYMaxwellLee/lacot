#!/usr/bin/env bash
#SBATCH -p admin
#SBATCH -A it
#SBATCH -q great-mage
#SBATCH --nodelist=zeldajr
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH -o /home/cymaxwelllee/Projects/lacot-rewrite/experiments/gen_sft/logs/%x-%j.out
# rewrite-v1：eval 定期重寫譜（receding horizon）掃 R —— CPU-only（mujoco 物理 + 小
# transformer 生成，重寫時額外的 regenerate 呼叫一樣是 CPU 小模型），不掛 --gres=gpu。
# ⛔ 施工在 worktree（~/Projects/lacot-rewrite，分支 rewrite-v1），主 repo main 不動。
set -eu
WT=/home/cymaxwelllee/Projects/lacot-rewrite
PY=/home/cymaxwelllee/venvs/lacot-rocm/bin/python
GENSFT_CKPT=${GENSFT_CKPT:-experiments/gen_sft/results/gensft_v1_best.pt}
N_TRAJ=${N_TRAJ:-200}
TAG=${TAG:-rewrite_eval}
EXTRA=${EXTRA:-}

cd "$WT"
echo "worktree: $(git -C "$WT" rev-parse --abbrev-ref HEAD) $(git -C "$WT" rev-parse HEAD)"
echo "node: $(hostname)  gensft_ckpt=$GENSFT_CKPT  n_traj=$N_TRAJ  tag=$TAG"

env OGBENCH_DATA_DIR=/home/cymaxwelllee/.ogbench/data MUJOCO_GL=osmesa \
    CUDA_VISIBLE_DEVICES="" HIP_VISIBLE_DEVICES="" \
    "$PY" -u experiments/gen_sft/eval_rewrite.py \
    --gensft-ckpt "$GENSFT_CKPT" --n-traj "$N_TRAJ" --workers 8 --tag "$TAG" \
    --out-dir experiments/gen_sft/results --gif-dir experiments/gen_sft/results/gifs_rewrite $EXTRA
