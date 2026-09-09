#!/usr/bin/env bash
#SBATCH -p admin
#SBATCH -A it
#SBATCH -q great-mage
#SBATCH --nodelist=zeldajr
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH -o /home/cymaxwelllee/Projects/lacot-gensft/experiments/gen_sft/logs/%x-%j.out
# gen-sft-v1：eval u —— CPU-only（mujoco 物理 + 小 transformer 生成），不掛 --gres=gpu。
# ⛔ 施工在 worktree（~/Projects/lacot-gensft，分支 gen-sft-v1），主 repo main 不動。
set -eu
WT=/home/cymaxwelllee/Projects/lacot-gensft
PY=/home/cymaxwelllee/venvs/lacot-rocm/bin/python
GENSFT_CKPT=${GENSFT_CKPT:?must set GENSFT_CKPT}
N_TRAJ=${N_TRAJ:-200}
BON_N=${BON_N:-8}
TAG=${TAG:-gensft_eval}
EXTRA=${EXTRA:-}

cd "$WT"
echo "worktree: $(git -C "$WT" rev-parse --abbrev-ref HEAD) $(git -C "$WT" rev-parse HEAD)"
echo "node: $(hostname)  gensft_ckpt=$GENSFT_CKPT  n_traj=$N_TRAJ  bon_n=$BON_N  tag=$TAG"

env OGBENCH_DATA_DIR=/home/cymaxwelllee/.ogbench/data MUJOCO_GL=osmesa \
    CUDA_VISIBLE_DEVICES="" HIP_VISIBLE_DEVICES="" \
    "$PY" -u experiments/gen_sft/eval_gen_sft.py \
    --gensft-ckpt "$GENSFT_CKPT" --n-traj "$N_TRAJ" --bon-n "$BON_N" --workers 8 --tag "$TAG" \
    --out-dir experiments/gen_sft/results --gif-dir experiments/gen_sft/results/gifs $EXTRA
