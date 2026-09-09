#!/usr/bin/env bash
#SBATCH -p admin
#SBATCH -A it
#SBATCH -q great-mage
#SBATCH --nodelist=zeldajr
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH -o /home/cymaxwelllee/Projects/lacot-rewrite/experiments/gen_sft/logs/%x-%j.out
# rewrite-v1 步2 追加驗收（主人 2026-09-09 點名）：姿態置換探針。CPU-only。
set -eu
WT=/home/cymaxwelllee/Projects/lacot-rewrite
PY=/home/cymaxwelllee/venvs/lacot-rocm/bin/python
GENSFT_CKPT=${GENSFT_CKPT:?must set GENSFT_CKPT}
N_PROBE=${N_PROBE:-40}
R_REWRITE=${R_REWRITE:-8}
TAG=${TAG:?must set TAG}

cd "$WT"
echo "worktree: $(git -C "$WT" rev-parse --abbrev-ref HEAD) $(git -C "$WT" rev-parse HEAD)"
echo "node: $(hostname)  gensft_ckpt=$GENSFT_CKPT  n_probe=$N_PROBE  r_rewrite=$R_REWRITE  tag=$TAG"

env OGBENCH_DATA_DIR=/home/cymaxwelllee/.ogbench/data MUJOCO_GL=osmesa \
    CUDA_VISIBLE_DEVICES="" HIP_VISIBLE_DEVICES="" \
    "$PY" -u experiments/gen_sft/pose_permute_probe.py \
    --gensft-ckpt "$GENSFT_CKPT" --n-probe "$N_PROBE" --r-rewrite "$R_REWRITE" --tag "$TAG" \
    --out-dir experiments/gen_sft/results
