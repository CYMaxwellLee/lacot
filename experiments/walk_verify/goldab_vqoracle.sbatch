#!/usr/bin/env bash
#SBATCH -J GOLDAB-vqo
#SBATCH -p admin
#SBATCH -A it
#SBATCH -q great-mage
#SBATCH --nodelist=zeldajr
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=1:00:00
#SBATCH -o /home/cymaxwelllee/Projects/lacot/experiments/walk_verify/logs/goldab_vqo-%j.out
# P2 施工驗收：golden A/A/B（main vs vq-oracle-v1 worktree）。
# ⛔ GOLD_OUT 指到新目錄 —— gold_ab_wrap.sh 開頭會 rm -rf 輸出目錄，
#    用預設值會刪掉 lo-v1 那次的既有證據。
set -u
export BRANCH_ROOT=/home/cymaxwelllee/Projects/lacot-vqoracle
export GOLD_OUT=/home/cymaxwelllee/Projects/lacot/results/goldab_vqoracle
bash /home/cymaxwelllee/Projects/lacot/slurm/goldab/gold_ab_wrap.sh
