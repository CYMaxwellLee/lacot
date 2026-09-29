#!/bin/bash
#SBATCH --job-name=ucontrast-smoke
#SBATCH --array=0-3
#SBATCH --partition=ada-lite
#SBATCH --nodelist=jasmine
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --output=experiments/_workorders/ucontrast1/smoke/sweep-%A_%a.out
set -euo pipefail
cd "$HOME/Projects/lacot"
# 0=A; 1/2/3=B at sigma .05/.10/.20. 5 official tasks x 5 episodes x 8 draws.
bash experiments/_workorders/ucontrast1/batch.sh smoke \
  --index "$SLURM_ARRAY_TASK_ID" --outdir experiments/_workorders/ucontrast1/smoke/sweep "$@"
