#!/bin/bash
# 封條解除（lead 2026-09-28）：gate 鏈全過（flagoff/calibration/mutant/sweep+selection σ=0.05）。
# 產物路徑改 jasmine 本機 /archive（主人 9/28 儲存裁示）；log 同。
#SBATCH --job-name=ucontrast-main
#SBATCH --array=0-1
#SBATCH --partition=ada-lite
#SBATCH --nodelist=jasmine
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --output=/archive/cymaxwelllee/ucontrast1/logs/main-%A_%a.out
set -euo pipefail
cd "$HOME/Projects/lacot"
# Default s33. Later s35: pass --seed 35 and its separately selected sigma file.
# Each index executes 200 questions x 8 complete rollouts, A or winning-sigma B.
bash experiments/_workorders/ucontrast1/batch.sh main \
  --index "$SLURM_ARRAY_TASK_ID" --outdir /archive/cymaxwelllee/ucontrast1/main "$@"
