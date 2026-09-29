#!/bin/bash
#SBATCH --job-name=ucontrast-zero
#SBATCH --partition=ada-lite
#SBATCH --nodelist=jasmine
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --output=experiments/_workorders/ucontrast1/smoke/mutant-%j.out
set -euo pipefail
cd "$HOME/Projects/lacot"
bash experiments/_workorders/ucontrast1/batch.sh mutant \
  --outdir experiments/_workorders/ucontrast1/smoke/mutant "$@"
