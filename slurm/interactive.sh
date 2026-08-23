#!/bin/bash
#
# Drop into an interactive bash shell on the compute node "frieren" (Slurm).
#
# Usage:   ./slurm_scripts/interactive.sh
#
# Override via environment variables (defaults in brackets):
#   GPUS       GPUs to request (0 = CPU-only)          [1]
#   PARTITION  Slurm partition                         [blackwell]
#   NODELIST   node to run on                          [bocchi]
#   CPUS       CPUs per task                           [8]
#   MEM        memory, e.g. 64G (0 = all on the node)  [0]
#   TIME       wall-clock limit HH:MM:SS               [04:00:00]

set -euo pipefail

GPUS=${GPUS:-1}
PARTITION=${PARTITION:-blackwell}
NODELIST=${NODELIST:-bocchi}  # 2026-08-12: the six Blackwell GPUs moved frieren -> bocchi
CPUS=${CPUS:-8}
MEM=${MEM:-0}
TIME=${TIME:-04:00:00}

GRES_ARG=()
if [ "$GPUS" -gt 0 ]; then
    GRES_ARG=(--gres="gpu:${GPUS}")
fi

exec srun \
    --job-name=interactive \
    --partition="$PARTITION" \
    --nodelist="$NODELIST" \
    --nodes=1 --ntasks=1 \
    --cpus-per-task="$CPUS" \
    --mem="$MEM" \
    --time="$TIME" \
    "${GRES_ARG[@]}" \
    --pty bash
