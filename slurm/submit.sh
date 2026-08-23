#!/bin/bash
#
# Submit a batch job to the compute node "frieren" on this Slurm cluster.
# (The old bv_scripts/*.sh use IBM LSF "bsub", which does not exist here.)
#
# Usage:   ./slurm_scripts/submit.sh <script_to_run> [script_args...]
#
# Override any of these via environment variables (defaults in brackets):
#   GPUS       GPUs to request (0 = CPU-only)          [1]
#   PARTITION  Slurm partition                         [blackwell]
#   NODELIST   node to run on                          [bocchi]
#   NODES      number of nodes                         [1]
#   CPUS       CPUs per task                           [10]
#   MEM        memory, e.g. 64G (0 = all on the node)  [0]
#   TIME       wall-clock limit HH:MM:SS               [02:00:00]
#              History: the owner asked on 2026-08-09 for the limits to be
#              removed, so this defaulted to the partition ceiling of 12
#              hours. That backfired, measurably. Slurm's backfill assumes a
#              running job may use its whole declared limit, so with 12 hours
#              declared everywhere it projects no free GPU for 12 hours and
#              refuses to slip ANY short job into a gap -- a two-minute
#              diagnostic then waits behind the entire queue. Reverted on
#              2026-08-10 to 2 hours, which is still about four times the
#              longest evaluation, so nothing is killed by the clock, while
#              backfill works again. Training wrappers ask for 6 hours (about
#              three times a 100k-step packed pair). Set TIME explicitly for
#              anything genuinely longer.
#   JOB_NAME   name shown in squeue                    [basename of script]
#   ACCOUNT    Slurm account (empty = your default)    []
#   DEPENDENCY Slurm --dependency value, e.g.
#              afterok:12345 (empty = none)             []
#
# Output goes to slurm_outputs/<jobid>.out and <jobid>.err
# The script is run with bash on the allocated node; args are forwarded safely.
#
# Example:  GPUS=2 TIME=02:00:00 ./slurm_scripts/submit.sh scripts/run_pipeline.sh
#           GPUS=0 ./slurm_scripts/submit.sh slurm_scripts/smoke.sh   # CPU-only check
#           DEPENDENCY=afterok:12345 ./slurm_scripts/submit.sh scripts/eval.sh   # chain after job 12345

set -euo pipefail

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <script_to_run> [script_args...]" >&2
    exit 1
fi

SCRIPT_TO_RUN=$1
shift

# The scheduler runs this script on the COMPUTE node, which has its own
# /tmp. A job script written to /tmp on the submitting host does not exist
# there, and the job dies instantly with exit 127 — a failure that looks
# like a crash in the work rather than a missing file. Refuse it here, where
# the message can say what to do, instead of in the job's stderr.
case "$(readlink -f "$SCRIPT_TO_RUN")" in
    /tmp/*|/var/tmp/*|/dev/shm/*|/run/*)
        echo "refusing to submit $SCRIPT_TO_RUN: it lives on node-local storage." >&2
        echo "The compute node has its own /tmp, so the file will not exist there" >&2
        echo "and the job will fail with exit 127. Put job scripts on the shared" >&2
        echo "filesystem (e.g. scripts/oneoff/) and submit that path instead." >&2
        exit 1
        ;;
esac

GPUS=${GPUS:-1}
PARTITION=${PARTITION:-blackwell}
NODELIST=${NODELIST:-bocchi}  # 2026-08-12: the six Blackwell GPUs moved frieren -> bocchi
NODES=${NODES:-1}
CPUS=${CPUS:-10}  # 2026-08-13: partition caps CPU at 10 per GPU (job_submit rule); wrappers set OMP threads from the allocation
MEM=${MEM:-0}
TIME=${TIME:-02:00:00}
JOB_NAME=${JOB_NAME:-$(basename "$SCRIPT_TO_RUN")}

mkdir -p ./slurm_outputs

GRES_ARG=()
if [ "$GPUS" -gt 0 ]; then
    GRES_ARG=(--gres="gpu:${GPUS}")
fi

ACCOUNT_ARG=()
if [ -n "${ACCOUNT:-}" ]; then
    ACCOUNT_ARG=(--account="$ACCOUNT")
fi

DEPENDENCY_ARG=()
if [ -n "${DEPENDENCY:-}" ]; then
    DEPENDENCY_ARG=(--dependency="$DEPENDENCY")
fi

# Force bash and forward args safely (printf %q quotes each token).
INNER_CMD=$(printf '%q ' bash "$SCRIPT_TO_RUN" "$@")

sbatch \
    --job-name="$JOB_NAME" \
    --partition="$PARTITION" \
    --nodelist="$NODELIST" \
    --nodes="$NODES" \
    --ntasks-per-node=1 \
    --cpus-per-task="$CPUS" \
    --mem="$MEM" \
    --time="$TIME" \
    "${GRES_ARG[@]}" \
    "${ACCOUNT_ARG[@]}" \
    "${DEPENDENCY_ARG[@]}" \
    --output=./slurm_outputs/%j.out \
    --error=./slurm_outputs/%j.err \
    --wrap "$INNER_CMD"
