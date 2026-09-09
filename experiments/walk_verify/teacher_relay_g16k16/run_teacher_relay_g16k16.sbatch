#!/usr/bin/env bash
#SBATCH -J TR-G16K16
#SBATCH -p admin
#SBATCH -A it
#SBATCH -q great-mage
#SBATCH --nodelist=zeldajr
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH -o /home/cymaxwelllee/Projects/lacot/experiments/walk_verify/teacher_relay_g16k16/logs/%x-%j.out
# G16K16 教材 hindsight 真字接力驗證（schedule 開環版）—— 對照 K32 .5536。
# 見 docs/NOTE-2026-09-09-teacher-g16k16.md。CPU-only，不掛 gres（任務要求）。
set -u
PY=/home/cymaxwelllee/venvs/lacot-rocm/bin/python
cd /home/cymaxwelllee/Projects/lacot
export OGBENCH_DATA_DIR=/home/cymaxwelllee/.ogbench/data
export MUJOCO_GL=osmesa
export CUDA_VISIBLE_DEVICES=
export HIP_VISIBLE_DEVICES=

echo "== host: $(hostname)  job: $SLURM_JOB_ID  cpus: $SLURM_CPUS_PER_TASK =="
"$PY" -u experiments/walk_verify/teacher_relay_g16k16/run_teacher_relay_g16k16.py --threads "${SLURM_CPUS_PER_TASK:-8}"
rc=$?
echo "=== sbatch script exit code: $rc ==="
exit $rc
