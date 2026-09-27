#!/bin/bash
#SBATCH --job-name=j-signal
#SBATCH --array=0-5
#SBATCH --partition=ada-lite
#SBATCH --nodelist=jasmine
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --output=j_signal_%A_%a.out
set -euo pipefail

cd "$HOME/Projects/lacot"
ARCHIVE=/archive/cymaxwelllee
PY="$ARCHIVE/LaCoT/.venv/bin/python"
DATA="$ARCHIVE/data/ogbench"
case "$((SLURM_ARRAY_TASK_ID % 3))" in
  0) ARM=noclimb; GRAD_REFINE=0; BON_N=0; GRAD_MODE=climb; SEL_N=8 ;;
  1) ARM=climb; GRAD_REFINE=1; BON_N=0; GRAD_MODE=climb; SEL_N=8 ;;
  2) ARM=select; GRAD_REFINE=1; BON_N=0; GRAD_MODE=select; SEL_N=8 ;;
esac
case "$((SLURM_ARRAY_TASK_ID / 3))" in
  0) CKPT_SEED=33 ;;
  1) CKPT_SEED=35 ;;
esac
CKPT="results/ckpt_large-stitch_self_K8_c256_ch4_st8000_T128_ep2_gu_eorecon_ictr_tch0.5_emw0.999_wu500_dssoft_norf_cd0.1_bci_s${CKPT_SEED}.pt"
OUT=results/j_signal_trial

[ -x "$PY" ] || { echo "Missing local venv: $PY" >&2; exit 2; }
[ -f "$DATA/pointmaze-large-stitch-v0.npz" ] || { echo "Missing dataset: $DATA/pointmaze-large-stitch-v0.npz" >&2; exit 3; }
[ -f "$CKPT" ] || { echo "Missing ckpt: $CKPT" >&2; exit 4; }
if grep -Fq 'wall = self.wall_depth(pts).mean(1)' lacot/refine_grad.py; then
  echo 'F6 segment interpolation is absent in lacot/refine_grad.py' >&2
  exit 5
fi
mkdir -p "$OUT"
echo "arm=$ARM ckpt_seed=$CKPT_SEED ckpt=$CKPT"
exec env OGBENCH_DATA_DIR="$DATA" MUJOCO_GL=osmesa PYTHONUNBUFFERED=1 \
  LACOT_ENV=pointmaze-large-stitch-v0 LACOT_CONS=self LACOT_K=8 \
  LACOT_COND=256 LACOT_CHUNK=4 LACOT_TCAP=128 LACOT_ENC_OBJ=recon_ictr \
  LACOT_LEARNED_REFINE=0 LACOT_COND_DROP=0.1 LACOT_BC_INDEP=1 \
  LACOT_TEACHER_MIX=0.5 LACOT_WARMUP=500 LACOT_DEC_START=soft \
  LACOT_LOAD_EMA=1 LACOT_CONT_TRAIN=0 LACOT_DEV_EVAL=0 LACOT_PREREQ=0 \
  LACOT_EVAL_EPISODES=40 LACOT_EVAL_RS=1 \
  LACOT_GRAD_R=50 LACOT_GRAD_ETA=0.1 LACOT_GRAD_LAM=0.3 \
  LACOT_GRAD_R_WARM=10 LACOT_W_LEN=0.3 LACOT_GRAD_PROJ=0 \
  LACOT_SUBGOAL= LACOT_FINISH_R=0 LACOT_INTENT= LACOT_U_SOURCE=flow \
  LACOT_SUB_ESEL=0 LACOT_INTENT_GUID_W=0 LACOT_DEC_ANCHOR=0 \
  LACOT_BON_MODE=plan LACOT_BOOT_DATA= LACOT_S1_FROM= \
  LACOT_FLOW_PROBE=0 LACOT_DIAG_DUMP=0 \
  LACOT_GRAD_REFINE="$GRAD_REFINE" LACOT_BON_N="$BON_N" \
  LACOT_GRAD_MODE="$GRAD_MODE" LACOT_SEL_N="$SEL_N" \
  LACOT_LOAD_CKPT="$CKPT" LACOT_SEED="$CKPT_SEED" \
  LACOT_BOOT_TAG="jsignal_${ARM}" LACOT_OUT_DIR="$OUT" \
  "$PY" -u experiments/scratch_lacot_rollout.py
