#!/bin/bash
# 9/6 午後：WS8 eval 重灑（zeldajr GPU hang 改道）— 全批 lady（Turing）、硬體一致
# ⚠️ LACOT_INTENT_ARMS=0：主檔工作區帶三腿 patch、判決 eval 不吃加時腿
set -euo pipefail
cd ~/Projects/lacot
APY=/archive/cymaxwelllee/LaCoT/.venv/bin/python
ADATA=/archive/cymaxwelllee/data/ogbench
BASE0="MUJOCO_GL=osmesa LACOT_ENC_OBJ=recon_ictr LACOT_LEARNED_REFINE=0 LACOT_BC_INDEP=1"
OFF="LACOT_DEV_EVAL=0 LACOT_EVAL_RS=0 LACOT_EVAL_EPISODES=50"
C2MA="LACOT_SUBGOAL=conf2 LACOT_SUB_POLICY=bc LACOT_GRAD_REFINE=1 LACOT_GRAD_R=0 LACOT_SUB_MAX_ARC=2 LACOT_FINISH_R=2.0"
LENV=pointmaze-large-stitch-v0
sub() { local name=$1 deps=$2; shift 2; local depflag=""
  [ "$deps" != "-" ] && depflag="--dependency=afterok:$deps"
  sbatch -p admin -A it -q great-mage --time=24:00:00 --nodelist=lady --gres=gpu:1 \
    --job-name=$name -o slurm/logs/%x-%j.out $depflag \
    --wrap "cd ~/Projects/lacot && env $*" | awk '{print $4}'; }
CKPRE="ckpt_large-stitch_self_K8_c256_ch4_st0_T128_ep2_gu_eorecon_ictr_tch0.5_btf27n_emw0.999_ite_idp0.3_ct4000_dssoft_norf_cd0.1_bci"
ep() { # $1=seed $2=ckdir $3=deps
  local S=$1 CK=$2/${CKPRE}_s$1.pt DEPS=$3
  local EVB="$BASE0 OGBENCH_DATA_DIR=$ADATA LACOT_COND_DROP=0.1 LACOT_ENV=$LENV LACOT_K=8 LACOT_TEACHER_MIX=0.5 LACOT_LOAD_EMA=1 LACOT_LOAD_CKPT=$CK LACOT_DEC_START=soft LACOT_INTENT=embed LACOT_BOOT_TAG=f27n LACOT_INTENT_DROP=0.3 LACOT_INTENT_ARMS=0"
  [ "$DEPS" = "-" ] && [ ! -f "$CK" ] && { echo "⛔ s$S ckpt 不在：$CK"; return 1; }
  sub WSE-s$S-on "$DEPS" "$EVB LACOT_OUT_DIR=results/day_0906/ws8_idp0.3 $OFF $C2MA $APY -u experiments/scratch_lacot_rollout.py" >/dev/null
  sub WSE-s$S-z0 "$DEPS" "$EVB LACOT_INTENT_ZERO=1 LACOT_OUT_DIR=results/day_0906/ws8_idp0.3_zero $OFF $C2MA $APY -u experiments/scratch_lacot_rollout.py" >/dev/null
  echo "eval s$S -> lady (deps=$DEPS)"
}
ep 40 results/day_0906/ws40 -
for S in 42 43 44 46 47; do ep $S results/day_0906/ws8_idp0.3 -; done
ep 41 results/day_0906/ws8_idp0.3 24759
ep 45 results/day_0906/ws8_idp0.3 24771
