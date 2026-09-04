#!/bin/bash
# 2026-09-04 intent 放量（主人「好跑下去」）：ITA/ITE/ITE0 三臂 × s41~47（s40 用先導那顆併入＝8 顆）
#   residual 判死收官不加碼。json 落先導同目錄（pilot_*），收表一起讀。
set -euo pipefail
cd ~/Projects/lacot
ZPY=$HOME/venvs/lacot-rocm/bin/python
APY=/archive/cymaxwelllee/LaCoT/.venv/bin/python
ZDATA=$HOME/data/ogbench
ADATA=/archive/cymaxwelllee/data/ogbench
BASE0="MUJOCO_GL=osmesa LACOT_ENC_OBJ=recon_ictr LACOT_LEARNED_REFINE=0 LACOT_BC_INDEP=1"
TRAIN0="LACOT_STEPS2=8000 LACOT_TEACHER_MIX=0.5 LACOT_EVAL_RS=0 LACOT_EVAL_EPISODES=2 LACOT_DIAG_TRAIN=1"
OFF="LACOT_DEV_EVAL=0 LACOT_EVAL_RS=0 LACOT_EVAL_EPISODES=50"
C2MA="LACOT_SUBGOAL=conf2 LACOT_SUB_POLICY=bc LACOT_GRAD_REFINE=1 LACOT_GRAD_R=0 LACOT_SUB_MAX_ARC=2 LACOT_FINISH_R=2.0"
LENV=pointmaze-large-stitch-v0
PRE=ckpt_large-stitch_self_K8_c256_ch4_st8000_T128_ep2_gu
SOFT20=results/${PRE}_eorecon_ictr_tch0.5_emw0.999_wu500_dssoft_norf_cd0.1_bci_s20.pt
FCK=results/night_0903/fsq/fsq_v2_d8L8_s20.pt
mkdir -p slurm/logs
sub() { local node=$1 name=$2 deps=$3; shift 3; local depflag=""
  [ "$deps" != "-" ] && depflag="--dependency=afterok:$deps"
  sbatch -p admin -A it -q great-mage --time=24:00:00 --nodelist=$node --gres=gpu:1 \
    --job-name=$name -o slurm/logs/%x-%j.out $depflag \
    --wrap "cd ~/Projects/lacot && env $*" | awk '{print $4}'; }
NODES=(jasmine lady moana pocahontas); i=0
FSQZ="LACOT_S1_FROM=$SOFT20 LACOT_FSQ_LOAD=$FCK LACOT_FSQ_SPACE=z LACOT_FSQ_TGT=snap LACOT_FSQ_ROUND=0"
EVF="LACOT_FSQ_LOAD=$FCK LACOT_FSQ_SPACE=z LACOT_FSQ_TGT=snap LACOT_FSQ_ROUND=0"

run_arm() { # $1=INTENT $2=fsq(1/0) $3=jobtag $4=cktag $5=outdir
  local IN=$1 F=$2 JT=$3 CT=$4 OUTD=results/day_0904/$5; mkdir -p "$OUTD"
  for S in 41 42 43 44 45 46 47; do
    local NODE=${NODES[$((i%4))]}; i=$((i+1))
    local TR="OGBENCH_DATA_DIR=$ADATA $BASE0 LACOT_COND_DROP=0.1 $TRAIN0 LACOT_ENV=$LENV LACOT_K=8 LACOT_SEED=$S LACOT_EMA_W=0.999 LACOT_WARMUP=500 LACOT_DEC_START=soft LACOT_INTENT=$IN"
    [ "$F" = 1 ] && TR="$TR $FSQZ"
    local J; J=$(sub $NODE $JT-s$S - "$TR $APY -u experiments/scratch_lacot_rollout.py")
    local SEG=""; [ "$F" = 1 ] && SEG="_s1from_fsqzc8x8"
    local CK=results/${PRE}_eorecon_ictr_tch0.5_emw0.999_wu500${SEG}_${CT}_dssoft_norf_cd0.1_bci_s$S.pt
    local EV="OGBENCH_DATA_DIR=$ZDATA $BASE0 LACOT_COND_DROP=0.1 LACOT_ENV=$LENV LACOT_K=8 LACOT_TEACHER_MIX=0.5 LACOT_LOAD_EMA=1 LACOT_LOAD_CKPT=$CK LACOT_DEC_START=soft LACOT_INTENT=$IN LACOT_OUT_DIR=$OUTD $OFF $C2MA"
    [ "$F" = 1 ] && EV="$EV $EVF"
    sub zeldajr $JT-s$S-ema $J "$EV $ZPY -u experiments/scratch_lacot_rollout.py" >/dev/null
    echo "$JT s$S -> $NODE ($J)"
  done
}

run_arm anchor 1 SCA ita pilot_ita
run_arm embed  1 SCE ite pilot_ite
run_arm embed  0 SC0 ite pilot_ite0
