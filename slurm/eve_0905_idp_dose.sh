#!/bin/bash
# 2026-09-05 晚批：內化劑量二臂（診斷鏈收攏後的正主實驗；四問已過、主人核流程）
#   臂A idp01＝p=0.1 × s40~47（曝光 90%）＋成對雙 eval（帶/零）
#   臂B idpxm＝p=0.3 曝光匹配單顆 s40（STEPS2=11429≈8000/0.7）＋成對雙 eval
#   判讀先釘：A 帶查 R0 回 .42+ ⇒ 增益長回、讀零模式=內化殘留；B 回 .42+ ⇒ 病=曝光量；
#   兩者皆不回 ⇒ dropout 結構本身妨礙 ⇒ 上訓練期藥（π-Distill 雙模式壓力）。
set -euo pipefail
cd ~/Projects/lacot
ZPY=$HOME/venvs/lacot-rocm/bin/python
APY=/archive/cymaxwelllee/LaCoT/.venv/bin/python
ZDATA=$HOME/data/ogbench
ADATA=/archive/cymaxwelllee/data/ogbench
BASE0="MUJOCO_GL=osmesa LACOT_ENC_OBJ=recon_ictr LACOT_LEARNED_REFINE=0 LACOT_BC_INDEP=1"
OFF="LACOT_DEV_EVAL=0 LACOT_EVAL_RS=0 LACOT_EVAL_EPISODES=50"
C2MA="LACOT_SUBGOAL=conf2 LACOT_SUB_POLICY=bc LACOT_GRAD_REFINE=1 LACOT_GRAD_R=0 LACOT_SUB_MAX_ARC=2 LACOT_FINISH_R=2.0"
LENV=pointmaze-large-stitch-v0
SOFT27=results/ckpt_large-stitch_self_K8_c256_ch4_st8000_T128_ep2_gu_eorecon_ictr_tch0.5_emw0.999_wu500_dssoft_norf_cd0.1_bci_s27.pt
mkdir -p slurm/logs results/day_0905/{idp01,idp01_zero,idpxm,idpxm_zero}
sub() { local node=$1 name=$2 deps=$3; shift 3; local depflag=""
  [ "$deps" != "-" ] && depflag="--dependency=afterok:$deps"
  sbatch -p admin -A it -q great-mage --time=24:00:00 --nodelist=$node --gres=gpu:1 \
    --job-name=$name -o slurm/logs/%x-%j.out $depflag \
    --wrap "cd ~/Projects/lacot && env $*" | awk '{print $4}'; }
NODES=(jasmine lady moana pocahontas); i=0
run_pair() { # $1=jobtag $2=seed $3=steps2 $4=drop $5=bt $6=outbase $7=ck_full
  local JT=$1 S=$2 ST=$3 DP=$4 BT=$5 OB=$6 CK=$7
  local NODE=${NODES[$((i%4))]}; i=$((i+1))
  local TRAIN0="LACOT_STEPS2=$ST LACOT_TEACHER_MIX=0.5 LACOT_EVAL_RS=0 LACOT_EVAL_EPISODES=2 LACOT_DIAG_TRAIN=1"
  local J; J=$(sub $NODE $JT-s$S - "OGBENCH_DATA_DIR=$ADATA $BASE0 LACOT_COND_DROP=0.1 $TRAIN0 LACOT_ENV=$LENV LACOT_K=8 LACOT_SEED=$S LACOT_EMA_W=0.999 LACOT_WARMUP=500 LACOT_DEC_START=soft LACOT_S1_FROM=$SOFT27 LACOT_INTENT=embed LACOT_INTENT_DROP=$DP LACOT_BOOT_TAG=$BT $APY -u experiments/scratch_lacot_rollout.py")
  local EVBASE="OGBENCH_DATA_DIR=$ZDATA $BASE0 LACOT_COND_DROP=0.1 LACOT_ENV=$LENV LACOT_K=8 LACOT_TEACHER_MIX=0.5 LACOT_LOAD_EMA=1 LACOT_LOAD_CKPT=$CK LACOT_DEC_START=soft LACOT_INTENT=embed LACOT_INTENT_DROP=$DP LACOT_BOOT_TAG=$BT"
  sub zeldajr $JT-s$S-on $J "$EVBASE LACOT_OUT_DIR=results/day_0905/$OB $OFF $C2MA $ZPY -u experiments/scratch_lacot_rollout.py" >/dev/null
  sub zeldajr $JT-s$S-z0 $J "$EVBASE LACOT_INTENT_ZERO=1 LACOT_OUT_DIR=results/day_0905/${OB}_zero $OFF $C2MA $ZPY -u experiments/scratch_lacot_rollout.py" >/dev/null
  echo "$JT s$S steps=$ST drop=$DP -> $NODE ($J)"
}
PRE8=ckpt_large-stitch_self_K8_c256_ch4_st8000_T128_ep2_gu
PREX=ckpt_large-stitch_self_K8_c256_ch4_st11429_T128_ep2_gu
for S in 40 41 42 43 44 45 46 47; do
  run_pair IDP01 $S 8000 0.1 idp01 idp01 results/${PRE8}_eorecon_ictr_tch0.5_btidp01_emw0.999_wu500_s1from_ite_idp0.1_dssoft_norf_cd0.1_bci_s$S.pt
done
run_pair IDPXM 40 11429 0.3 idpxm idpxm results/${PREX}_eorecon_ictr_tch0.5_btidpxm_emw0.999_wu500_s1from_ite_idp0.3_dssoft_norf_cd0.1_bci_s40.pt
