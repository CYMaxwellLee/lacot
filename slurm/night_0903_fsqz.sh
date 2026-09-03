#!/bin/bash
# 2026-09-03 晚 FSQ z 空間版（主人裁「修好、檢查、再開始跑」；檢查三格 18:1x 全綠：
#   刻度恰 8（半整數格＋clamp）、z 佔格每維 8/8（batch 424 種格點）、fit recon .0177 ≈ 無FSQ .019）。
#   flow 在 8 維字典座標上建模（滿秩 ⇒ 無白天臂 B 薄片病）；甲＝學連續 z（推論才 round）、乙＝學格點+噪聲。
#   場＝凍 s20 soft stage1、seed s40~47、對 .792 sd .040 逐顆配對；fsq＝fsq_v2_d8L8_s20.pt。
set -euo pipefail
cd ~/Projects/lacot
ZPY=$HOME/venvs/lacot-rocm/bin/python
APY=/archive/cymaxwelllee/LaCoT/.venv/bin/python
ZDATA=$HOME/data/ogbench
ADATA=/archive/cymaxwelllee/data/ogbench
BASE="MUJOCO_GL=osmesa LACOT_ENC_OBJ=recon_ictr LACOT_LEARNED_REFINE=0 LACOT_COND_DROP=0.1 LACOT_BC_INDEP=1"
TRAIN_BASE="$BASE LACOT_STEPS2=8000 LACOT_TEACHER_MIX=0.5 LACOT_EVAL_RS=0 LACOT_EVAL_EPISODES=2 LACOT_DIAG_TRAIN=1"
OFF="LACOT_DEV_EVAL=0 LACOT_EVAL_RS=0 LACOT_EVAL_EPISODES=50"
C2MA="LACOT_SUBGOAL=conf2 LACOT_SUB_POLICY=bc LACOT_GRAD_REFINE=1 LACOT_GRAD_R=0 LACOT_SUB_MAX_ARC=2 LACOT_FINISH_R=2.0"
LENV=pointmaze-large-stitch-v0
PRE=ckpt_large-stitch_self_K8_c256_ch4_st8000_T128_ep2_gu
S1CK=results/${PRE}_eorecon_ictr_tch0.5_emw0.999_wu500_dssoft_norf_cd0.1_bci_s20.pt
FCK=results/night_0903/fsq/fsq_v2_d8L8_s20.pt
[ -f "$FCK" ] || { echo "⛔ fsq v2 ckpt 不在：$FCK"; exit 1; }
mkdir -p slurm/logs
sub() { local node=$1 name=$2 deps=$3; shift 3; local depflag=""
  [ "$deps" != "-" ] && depflag="--dependency=afterok:$deps"
  sbatch -p admin -A it -q great-mage --time=24:00:00 --nodelist=$node --gres=gpu:1 \
    --job-name=$name -o slurm/logs/%x-%j.out $depflag \
    --wrap "cd ~/Projects/lacot && env $*" | awk '{print $4}'; }
NODES=(lady moana pocahontas); i=0
for TGT in ${TGTS:-snap dequant}; do
  A=$([ $TGT = snap ] && echo ZA || echo ZB); ZTAG=$([ $TGT = snap ] && echo fsqz8x8 || echo fsqzd8x8)
  OUTD=results/night_0903/fsqz_$TGT; mkdir -p "$OUTD"
  for S in ${SEEDS:-40 41 42 43 44 45 46 47}; do
    NODE=${NODES[$((i%3))]}; i=$((i+1))
    J=$(sub $NODE $A-s$S - "OGBENCH_DATA_DIR=$ADATA $TRAIN_BASE LACOT_ENV=$LENV LACOT_K=8 LACOT_SEED=$S LACOT_EMA_W=0.999 LACOT_WARMUP=500 LACOT_S1_FROM=$S1CK LACOT_DEC_START=soft LACOT_FSQ_LOAD=$FCK LACOT_FSQ_SPACE=z LACOT_FSQ_TGT=$TGT $APY -u experiments/scratch_lacot_rollout.py")
    CKZ=results/${PRE}_eorecon_ictr_tch0.5_emw0.999_wu500_s1from_${ZTAG}_dssoft_norf_cd0.1_bci_s$S.pt
    if [ $((S % 2)) -eq 0 ]; then EN=zeldajr; EP=$ZPY; ED=$ZDATA; else EN=$NODE; EP=$APY; ED=$ADATA; fi
    sub $EN $A-s$S-ema $J "OGBENCH_DATA_DIR=$ED $BASE LACOT_ENV=$LENV LACOT_K=8 LACOT_TEACHER_MIX=0.5 LACOT_LOAD_EMA=1 LACOT_LOAD_CKPT=$CKZ LACOT_DEC_START=soft LACOT_FSQ_LOAD=$FCK LACOT_FSQ_SPACE=z LACOT_FSQ_TGT=$TGT LACOT_OUT_DIR=$OUTD $OFF $C2MA $EP -u experiments/scratch_lacot_rollout.py" >/dev/null
    echo "$A($TGT) s$S -> $NODE ($J); eval -> $EN"
  done
done
