#!/bin/bash
# 2026-09-03 方言分辨實驗（主人 9/2 23:25 核「明天跑」；決策 10）：
#   凍同一個 stage 1（s20 soft ckpt、往返尺 .019 最好）、只讓 stage 2 換 seed s40~47。
#   讀法：散佈仍大 ⇒ 抽籤住在 stage 2 訓練裡（VQ＋分類有道理）；
#         散佈變小 ⇒ 住在 u 空間形狀裡（要選 FSQ 這種字彙固定的）。
#   對照：soft 八顆 .752 sd .168。用法：SEEDS="40" 先灑第一支，跑通後 SEEDS="41 ... 47" 灑其餘。
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
S1CK=results/ckpt_large-stitch_self_K8_c256_ch4_st8000_T128_ep2_gu_eorecon_ictr_tch0.5_emw0.999_wu500_dssoft_norf_cd0.1_bci_s20.pt
[ -f "$S1CK" ] || { echo "⛔ S1_FROM ckpt 不在：$S1CK"; exit 1; }
OUTD=results/night_0903/dialect; mkdir -p "$OUTD" slurm/logs
sub() { local node=$1 name=$2 deps=$3; shift 3; local depflag=""
  [ "$deps" != "-" ] && depflag="--dependency=afterok:$deps"
  sbatch -p admin -A it -q great-mage --time=24:00:00 --nodelist=$node --gres=gpu:1 \
    --job-name=$name -o slurm/logs/%x-%j.out $depflag \
    --wrap "cd ~/Projects/lacot && env $*" | awk '{print $4}'; }
NODES=(lady moana pocahontas); i=0
for S in ${SEEDS:-40 41 42 43 44 45 46 47}; do
  NODE=${NODES[$((i%3))]}; i=$((i+1))
  J=$(sub $NODE DL-s$S - "OGBENCH_DATA_DIR=$ADATA $TRAIN_BASE LACOT_ENV=$LENV LACOT_K=8 LACOT_SEED=$S LACOT_EMA_W=0.999 LACOT_WARMUP=500 LACOT_S1_FROM=$S1CK LACOT_DEC_START=soft $APY -u experiments/scratch_lacot_rollout.py")
  CK=results/${PRE}_eorecon_ictr_tch0.5_emw0.999_wu500_s1from_dssoft_norf_cd0.1_bci_s$S.pt
  if [ $((S % 2)) -eq 0 ]; then EN=zeldajr; EP=$ZPY; ED=$ZDATA; else EN=$NODE; EP=$APY; ED=$ADATA; fi
  sub $EN DL-s$S-ema $J "OGBENCH_DATA_DIR=$ED $BASE LACOT_ENV=$LENV LACOT_K=8 LACOT_TEACHER_MIX=0.5 LACOT_LOAD_EMA=1 LACOT_LOAD_CKPT=$CK LACOT_DEC_START=soft LACOT_DIAG_DUMP=1 LACOT_FLOW_PROBE=32 LACOT_OUT_DIR=$OUTD $OFF $C2MA $EP -u experiments/scratch_lacot_rollout.py" >/dev/null
  echo "DL-s$S -> $NODE ($J); eval -> $EN"
done
