#!/bin/bash
# 2026-09-03 FSQ 批（主人裁「0.792 用比較好的 VQ 看能不能更好」＋「跑到傍晚都撒下去」）：
#   場＝凍 s20 soft stage1（同方言分辨批、同 seed s40~47 可逐顆配對）；fsq 兩檔 d8L8/d6L8（fit recon .016~.023 ≈ 無FSQ .019）。
#   A  snap-only：分辨批八顆 ckpt 直接 eval＋推論 snap（flow 不動）⇒ 「把 sample 拉回字彙格」值多少
#   B  dequant：stage 2 重訓（flow 目標＝格點＋均勻噪聲）＋eval snap ⇒ 「flow 在錨定字彙上學」值多少
#   對照：.792 sd .040（night_0903/dialect）。⛔ A/B eval 檔名靠 LACOT_FSQ_TAG 覆蓋＋分目錄雙保險。
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
mkdir -p slurm/logs
sub() { local node=$1 name=$2 deps=$3; shift 3; local depflag=""
  [ "$deps" != "-" ] && depflag="--dependency=afterok:$deps"
  sbatch -p admin -A it -q great-mage --time=24:00:00 --nodelist=$node --gres=gpu:1 \
    --job-name=$name -o slurm/logs/%x-%j.out $depflag \
    --wrap "cd ~/Projects/lacot && env $*" | awk '{print $4}'; }
NODES=(lady moana pocahontas); i=0
for FQ in d8L8 d6L8; do
  FCK=results/night_0903/fsq/fsq_${FQ}_s20.pt
  [ -f "$FCK" ] || { echo "⛔ fsq ckpt 不在：$FCK"; exit 1; }
  for ARM in ${ARMS:-A B}; do
    OUTD=results/night_0903/fsq${ARM}_${FQ}; mkdir -p "$OUTD"
    for S in ${SEEDS:-40 41 42 43 44 45 46 47}; do
      if [ "$ARM" = A ]; then
        # A：無訓練、直接 eval 分辨批 ckpt＋snap
        CK=results/${PRE}_eorecon_ictr_tch0.5_emw0.999_wu500_s1from_dssoft_norf_cd0.1_bci_s$S.pt
        if [ $((S % 2)) -eq 0 ]; then EN=zeldajr; EP=$ZPY; ED=$ZDATA; else EN=${NODES[$((i%3))]}; EP=$APY; ED=$ADATA; i=$((i+1)); fi
        sub $EN FA${FQ:1:1}-s$S - "OGBENCH_DATA_DIR=$ED $BASE LACOT_ENV=$LENV LACOT_K=8 LACOT_TEACHER_MIX=0.5 LACOT_LOAD_EMA=1 LACOT_LOAD_CKPT=$CK LACOT_DEC_START=soft LACOT_FSQ_LOAD=$FCK LACOT_FSQ_TAG=_fsqA${FQ} LACOT_OUT_DIR=$OUTD $OFF $C2MA $EP -u experiments/scratch_lacot_rollout.py" >/dev/null
        echo "FSQ-A $FQ s$S -> $EN"
      else
        # B：stage 2 重訓（dequant 目標）＋ eval（snap）
        NODE=${NODES[$((i%3))]}; i=$((i+1))
        J=$(sub $NODE FB${FQ:1:1}-s$S - "OGBENCH_DATA_DIR=$ADATA $TRAIN_BASE LACOT_ENV=$LENV LACOT_K=8 LACOT_SEED=$S LACOT_EMA_W=0.999 LACOT_WARMUP=500 LACOT_S1_FROM=$S1CK LACOT_DEC_START=soft LACOT_FSQ_LOAD=$FCK LACOT_FSQ_TGT=dequant $APY -u experiments/scratch_lacot_rollout.py")
        CKB=results/${PRE}_eorecon_ictr_tch0.5_emw0.999_wu500_s1from_fsqd${FQ:1:1}x8_dssoft_norf_cd0.1_bci_s$S.pt
        if [ $((S % 2)) -eq 0 ]; then EN=zeldajr; EP=$ZPY; ED=$ZDATA; else EN=$NODE; EP=$APY; ED=$ADATA; fi
        sub $EN FB${FQ:1:1}-s$S-ema $J "OGBENCH_DATA_DIR=$ED $BASE LACOT_ENV=$LENV LACOT_K=8 LACOT_TEACHER_MIX=0.5 LACOT_LOAD_EMA=1 LACOT_LOAD_CKPT=$CKB LACOT_DEC_START=soft LACOT_FSQ_LOAD=$FCK LACOT_FSQ_TAG=_fsqB${FQ} LACOT_OUT_DIR=$OUTD $OFF $C2MA $EP -u experiments/scratch_lacot_rollout.py" >/dev/null
        echo "FSQ-B $FQ s$S -> $NODE ($J); eval -> $EN"
      fi
    done
  done
done
