#!/bin/bash
# 2026-09-02 15:5x（主人「跑」）：embedding 先做好 —— 兩帖各八顆（held-out s20~s27、定版配方 warmup500+EMA、無 boot）
# A. 長 stage 1：LACOT_STEPS1=6000（原 1500；V8 recon 到 1500 步仍在降）
# B. VQ 錨定：LACOT_VQ=64（stage 1 步數不動、只加 VQ；noise_p=0.1 保險）
# 驗收（事前註冊）：八顆 sd（對 V8 0.149）、平均（對 0.665）、爛顆、BC 通道、⭐ rt_gate 往返尺（每支 eval 自帶）
# eval：偶數顆 zeldajr、奇數顆同一台 NVIDIA；產物各分目錄（檔名已帶 _s16000 / _vq64）
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
mkdir -p slurm/logs results/night_0902/emb_s16000 results/night_0902/emb_vq64
sub() { local node=$1 name=$2 deps=$3; shift 3; local depflag=""
  [ "$deps" != "-" ] && depflag="--dependency=afterok:$deps"
  sbatch -p admin -A it -q great-mage --time=24:00:00 --nodelist=$node --gres=gpu:1 \
    --job-name=$name -o slurm/logs/%x-%j.out $depflag \
    --wrap "cd ~/Projects/lacot && env $*" | awk '{print $4}'; }
NODES=(lady moana pocahontas); i=0
run() {  # $1=arm(A|B) $2=seed
  local arm=$1 S=$2 NODE=${NODES[$((i%3))]}; i=$((i+1))
  if [ "$arm" = A ]; then EXTRA="LACOT_STEPS1=6000"; TAG="_s16000"; OUTD=results/night_0902/emb_s16000; NAME=EA-s$S
  else EXTRA="LACOT_VQ=64 LACOT_VQ_NOISE_P=0.1"; TAG="_vq64"; OUTD=results/night_0902/emb_vq64; NAME=EB-s$S; fi
  local J=$(sub $NODE $NAME - "OGBENCH_DATA_DIR=$ADATA $TRAIN_BASE LACOT_ENV=$LENV LACOT_K=8 LACOT_SEED=$S LACOT_EMA_W=0.999 LACOT_WARMUP=500 $EXTRA $APY -u experiments/scratch_lacot_rollout.py")
  local CK=results/${PRE}_eorecon_ictr_tch0.5_emw0.999_wu500${TAG}_norf_cd0.1_bci_s$S.pt
  if [ $((S % 2)) -eq 0 ]; then EN=zeldajr; EP=$ZPY; ED=$ZDATA; else EN=$NODE; EP=$APY; ED=$ADATA; fi
  sub $EN $NAME-ema $J "OGBENCH_DATA_DIR=$ED $BASE LACOT_ENV=$LENV LACOT_K=8 LACOT_TEACHER_MIX=0.5 LACOT_LOAD_EMA=1 LACOT_LOAD_CKPT=$CK LACOT_DIAG_DUMP=1 LACOT_OUT_DIR=$OUTD $OFF $C2MA $EP -u experiments/scratch_lacot_rollout.py" >/dev/null
  echo "$NAME -> $NODE ($J); eval -> $EN"
}
echo "== A 長 stage 1（6000）"; for S in 20 21 22 23 24 25 26 27; do run A $S; done
echo "== B VQ64";              for S in 20 21 22 23 24 25 26 27; do run B $S; done
echo "== 共 32 支（16 訓＋16 eval）"
