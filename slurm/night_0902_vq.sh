#!/bin/bash
# 2026-09-02（草稿、⛔ 未經主人點頭不送）：VQ 錨定 × 定版配方 × held-out 八顆（同 V8）× V∈{64,256}
# 事前註冊驗收（DESIGN-DRAFT-2026-09-02-vq-anchor.md）：主指標＝八顆 sd 從 0.149 縮多少；次＝平均（對 0.665）、爛顆、BC。
# 16 訓（三台 NVIDIA）＋16 ema eval（zeldajr 四卡＋NVIDIA 分攤）。eval 各 V 分目錄（檔名已帶 _vq）。
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
VQ_NOISE_P=${VQ_NOISE_P:-0.1}      # stochastic VQ 保險（主人裁）
mkdir -p slurm/logs
sub() { local node=$1 name=$2 deps=$3; shift 3; local depflag=""
  [ "$deps" != "-" ] && depflag="--dependency=afterok:$deps"
  sbatch -p admin -A it -q great-mage --time=24:00:00 --nodelist=$node --gres=gpu:1 \
    --job-name=$name -o slurm/logs/%x-%j.out $depflag \
    --wrap "cd ~/Projects/lacot && env $*" | awk '{print $4}'; }
NODES=(lady moana pocahontas); i=0
for V in 64 256; do
  OUTD=results/night_0902/vq$V; mkdir -p "$OUTD"
  echo "== VQ V=$V"
  for S in 20 21 22 23 24 25 26 27; do
    NODE=${NODES[$((i%3))]}; i=$((i+1))
    J=$(sub $NODE VQ$V-s$S - "OGBENCH_DATA_DIR=$ADATA $TRAIN_BASE LACOT_ENV=$LENV LACOT_K=8 LACOT_SEED=$S LACOT_EMA_W=0.999 LACOT_WARMUP=500 LACOT_VQ=$V LACOT_VQ_NOISE_P=$VQ_NOISE_P $APY -u experiments/scratch_lacot_rollout.py")
    CK=results/${PRE}_eorecon_ictr_tch0.5_emw0.999_wu500_vq${V}_norf_cd0.1_bci_s$S.pt
    # eval：偶數顆走 zeldajr、奇數顆走同一台 NVIDIA（分攤 zeldajr 的 25 分鐘 × 16）
    if [ $((S % 2)) -eq 0 ]; then EN=zeldajr; EP=$ZPY; ED=$ZDATA; else EN=$NODE; EP=$APY; ED=$ADATA; fi
    sub $EN VQ$V-s$S-ema $J "OGBENCH_DATA_DIR=$ED $BASE LACOT_ENV=$LENV LACOT_K=8 LACOT_TEACHER_MIX=0.5 LACOT_LOAD_EMA=1 LACOT_LOAD_CKPT=$CK LACOT_DIAG_DUMP=1 LACOT_OUT_DIR=$OUTD $OFF $C2MA $EP -u experiments/scratch_lacot_rollout.py" >/dev/null
    echo "VQ$V-s$S -> $NODE ($J); eval -> $EN"
  done
done
echo "== 共 32 支（16 訓＋16 eval）"
