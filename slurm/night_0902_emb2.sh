#!/bin/bash
# 2026-09-02 晚（草稿、⛔ 未經主人點頭不送）：embedding 第二輪，臂用 ARMS 環境變數選（空白分隔）：
#   S256   軟錨 VQ256（commitment 拉向字彙、不硬量化）                       LACOT_VQ=256 LACOT_VQ_SOFT=1
#   H256L  硬量化 VQ256 ＋ 長 stage1 6000（給量化時間與容量）               LACOT_VQ=256 LACOT_STEPS1=6000
#   LL     長 stage1 6000 ＋ 長 stage2 16000（u 空間變細 ⇒ flow 也要練久）    LACOT_STEPS1=6000 LACOT_STEPS2=16000
# 每臂 held-out s20~s27 八顆、定版配方 warmup500+EMA、無 boot；eval 偶數顆 zeldajr、奇數顆同台 NVIDIA。
# 用法：ARMS="S256 LL" bash slurm/night_0902_emb2.sh
set -euo pipefail
cd ~/Projects/lacot
ARMS=${ARMS:-"S256"}
ZPY=$HOME/venvs/lacot-rocm/bin/python
APY=/archive/cymaxwelllee/LaCoT/.venv/bin/python
ZDATA=$HOME/data/ogbench
ADATA=/archive/cymaxwelllee/data/ogbench
BASE="MUJOCO_GL=osmesa LACOT_ENC_OBJ=recon_ictr LACOT_LEARNED_REFINE=0 LACOT_COND_DROP=0.1 LACOT_BC_INDEP=1"
TRAIN_COMMON="$BASE LACOT_TEACHER_MIX=0.5 LACOT_EVAL_RS=0 LACOT_EVAL_EPISODES=2 LACOT_DIAG_TRAIN=1 LACOT_EMA_W=0.999 LACOT_WARMUP=500"
OFF="LACOT_DEV_EVAL=0 LACOT_EVAL_RS=0 LACOT_EVAL_EPISODES=50"
C2MA="LACOT_SUBGOAL=conf2 LACOT_SUB_POLICY=bc LACOT_GRAD_REFINE=1 LACOT_GRAD_R=0 LACOT_SUB_MAX_ARC=2 LACOT_FINISH_R=2.0"
LENV=pointmaze-large-stitch-v0
mkdir -p slurm/logs
sub() { local node=$1 name=$2 deps=$3; shift 3; local depflag=""
  [ "$deps" != "-" ] && depflag="--dependency=afterok:$deps"
  sbatch -p admin -A it -q great-mage --time=24:00:00 --nodelist=$node --gres=gpu:1 \
    --job-name=$name -o slurm/logs/%x-%j.out $depflag \
    --wrap "cd ~/Projects/lacot && env $*" | awk '{print $4}'; }
NODES=(lady moana pocahontas); i=0
arm_cfg() {  # 印：EXTRA|TAG|ST2
  case $1 in
    S256)  echo "LACOT_VQ=256 LACOT_VQ_SOFT=1 LACOT_VQ_NOISE_P=0.1|_vq256s|8000" ;;
    H256L) echo "LACOT_VQ=256 LACOT_VQ_NOISE_P=0.1 LACOT_STEPS1=6000|_vq256_s16000|8000" ;;
    LL)    echo "LACOT_STEPS1=6000 LACOT_STEPS2=16000|_s16000|16000" ;;
    *) echo "⛔ 不認得的臂 $1" >&2; exit 1 ;;
  esac
}
for ARM in $ARMS; do
  IFS='|' read -r EXTRA TAG ST2 <<< "$(arm_cfg $ARM)"
  OUTD=results/night_0902/emb2_$ARM; mkdir -p "$OUTD"
  echo "== $ARM: $EXTRA"
  for S in 20 21 22 23 24 25 26 27; do
    NODE=${NODES[$((i%3))]}; i=$((i+1)); NAME=E2${ARM}-s$S
    J=$(sub $NODE $NAME - "OGBENCH_DATA_DIR=$ADATA $TRAIN_COMMON LACOT_STEPS2=$ST2 LACOT_ENV=$LENV LACOT_K=8 LACOT_SEED=$S $EXTRA $APY -u experiments/scratch_lacot_rollout.py")
    CK=results/ckpt_large-stitch_self_K8_c256_ch4_st${ST2}_T128_ep2_gu_eorecon_ictr_tch0.5_emw0.999_wu500${TAG}_norf_cd0.1_bci_s$S.pt
    if [ $((S % 2)) -eq 0 ]; then EN=zeldajr; EP=$ZPY; ED=$ZDATA; else EN=$NODE; EP=$APY; ED=$ADATA; fi
    sub $EN $NAME-ema $J "OGBENCH_DATA_DIR=$ED $BASE LACOT_ENV=$LENV LACOT_K=8 LACOT_TEACHER_MIX=0.5 LACOT_LOAD_EMA=1 LACOT_LOAD_CKPT=$CK LACOT_DIAG_DUMP=1 LACOT_OUT_DIR=$OUTD $OFF $C2MA $EP -u experiments/scratch_lacot_rollout.py" >/dev/null
    echo "$NAME -> $NODE ($J); eval -> $EN"
  done
done
