#!/bin/bash
# 2026-09-02 下午（草稿、⛔ 未經主人點頭不送）：定版配方（warmup500＋EMA、無 boot）上的方差溯源 —— stability 下一層
# 背景：C 批（舊配方＋boot）四顆裡兩顆災難級、兩個隨機源都會引爆 ⇒ 量到的是舊配方的脆弱，不是分級方差。
# V8 的分級方差（0.49~0.87、無災難）只有兩個隨機源：init 與主資料順序。
# D1 固定 init=s23（V8 最低顆）、換資料順序 DATA_SEED∈{101,102,103}
# D2 固定資料順序 DATA_SEED=23、換 init SEED∈{123,223,323}
# ⇒ 哪一組散得開，哪一個就是分級方差的病根。6 訓（三台 NVIDIA）＋6 ema eval（zeldajr、各自分目錄）。
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
mkdir -p slurm/logs
sub() { local node=$1 name=$2 deps=$3; shift 3; local depflag=""
  [ "$deps" != "-" ] && depflag="--dependency=afterok:$deps"
  sbatch -p admin -A it -q great-mage --time=24:00:00 --nodelist=$node --gres=gpu:1 \
    --job-name=$name -o slurm/logs/%x-%j.out $depflag \
    --wrap "cd ~/Projects/lacot && env $*" | awk '{print $4}'; }
NODES=(lady moana pocahontas); i=0
run_pair() {  # $1=name $2=SEED(init) $3=DATA_SEED
  local name=$1 S=$2 D=$3 NODE=${NODES[$((i%3))]}; i=$((i+1))
  local OUTD=results/night_0902/varsrc2_${name}; mkdir -p "$OUTD"
  local J=$(sub $NODE $name - "OGBENCH_DATA_DIR=$ADATA $TRAIN_BASE LACOT_ENV=$LENV LACOT_K=8 LACOT_SEED=$S LACOT_DATA_SEED=$D LACOT_EMA_W=0.999 LACOT_WARMUP=500 $APY -u experiments/scratch_lacot_rollout.py")
  local CK=results/${PRE}_eorecon_ictr_tch0.5_emw0.999_wu500_dseed${D}_norf_cd0.1_bci_s${S}.pt
  sub zeldajr $name-ema $J "OGBENCH_DATA_DIR=$ZDATA $BASE LACOT_ENV=$LENV LACOT_K=8 LACOT_TEACHER_MIX=0.5 LACOT_LOAD_EMA=1 LACOT_LOAD_CKPT=$CK LACOT_DIAG_DUMP=1 LACOT_OUT_DIR=$OUTD $OFF $C2MA $ZPY -u experiments/scratch_lacot_rollout.py" >/dev/null
  echo "$name (init=$S dseed=$D) -> $NODE ($J); eval -> zeldajr"
}
echo "== D1 固定 init=s23、換資料順序"
for D in 101 102 103; do run_pair D1-d$D 23 $D; done
echo "== D2 固定資料順序=23、換 init"
for S in 123 223 323; do run_pair D2-s$S $S 23; done
echo "== 共 12 支"
