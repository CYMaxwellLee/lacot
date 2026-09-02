#!/bin/bash
# 2026-09-02 下午（草稿、⛔ 未經主人點頭不送）：s2 型方差溯源 —— 拆「主資料順序」vs「boot 抽到哪些條」
# 背景：B 批（dz2 重現）同 rng 0.904、只換 DATA_SEED 0.35~0.56 ⇒ 方差源在抽樣；但 DATA_SEED 同時管
#       主資料順序與 boot 抽樣（同一條 rng）。LACOT_BOOT_SEED（9/2 新開關）讓 boot 抽樣走獨立流。
# C1 只動 boot 抽樣：DATA_SEED=2 固定、BOOT_SEED∈{1,2,3}
# C2 只動主資料順序：BOOT_SEED=1 固定、DATA_SEED∈{12,22,32}
# ⇒ 哪一組散得開，哪一個就是病根。共 6 訓（四台 NVIDIA）＋ 6 ema eval（zeldajr，各自分目錄防互蓋）。
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
BOOT=results/boot_s2_dz2.npz
[ -f "$BOOT" ] || { echo "⛔ 缺 $BOOT"; exit 1; }
mkdir -p slurm/logs

sub() { local node=$1 name=$2 deps=$3; shift 3; local depflag=""
  [ "$deps" != "-" ] && depflag="--dependency=afterok:$deps"
  sbatch -p admin -A it -q great-mage --time=24:00:00 --nodelist=$node --gres=gpu:1 \
    --job-name=$name -o slurm/logs/%x-%j.out $depflag \
    --wrap "cd ~/Projects/lacot && env $*" | awk '{print $4}'; }

NODES=(lady moana pocahontas jasmine); i=0
run_pair() {  # $1=name $2=DATA_SEED $3=BOOT_SEED
  local name=$1 D=$2 B=$3 NODE=${NODES[$((i%4))]}; i=$((i+1))
  local OUTD=results/night_0902/varsrc_${name}; mkdir -p "$OUTD"
  local J=$(sub $NODE $name - "OGBENCH_DATA_DIR=$ADATA $TRAIN_BASE LACOT_ENV=$LENV LACOT_K=8 LACOT_SEED=2 LACOT_DATA_SEED=$D LACOT_BOOT_SEED=$B LACOT_EMA_W=0.999 LACOT_BOOT_DATA=$BOOT LACOT_BOOT_TAG=dz2 $APY -u experiments/scratch_lacot_rollout.py")
  local CK=results/${PRE}_eorecon_ictr_tch0.5_btdz2_emw0.999_dseed${D}_bseed${B}_norf_cd0.1_bci_s2.pt
  sub zeldajr $name-ema $J "OGBENCH_DATA_DIR=$ZDATA $BASE LACOT_ENV=$LENV LACOT_K=8 LACOT_TEACHER_MIX=0.5 LACOT_LOAD_EMA=1 LACOT_LOAD_CKPT=$CK LACOT_DIAG_DUMP=1 LACOT_OUT_DIR=$OUTD $OFF $C2MA $ZPY -u experiments/scratch_lacot_rollout.py" >/dev/null
  echo "$name (dseed=$D bseed=$B) -> $NODE ($J); eval -> zeldajr"
}
echo "== C1 只動 boot 抽樣（DATA_SEED=2）"
for B in 1 2 3; do run_pair C1-b$B 2 $B; done
echo "== C2 只動主資料順序（BOOT_SEED=1）"
for D in 12 22 32; do run_pair C2-d$D $D 1; done
echo "== 共 12 支"
