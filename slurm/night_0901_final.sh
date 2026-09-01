#!/bin/bash
# 2026-09-01 過夜主菜（主人 22:39「好可以灑」）：
# 主菜：定版配方 8-seed 乾淨驗證 —— warmup500+EMA（事前註冊）× s20~s27（held-out 零挑選）
#   ⇒ ① 乾淨 8-seed 官方平均 vs QRL 84（paper 主數字）② 八顆散佈＝剩餘 stability 的尺
# 加菜：dz3 三輪補課 —— btdz2 顆（0.880）再生成 → 重訓 → eval+diag（curriculum 收斂＋task2）
# ⛔ eval 產物走 results/night_0901/final8（9/1 兩次互蓋的教訓：eval 端檔名不帶訓練 tag）
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
OUT="LACOT_OUT_DIR=results/night_0901/final8"
PRE=ckpt_large-stitch_self_K8_c256_ch4_st8000_T128_ep2_gu
mkdir -p results/night_0901/final8 results/night_0901/dz3 slurm/logs

sub() { local node=$1 name=$2 deps=$3; shift 3; local depflag=""
  [ "$deps" != "-" ] && depflag="--dependency=afterok:$deps"
  sbatch -p admin -A it -q great-mage --time=24:00:00 --nodelist=$node --gres=gpu:1 \
    --job-name=$name -o slurm/logs/%x-%j.out $depflag \
    --wrap "cd ~/Projects/lacot && env $*" | awk '{print $4}'; }

echo "== 主菜：定版配方 s20~s27"
i=0
for S in 20 21 22 23 24 25 26 27; do
  NODE=$(echo lady moana pocahontas | cut -d" " -f$((i%3+1))); i=$((i+1))
  J=$(sub $NODE V8-s$S - "OGBENCH_DATA_DIR=$ADATA $TRAIN_BASE LACOT_ENV=$LENV LACOT_K=8 LACOT_SEED=$S LACOT_EMA_W=0.999 LACOT_WARMUP=500 $APY -u experiments/scratch_lacot_rollout.py")
  CK=results/${PRE}_eorecon_ictr_tch0.5_emw0.999_wu500_norf_cd0.1_bci_s$S.pt
  sub zeldajr V8-s$S-ema $J "OGBENCH_DATA_DIR=$ZDATA $BASE LACOT_ENV=$LENV LACOT_K=8 LACOT_TEACHER_MIX=0.5 LACOT_LOAD_EMA=1 LACOT_LOAD_CKPT=$CK $OUT $OFF $C2MA $ZPY -u experiments/scratch_lacot_rollout.py" >/dev/null
  echo "V8-s$S -> $NODE ($J)"
done

echo "== 加菜：dz3 三輪補課"
DZ2=results/${PRE}_eorecon_ictr_tch0.5_btdz2_emw0.999_norf_cd0.1_bci_s2.pt
J1=$(sub moana Z3-gen - "OGBENCH_DATA_DIR=$ADATA $BASE LACOT_ENV=$LENV LACOT_K=8 LACOT_TEACHER_MIX=0.5 LACOT_LOAD_CKPT=$DZ2 LACOT_BOOT_GEN=results/boot_s2_dz3.npz LACOT_BOOT_DESERT=1 LACOT_BOOT_Q=512 LACOT_BOOT_M=8 LACOT_BOOT_RMIN=8 LACOT_BOOT_RMAX=25 $APY -u experiments/scratch_lacot_rollout.py")
J2=$(sub moana Z3-dz3 $J1 "OGBENCH_DATA_DIR=$ADATA $TRAIN_BASE LACOT_ENV=$LENV LACOT_K=8 LACOT_SEED=2 LACOT_EMA_W=0.999 LACOT_BOOT_DATA=results/boot_s2_dz3.npz LACOT_BOOT_TAG=dz3 $APY -u experiments/scratch_lacot_rollout.py")
sub zeldajr Z3-dz3-ema $J2 "OGBENCH_DATA_DIR=$ZDATA $BASE LACOT_ENV=$LENV LACOT_K=8 LACOT_TEACHER_MIX=0.5 LACOT_LOAD_EMA=1 LACOT_LOAD_CKPT=results/${PRE}_eorecon_ictr_tch0.5_btdz3_emw0.999_norf_cd0.1_bci_s2.pt LACOT_DIAG_DUMP=1 LACOT_OUT_DIR=results/night_0901/dz3 $OFF $C2MA $ZPY -u experiments/scratch_lacot_rollout.py" >/dev/null
echo "dz3 chain: $J1 -> $J2"
echo "== 全部排入：主菜 8訓+8評、加菜 1生+1訓+1評 ＝ 19 支"
