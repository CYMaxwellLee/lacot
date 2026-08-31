#!/bin/bash
# 檢討實驗 2026-08-31 夜（RETRO-2026-08-31.md ④；跑到 9/1 中午）
# R1 DS 軸 dev 尺重掃（洗 test-tuning）：DS 3/4/5/7.5 × medium s0/s1/s2 顆＝12 支 dev
# R2 EMA 嚴格配對補完：s3~s7 EMA 顆的 raw 臂官方＝5 支
# R3 bcown fork_rng 修後重訓驗證：medium s1＋large s0 重訓 → 官方（分段應回常軌）
set -euo pipefail
cd ~/Projects/lacot

ZPY=$HOME/venvs/lacot-rocm/bin/python
APY=/archive/cymaxwelllee/LaCoT/.venv/bin/python
ZDATA=$HOME/data/ogbench
ADATA=/archive/cymaxwelllee/data/ogbench
BASE="MUJOCO_GL=osmesa LACOT_ENC_OBJ=recon_ictr LACOT_LEARNED_REFINE=0 LACOT_COND_DROP=0.1 LACOT_BC_INDEP=1"
OFF="LACOT_DEV_EVAL=0 LACOT_EVAL_RS=0 LACOT_EVAL_EPISODES=50"
DEV="LACOT_DEV_EVAL=1 LACOT_DEV_TIERS=2 LACOT_DEV_PER_TIER=100 LACOT_EVAL_RS=0 LACOT_EVAL_EPISODES=2"
C2MA="LACOT_SUBGOAL=conf2 LACOT_SUB_POLICY=bc LACOT_GRAD_REFINE=1 LACOT_GRAD_R=0 LACOT_SUB_MAX_ARC=2 LACOT_FINISH_R=2.0"
LENV=pointmaze-large-stitch-v0
MENV=pointmaze-medium-stitch-v0
TRAIN_BASE="$BASE LACOT_STEPS2=8000 LACOT_TEACHER_MIX=0.5 LACOT_EVAL_RS=0 LACOT_EVAL_EPISODES=2"

sub() {
  local node=$1 name=$2 deps=$3; shift 3
  local depflag=""
  [ "$deps" != "-" ] && depflag="--dependency=afterok:$deps"
  sbatch -p admin -A it -q great-mage --time=24:00:00 --nodelist=$node --gres=gpu:1 \
    --job-name=$name -o slurm/logs/%x-%j.out $depflag \
    --wrap "cd ~/Projects/lacot && env $*" | awk '{print $4}'
}

echo "== R1：DS dev 尺重掃（12 支、zeldajr）"
for S in 0 1 2; do
  CK=results/ckpt_medium-stitch_self_K8_c256_ch4_st8000_T128_ep2_gu_eorecon_ictr_tch0.5_norf_cd0.1_bci_s$S.pt
  for DS in 3 4 5 7.5; do
    sub zeldajr R1-s$S-d$DS - "OGBENCH_DATA_DIR=$ZDATA $BASE LACOT_ENV=$MENV LACOT_K=8 LACOT_TEACHER_MIX=0.5 LACOT_LOAD_CKPT=$CK $DEV $C2MA LACOT_DELTA_SUB=$DS $ZPY -u experiments/scratch_lacot_rollout.py"
  done
done

echo "== R2：EMA 嚴格配對 raw 臂（5 支、zeldajr）"
for S in 3 4 5 6 7; do
  EMCKPT=results/ckpt_large-stitch_self_K8_c256_ch4_st8000_T128_ep2_gu_eorecon_ictr_tch0.5_emw0.999_norf_cd0.1_bci_s$S.pt
  sub zeldajr R2-LE$S-raw - "OGBENCH_DATA_DIR=$ZDATA $BASE LACOT_ENV=$LENV LACOT_K=8 LACOT_TEACHER_MIX=0.5 LACOT_LOAD_CKPT=$EMCKPT $OFF $C2MA $ZPY -u experiments/scratch_lacot_rollout.py"
done

echo "== R3：bcown fork_rng 修後重訓驗證（moana 訓 → zeldajr 官方）"
J=$(sub moana R3-Mbco - "OGBENCH_DATA_DIR=$ADATA $TRAIN_BASE LACOT_ENV=$MENV LACOT_K=8 LACOT_SEED=1 LACOT_BC_OWN=1 $APY -u experiments/scratch_lacot_rollout.py")
sub zeldajr R3-Mbco-off $J "OGBENCH_DATA_DIR=$ZDATA $BASE LACOT_ENV=$MENV LACOT_K=8 LACOT_TEACHER_MIX=0.5 LACOT_BC_OWN=1 LACOT_LOAD_CKPT=results/ckpt_medium-stitch_self_K8_c256_ch4_st8000_T128_ep2_gu_eorecon_ictr_tch0.5_bcown_norf_cd0.1_bci_s1.pt $OFF $C2MA $ZPY -u experiments/scratch_lacot_rollout.py"
J=$(sub moana R3-Lbco - "OGBENCH_DATA_DIR=$ADATA $TRAIN_BASE LACOT_ENV=$LENV LACOT_K=8 LACOT_SEED=0 LACOT_BC_OWN=1 $APY -u experiments/scratch_lacot_rollout.py")
sub zeldajr R3-Lbco-off $J "OGBENCH_DATA_DIR=$ZDATA $BASE LACOT_ENV=$LENV LACOT_K=8 LACOT_TEACHER_MIX=0.5 LACOT_BC_OWN=1 LACOT_LOAD_CKPT=results/ckpt_large-stitch_self_K8_c256_ch4_st8000_T128_ep2_gu_eorecon_ictr_tch0.5_bcown_norf_cd0.1_bci_s0.pt $OFF $C2MA $ZPY -u experiments/scratch_lacot_rollout.py"

echo "== 全部排入（R1:12 R2:5 R3:4 ＝ 21 支）。"
