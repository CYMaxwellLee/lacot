#!/bin/bash
# 下午排程 2026-08-31（主人 11:49「灑下去吧」）
# A. medium K8 補 s2（上表紅線 3+ seeds）
# B. large K8 挑顆：s3~s7 五顆新 seed → dev 判決（top2 官方複證等 dev 開獎後手動發）
# C. chunk=1 主打還債：medium K8 s1 + large K8 s0 的 chunk1 版 → 官方
# E. 自舉第一輪（medium）：K8 s1 好顆生成（半徑推 30）→ exp(−βE) 蒸餾 → dev 判決
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

MK8S1=results/ckpt_medium-stitch_self_K8_c256_ch4_st8000_T128_ep2_gu_eorecon_ictr_tch0.5_norf_cd0.1_bci_s1.pt
BOOTNPZ=results/boot/boot_medium_it1.npz
mkdir -p results/boot

sub() {  # sub <node> <name> <deps-or-> <env+cmd string>
  local node=$1 name=$2 deps=$3; shift 3
  local depflag=""
  [ "$deps" != "-" ] && depflag="--dependency=afterok:$deps"
  sbatch -p admin -A it -q great-mage --time=24:00:00 --nodelist=$node --gres=gpu:1 \
    --job-name=$name -o slurm/logs/%x-%j.out $depflag \
    --wrap "cd ~/Projects/lacot && env $*" | awk '{print $4}'
}

echo "== A：medium K8 s2（moana 訓 → zeldajr 官方 ma2）"
J=$(sub moana P-MK8s2 - "OGBENCH_DATA_DIR=$ADATA $TRAIN_BASE LACOT_ENV=$MENV LACOT_K=8 LACOT_SEED=2 $APY -u experiments/scratch_lacot_rollout.py")
sub zeldajr P-MK8s2-off $J "OGBENCH_DATA_DIR=$ZDATA $BASE LACOT_ENV=$MENV LACOT_K=8 LACOT_TEACHER_MIX=0.5 LACOT_LOAD_CKPT=results/ckpt_medium-stitch_self_K8_c256_ch4_st8000_T128_ep2_gu_eorecon_ictr_tch0.5_norf_cd0.1_bci_s2.pt $OFF $C2MA $ZPY -u experiments/scratch_lacot_rollout.py"

echo "== B：large K8 挑顆 s3~s7（lady 訓 → zeldajr dev 判決；top2 官方複證等開獎手動發）"
for S in 3 4 5 6 7; do
  J=$(sub lady P-LK8s$S - "OGBENCH_DATA_DIR=$ADATA $TRAIN_BASE LACOT_ENV=$LENV LACOT_K=8 LACOT_SEED=$S $APY -u experiments/scratch_lacot_rollout.py")
  sub zeldajr P-LK8s$S-dv $J "OGBENCH_DATA_DIR=$ZDATA $BASE LACOT_ENV=$LENV LACOT_K=8 LACOT_TEACHER_MIX=0.5 LACOT_LOAD_CKPT=results/ckpt_large-stitch_self_K8_c256_ch4_st8000_T128_ep2_gu_eorecon_ictr_tch0.5_norf_cd0.1_bci_s$S.pt $DEV $C2MA $ZPY -u experiments/scratch_lacot_rollout.py"
done

echo "== C：chunk=1 還債（medium K8 s1＋large K8 s0；⚠️ eval 支必帶 LACOT_CHUNK=1 過 cfg 檢查）"
J=$(sub moana P-MC1 - "OGBENCH_DATA_DIR=$ADATA $TRAIN_BASE LACOT_ENV=$MENV LACOT_K=8 LACOT_SEED=1 LACOT_CHUNK=1 $APY -u experiments/scratch_lacot_rollout.py")
sub zeldajr P-MC1-off $J "OGBENCH_DATA_DIR=$ZDATA $BASE LACOT_ENV=$MENV LACOT_K=8 LACOT_CHUNK=1 LACOT_TEACHER_MIX=0.5 LACOT_LOAD_CKPT=results/ckpt_medium-stitch_self_K8_c256_ch1_st8000_T128_ep2_gu_eorecon_ictr_tch0.5_norf_cd0.1_bci_s1.pt $OFF $C2MA $ZPY -u experiments/scratch_lacot_rollout.py"
J=$(sub lady P-LC1 - "OGBENCH_DATA_DIR=$ADATA $TRAIN_BASE LACOT_ENV=$LENV LACOT_K=8 LACOT_SEED=0 LACOT_CHUNK=1 $APY -u experiments/scratch_lacot_rollout.py")
sub zeldajr P-LC1-off $J "OGBENCH_DATA_DIR=$ZDATA $BASE LACOT_ENV=$LENV LACOT_K=8 LACOT_CHUNK=1 LACOT_TEACHER_MIX=0.5 LACOT_LOAD_CKPT=results/ckpt_large-stitch_self_K8_c256_ch1_st8000_T128_ep2_gu_eorecon_ictr_tch0.5_norf_cd0.1_bci_s0.pt $OFF $C2MA $ZPY -u experiments/scratch_lacot_rollout.py"

echo "== E：自舉第一輪（zeldajr 生成 → moana 蒸餾 → zeldajr dev 判決）"
J=$(sub zeldajr P-bootgen - "OGBENCH_DATA_DIR=$ZDATA $BASE LACOT_ENV=$MENV LACOT_K=8 LACOT_LOAD_CKPT=$MK8S1 LACOT_BOOT_GEN=$BOOTNPZ LACOT_BOOT_Q=512 LACOT_BOOT_M=8 LACOT_BOOT_RMIN=8 LACOT_BOOT_RMAX=30 $ZPY -u experiments/scratch_lacot_rollout.py")
J2=$(sub moana P-bootit1 $J "OGBENCH_DATA_DIR=$ADATA $TRAIN_BASE LACOT_ENV=$MENV LACOT_K=8 LACOT_SEED=1 LACOT_BOOT_DATA=$BOOTNPZ LACOT_BOOT_TAG=it1 LACOT_BOOT_FRAC=0.5 $APY -u experiments/scratch_lacot_rollout.py")
sub zeldajr P-bootit1-dv $J2 "OGBENCH_DATA_DIR=$ZDATA $BASE LACOT_ENV=$MENV LACOT_K=8 LACOT_TEACHER_MIX=0.5 LACOT_BOOT_TAG=it1 LACOT_LOAD_CKPT=results/ckpt_medium-stitch_self_K8_c256_ch4_st8000_T128_ep2_gu_eorecon_ictr_tch0.5_btit1_norf_cd0.1_bci_s1.pt $DEV $C2MA $ZPY -u experiments/scratch_lacot_rollout.py"

echo "== 全部排入（A:2 B:10 C:4 E:3 ＝ 19 支）。squeue 看隊。"
