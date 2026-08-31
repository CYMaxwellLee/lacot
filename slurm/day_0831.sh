#!/bin/bash
# 白天排程 2026-08-31（主人上午核可）
# 三個問題：A medium 贏 QRL 站不站得穩（K4 跨 seed＋K8 官方）
#          B large 40 分 seed 方差怎麼治（12000 步×3 seed＋s0 好顆疊 ma3）
#          C K12 是真好還是好運（跨 seed 解顆品質混雜）
# 機制同 overnight_0830.sh：sbatch 全排、訓練→官方 eval 用 afterok 鏈。
set -euo pipefail
cd ~/Projects/lacot

ZPY=$HOME/venvs/lacot-rocm/bin/python
APY=/archive/cymaxwelllee/LaCoT/.venv/bin/python
ZDATA=$HOME/data/ogbench
ADATA=/archive/cymaxwelllee/data/ogbench

BASE="MUJOCO_GL=osmesa LACOT_ENC_OBJ=recon_ictr LACOT_LEARNED_REFINE=0 LACOT_COND_DROP=0.1 LACOT_BC_INDEP=1"
OFF="LACOT_DEV_EVAL=0 LACOT_EVAL_RS=0 LACOT_EVAL_EPISODES=50"
C2MA="LACOT_SUBGOAL=conf2 LACOT_SUB_POLICY=bc LACOT_GRAD_REFINE=1 LACOT_GRAD_R=0 LACOT_SUB_MAX_ARC=2 LACOT_FINISH_R=2.0"
C2MA3="LACOT_SUBGOAL=conf2 LACOT_SUB_POLICY=bc LACOT_GRAD_REFINE=1 LACOT_GRAD_R=0 LACOT_SUB_MAX_ARC=3 LACOT_FINISH_R=2.0"
LENV=pointmaze-large-stitch-v0
MENV=pointmaze-medium-stitch-v0
TRAIN_BASE="$BASE LACOT_STEPS2=8000 LACOT_TEACHER_MIX=0.5 LACOT_EVAL_RS=0 LACOT_EVAL_EPISODES=2"
TRAIN12="$BASE LACOT_STEPS2=12000 LACOT_TEACHER_MIX=0.5 LACOT_EVAL_RS=0 LACOT_EVAL_EPISODES=2"

MK8S0=results/ckpt_medium-stitch_self_K8_c256_ch4_st8000_T128_ep2_gu_eorecon_ictr_tch0.5_norf_cd0.1_bci_s0.pt
MK8S1=results/ckpt_medium-stitch_self_K8_c256_ch4_st8000_T128_ep2_gu_eorecon_ictr_tch0.5_norf_cd0.1_bci_s1.pt
LK8S0=results/ckpt_large-stitch_self_K8_c256_ch4_st8000_T128_ep2_gu_eorecon_ictr_tch0.5_norf_cd0.1_bci_s0.pt

sub() {  # sub <node> <name> <deps-or-> <env+cmd string>
  local node=$1 name=$2 deps=$3; shift 3
  local depflag=""
  [ "$deps" != "-" ] && depflag="--dependency=afterok:$deps"
  sbatch -p admin -A it -q great-mage --time=24:00:00 --nodelist=$node --gres=gpu:1 \
    --job-name=$name -o slurm/logs/%x-%j.out $depflag \
    --wrap "cd ~/Projects/lacot && env $*" | awk '{print $4}'
}

echo "== 批1：eval-only 立即（顆已存在、zeldajr）"
# A：medium K8 兩顆上官方段+ma2（dev 判決 0.700/0.900）
sub zeldajr D-MK8s0-off - "OGBENCH_DATA_DIR=$ZDATA $BASE LACOT_ENV=$MENV LACOT_K=8 LACOT_TEACHER_MIX=0.5 LACOT_LOAD_CKPT=$MK8S0 $OFF $C2MA $ZPY -u experiments/scratch_lacot_rollout.py"
sub zeldajr D-MK8s1-off - "OGBENCH_DATA_DIR=$ZDATA $BASE LACOT_ENV=$MENV LACOT_K=8 LACOT_TEACHER_MIX=0.5 LACOT_LOAD_CKPT=$MK8S1 $OFF $C2MA $ZPY -u experiments/scratch_lacot_rollout.py"
# B：large s0 好顆（0.796）疊 ma3（s1 線 ma3>ma2 的驗證）
sub zeldajr D-Ls0ma3 - "OGBENCH_DATA_DIR=$ZDATA $BASE LACOT_ENV=$LENV LACOT_K=8 LACOT_TEACHER_MIX=0.5 LACOT_LOAD_CKPT=$LK8S0 $OFF $C2MA3 $ZPY -u experiments/scratch_lacot_rollout.py"

echo "== 批2：A medium K4 跨 seed（moana 訓 → zeldajr 官方 ma2）"
J=$(sub moana D-Ms1 - "OGBENCH_DATA_DIR=$ADATA $TRAIN_BASE LACOT_ENV=$MENV LACOT_SEED=1 $APY -u experiments/scratch_lacot_rollout.py")
sub zeldajr D-Ms1-off $J "OGBENCH_DATA_DIR=$ZDATA $BASE LACOT_ENV=$MENV LACOT_TEACHER_MIX=0.5 LACOT_LOAD_CKPT=results/ckpt_medium-stitch_self_K4_c256_ch4_st8000_T128_ep2_gu_eorecon_ictr_tch0.5_norf_cd0.1_bci_s1.pt $OFF $C2MA $ZPY -u experiments/scratch_lacot_rollout.py"
J=$(sub moana D-Ms2 - "OGBENCH_DATA_DIR=$ADATA $TRAIN_BASE LACOT_ENV=$MENV LACOT_SEED=2 $APY -u experiments/scratch_lacot_rollout.py")
sub zeldajr D-Ms2-off $J "OGBENCH_DATA_DIR=$ZDATA $BASE LACOT_ENV=$MENV LACOT_TEACHER_MIX=0.5 LACOT_LOAD_CKPT=results/ckpt_medium-stitch_self_K4_c256_ch4_st8000_T128_ep2_gu_eorecon_ictr_tch0.5_norf_cd0.1_bci_s2.pt $OFF $C2MA $ZPY -u experiments/scratch_lacot_rollout.py"

echo "== 批3：B large 12000 步×3 seed（lady 訓 → zeldajr 官方 ma2）"
J=$(sub lady D-L12s0 - "OGBENCH_DATA_DIR=$ADATA $TRAIN12 LACOT_ENV=$LENV LACOT_K=8 LACOT_SEED=0 $APY -u experiments/scratch_lacot_rollout.py")
sub zeldajr D-L12s0-off $J "OGBENCH_DATA_DIR=$ZDATA $BASE LACOT_ENV=$LENV LACOT_K=8 LACOT_TEACHER_MIX=0.5 LACOT_LOAD_CKPT=results/ckpt_large-stitch_self_K8_c256_ch4_st12000_T128_ep2_gu_eorecon_ictr_tch0.5_norf_cd0.1_bci_s0.pt $OFF $C2MA $ZPY -u experiments/scratch_lacot_rollout.py"
J=$(sub lady D-L12s1 - "OGBENCH_DATA_DIR=$ADATA $TRAIN12 LACOT_ENV=$LENV LACOT_K=8 LACOT_SEED=1 $APY -u experiments/scratch_lacot_rollout.py")
sub zeldajr D-L12s1-off $J "OGBENCH_DATA_DIR=$ZDATA $BASE LACOT_ENV=$LENV LACOT_K=8 LACOT_TEACHER_MIX=0.5 LACOT_LOAD_CKPT=results/ckpt_large-stitch_self_K8_c256_ch4_st12000_T128_ep2_gu_eorecon_ictr_tch0.5_norf_cd0.1_bci_s1.pt $OFF $C2MA $ZPY -u experiments/scratch_lacot_rollout.py"
J=$(sub lady D-L12s2 - "OGBENCH_DATA_DIR=$ADATA $TRAIN12 LACOT_ENV=$LENV LACOT_K=8 LACOT_SEED=2 $APY -u experiments/scratch_lacot_rollout.py")
sub zeldajr D-L12s2-off $J "OGBENCH_DATA_DIR=$ZDATA $BASE LACOT_ENV=$LENV LACOT_K=8 LACOT_TEACHER_MIX=0.5 LACOT_LOAD_CKPT=results/ckpt_large-stitch_self_K8_c256_ch4_st12000_T128_ep2_gu_eorecon_ictr_tch0.5_norf_cd0.1_bci_s2.pt $OFF $C2MA $ZPY -u experiments/scratch_lacot_rollout.py"

echo "== 批4：C K12 跨 seed（lady 訓 → zeldajr 官方 ma2）"
J=$(sub lady D-K12s0 - "OGBENCH_DATA_DIR=$ADATA $TRAIN_BASE LACOT_ENV=$LENV LACOT_K=12 LACOT_SEED=0 $APY -u experiments/scratch_lacot_rollout.py")
sub zeldajr D-K12s0-off $J "OGBENCH_DATA_DIR=$ZDATA $BASE LACOT_ENV=$LENV LACOT_K=12 LACOT_TEACHER_MIX=0.5 LACOT_LOAD_CKPT=results/ckpt_large-stitch_self_K12_c256_ch4_st8000_T128_ep2_gu_eorecon_ictr_tch0.5_norf_cd0.1_bci_s0.pt $OFF $C2MA $ZPY -u experiments/scratch_lacot_rollout.py"
J=$(sub lady D-K12s2 - "OGBENCH_DATA_DIR=$ADATA $TRAIN_BASE LACOT_ENV=$LENV LACOT_K=12 LACOT_SEED=2 $APY -u experiments/scratch_lacot_rollout.py")
sub zeldajr D-K12s2-off $J "OGBENCH_DATA_DIR=$ZDATA $BASE LACOT_ENV=$LENV LACOT_K=12 LACOT_TEACHER_MIX=0.5 LACOT_LOAD_CKPT=results/ckpt_large-stitch_self_K12_c256_ch4_st8000_T128_ep2_gu_eorecon_ictr_tch0.5_norf_cd0.1_bci_s2.pt $OFF $C2MA $ZPY -u experiments/scratch_lacot_rollout.py"

echo "== 全部排入（17 支）。squeue 看隊。"
