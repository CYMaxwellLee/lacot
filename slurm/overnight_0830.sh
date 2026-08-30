#!/bin/bash
# 過夜排程 2026-08-30（主人：「可以排程跑到明天中午或下午」）
# 目標：①歸因交互格補齊 ②K8+tch 跨 seed 坐實 ③mix 掃＋medium K8 ④官方協定第一批正式數字
# 機制：sbatch 一次全排 — 同節點超額自動排隊；訓練→判決 eval 用 afterok 鏈。
set -euo pipefail
cd ~/Projects/lacot

ZPY=$HOME/venvs/lacot-rocm/bin/python
APY=/archive/cymaxwelllee/LaCoT/.venv/bin/python
ZDATA=$HOME/data/ogbench
ADATA=/archive/cymaxwelllee/data/ogbench

BASE="MUJOCO_GL=osmesa LACOT_ENC_OBJ=recon_ictr LACOT_LEARNED_REFINE=0 LACOT_COND_DROP=0.1 LACOT_BC_INDEP=1"
DEV="LACOT_DEV_EVAL=1 LACOT_DEV_TIERS=2 LACOT_DEV_PER_TIER=100 LACOT_EVAL_RS=0 LACOT_EVAL_EPISODES=2"
C2MA="LACOT_SUBGOAL=conf2 LACOT_SUB_POLICY=bc LACOT_GRAD_REFINE=1 LACOT_GRAD_R=0 LACOT_SUB_MAX_ARC=2 LACOT_FINISH_R=2.0"
LENV=pointmaze-large-stitch-v0
MENV=pointmaze-medium-stitch-v0

L2CKPT=results/ckpt_large-stitch_self_K4_c256_ch4_st8000_T128_ep2_gu_eorecon_ictr_norf_cd0.1_bci_s1.pt
LT8CKPT=results/ckpt_large-stitch_self_K8_c256_ch4_st8000_T128_ep2_gu_eorecon_ictr_tch0.5_norf_cd0.1_bci_s1.pt
MTCKPT=results/ckpt_medium-stitch_self_K4_c256_ch4_st8000_T128_ep2_gu_eorecon_ictr_tch0.5_norf_cd0.1_bci_s0.pt
A2CKPT=results/ckpt_medium-stitch_self_K4_c256_ch4_st8000_T128_ep50_gu_eorecon_ictr_norf_cd0.1_bci_s0.pt

sub() {  # sub <node> <name> <deps-or-> <env+cmd string>
  local node=$1 name=$2 deps=$3; shift 3
  local depflag=""
  [ "$deps" != "-" ] && depflag="--dependency=afterok:$deps"
  sbatch -p admin -A it -q great-mage --time=24:00:00 --nodelist=$node --gres=gpu:1 \
    --job-name=$name -o slurm/logs/%x-%j.out $depflag \
    --wrap "cd ~/Projects/lacot && env $*" | awk '{print $4}'
}

echo "== 批1：歸因交互格（eval-only、zeldajr 立即）"
# N-ma-only：上限【單獨】在無 teacher 顆 ⇒ 拆 teacher×上限交互
sub zeldajr N-ma-only - "OGBENCH_DATA_DIR=$ZDATA $BASE LACOT_ENV=$LENV LACOT_LOAD_CKPT=$L2CKPT $DEV $C2MA $ZPY -u experiments/scratch_lacot_rollout.py"
# N-k8-noma：K8 顆【無上限】 ⇒ 拆 K×上限交互
sub zeldajr N-k8-noma - "OGBENCH_DATA_DIR=$ZDATA $BASE LACOT_ENV=$LENV LACOT_K=8 LACOT_TEACHER_MIX=0.5 LACOT_LOAD_CKPT=$LT8CKPT $DEV LACOT_SUBGOAL=conf2 LACOT_SUB_POLICY=bc LACOT_GRAD_REFINE=1 LACOT_GRAD_R=0 LACOT_FINISH_R=2.0 $ZPY -u experiments/scratch_lacot_rollout.py"
# N-mt-ebfs / N-mt-orc：M-tch 顆的圖搜索/oracle 格 ⇒ teacher 顆供點矩陣完整
sub zeldajr N-mt-ebfs - "OGBENCH_DATA_DIR=$ZDATA $BASE LACOT_ENV=$MENV LACOT_TEACHER_MIX=0.5 LACOT_LOAD_CKPT=$MTCKPT $DEV LACOT_SUBGOAL=ebfs LACOT_SUB_POLICY=bc LACOT_GRAD_REFINE=1 LACOT_FINISH_R=2.0 $ZPY -u experiments/scratch_lacot_rollout.py"
sub zeldajr N-mt-orc - "OGBENCH_DATA_DIR=$ZDATA $BASE LACOT_ENV=$MENV LACOT_TEACHER_MIX=0.5 LACOT_LOAD_CKPT=$MTCKPT $DEV LACOT_SUBGOAL=bfs LACOT_SUB_POLICY=bc LACOT_GRAD_REFINE=1 LACOT_FINISH_R=2.0 $ZPY -u experiments/scratch_lacot_rollout.py"

echo "== 批2：訓練（lady×2＋moana×1 輪班）＋批3：各自判決 eval（afterok、zeldajr）"
TRAIN_BASE="$BASE LACOT_STEPS2=8000 LACOT_TEACHER_MIX=0.5 LACOT_EVAL_RS=0 LACOT_EVAL_EPISODES=2"
# K8 跨 seed（s0/s2；s1 已有 0.720）
J=$(sub lady N-K8s0 - "OGBENCH_DATA_DIR=$ADATA $TRAIN_BASE LACOT_ENV=$LENV LACOT_K=8 LACOT_SEED=0 $APY -u experiments/scratch_lacot_rollout.py")
sub zeldajr N-K8s0-ev $J "OGBENCH_DATA_DIR=$ZDATA $BASE LACOT_ENV=$LENV LACOT_K=8 LACOT_TEACHER_MIX=0.5 LACOT_LOAD_CKPT=results/ckpt_large-stitch_self_K8_c256_ch4_st8000_T128_ep2_gu_eorecon_ictr_tch0.5_norf_cd0.1_bci_s0.pt $DEV $C2MA $ZPY -u experiments/scratch_lacot_rollout.py"
J=$(sub lady N-K8s2 - "OGBENCH_DATA_DIR=$ADATA $TRAIN_BASE LACOT_ENV=$LENV LACOT_K=8 LACOT_SEED=2 $APY -u experiments/scratch_lacot_rollout.py")
sub zeldajr N-K8s2-ev $J "OGBENCH_DATA_DIR=$ZDATA $BASE LACOT_ENV=$LENV LACOT_K=8 LACOT_TEACHER_MIX=0.5 LACOT_LOAD_CKPT=results/ckpt_large-stitch_self_K8_c256_ch4_st8000_T128_ep2_gu_eorecon_ictr_tch0.5_norf_cd0.1_bci_s2.pt $DEV $C2MA $ZPY -u experiments/scratch_lacot_rollout.py"
# medium K8（K 甜蜜點的地圖相關性）
J=$(sub moana N-MK8 - "OGBENCH_DATA_DIR=$ADATA $TRAIN_BASE LACOT_ENV=$MENV LACOT_K=8 LACOT_SEED=0 $APY -u experiments/scratch_lacot_rollout.py")
sub zeldajr N-MK8-ev $J "OGBENCH_DATA_DIR=$ZDATA $BASE LACOT_ENV=$MENV LACOT_K=8 LACOT_TEACHER_MIX=0.5 LACOT_LOAD_CKPT=results/ckpt_medium-stitch_self_K8_c256_ch4_st8000_T128_ep2_gu_eorecon_ictr_tch0.5_norf_cd0.1_bci_s0.pt $DEV $C2MA $ZPY -u experiments/scratch_lacot_rollout.py"
# mix 掃（large K8、0.3／0.7）
J=$(sub lady N-mx3 - "OGBENCH_DATA_DIR=$ADATA $BASE LACOT_STEPS2=8000 LACOT_TEACHER_MIX=0.3 LACOT_EVAL_RS=0 LACOT_EVAL_EPISODES=2 LACOT_ENV=$LENV LACOT_K=8 LACOT_SEED=1 $APY -u experiments/scratch_lacot_rollout.py")
sub zeldajr N-mx3-ev $J "OGBENCH_DATA_DIR=$ZDATA $BASE LACOT_ENV=$LENV LACOT_K=8 LACOT_TEACHER_MIX=0.3 LACOT_LOAD_CKPT=results/ckpt_large-stitch_self_K8_c256_ch4_st8000_T128_ep2_gu_eorecon_ictr_tch0.3_norf_cd0.1_bci_s1.pt $DEV $C2MA $ZPY -u experiments/scratch_lacot_rollout.py"
J=$(sub lady N-mx7 - "OGBENCH_DATA_DIR=$ADATA $BASE LACOT_STEPS2=8000 LACOT_TEACHER_MIX=0.7 LACOT_EVAL_RS=0 LACOT_EVAL_EPISODES=2 LACOT_ENV=$LENV LACOT_K=8 LACOT_SEED=1 $APY -u experiments/scratch_lacot_rollout.py")
sub zeldajr N-mx7-ev $J "OGBENCH_DATA_DIR=$ZDATA $BASE LACOT_ENV=$LENV LACOT_K=8 LACOT_TEACHER_MIX=0.7 LACOT_LOAD_CKPT=results/ckpt_large-stitch_self_K8_c256_ch4_st8000_T128_ep2_gu_eorecon_ictr_tch0.7_norf_cd0.1_bci_s1.pt $DEV $C2MA $ZPY -u experiments/scratch_lacot_rollout.py"

echo "== 批4：官方協定正式數字（長跑、整夜）"
OFF="LACOT_DEV_EVAL=0 LACOT_EVAL_RS=0 LACOT_EVAL_EPISODES=50"
# 主打配置：L-tchK8＋conf2+ma2+bc（官方段含新加的分段臂）
sub zeldajr N-off-L - "OGBENCH_DATA_DIR=$ZDATA $BASE LACOT_ENV=$LENV LACOT_K=8 LACOT_TEACHER_MIX=0.5 LACOT_LOAD_CKPT=$LT8CKPT $OFF $C2MA $ZPY -u experiments/scratch_lacot_rollout.py"
# medium 主打：M-tch＋conf2（無 ma — medium 0.930 格的官方版）
sub zeldajr N-off-M - "OGBENCH_DATA_DIR=$ZDATA $BASE LACOT_ENV=$MENV LACOT_TEACHER_MIX=0.5 LACOT_LOAD_CKPT=$MTCKPT $OFF LACOT_SUBGOAL=conf2 LACOT_SUB_POLICY=bc LACOT_GRAD_REFINE=1 LACOT_GRAD_R=0 LACOT_FINISH_R=2.0 $ZPY -u experiments/scratch_lacot_rollout.py"
# teacher 上界的官方版（ebfs+bc、兩張圖）
sub zeldajr N-off-Le - "OGBENCH_DATA_DIR=$ZDATA $BASE LACOT_ENV=$LENV LACOT_LOAD_CKPT=$L2CKPT $OFF LACOT_SUBGOAL=ebfs LACOT_SUB_POLICY=bc LACOT_GRAD_REFINE=1 LACOT_FINISH_R=2.0 $ZPY -u experiments/scratch_lacot_rollout.py"
sub zeldajr N-off-Me - "OGBENCH_DATA_DIR=$ZDATA $BASE LACOT_ENV=$MENV LACOT_LOAD_CKPT=$A2CKPT $OFF LACOT_SUBGOAL=ebfs LACOT_SUB_POLICY=bc LACOT_GRAD_REFINE=1 LACOT_FINISH_R=2.0 $ZPY -u experiments/scratch_lacot_rollout.py"

echo "== 全部排入。squeue 看隊。"