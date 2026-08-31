#!/bin/bash
# 評估端掃參 2026-08-31（主人 13:52「好」）：兩顆王牌上擠分、全 eval-only 官方尺
# 軸：SUB_M 8/16（test-time scaling）、DELTA_SUB 5/10（供點幾何、ma 聯動）、ma1.5（好顆綁更緊）、SUB_CAP 15（遠題段數）
# tau 無 env 口（自校準、要改 code）⇒ 本波擱置。
set -euo pipefail
cd ~/Projects/lacot

ZPY=$HOME/venvs/lacot-rocm/bin/python
ZDATA=$HOME/data/ogbench
BASE="MUJOCO_GL=osmesa LACOT_ENC_OBJ=recon_ictr LACOT_LEARNED_REFINE=0 LACOT_COND_DROP=0.1 LACOT_BC_INDEP=1"
OFF="LACOT_DEV_EVAL=0 LACOT_EVAL_RS=0 LACOT_EVAL_EPISODES=50"
C2="LACOT_SUBGOAL=conf2 LACOT_SUB_POLICY=bc LACOT_GRAD_REFINE=1 LACOT_GRAD_R=0 LACOT_FINISH_R=2.0"
LENV=pointmaze-large-stitch-v0
MENV=pointmaze-medium-stitch-v0
LS5=results/ckpt_large-stitch_self_K8_c256_ch4_st8000_T128_ep2_gu_eorecon_ictr_tch0.5_norf_cd0.1_bci_s5.pt
MS1=results/ckpt_medium-stitch_self_K8_c256_ch4_st8000_T128_ep2_gu_eorecon_ictr_tch0.5_norf_cd0.1_bci_s1.pt

sub() {
  local name=$1; shift
  sbatch -p admin -A it -q great-mage --time=24:00:00 --nodelist=zeldajr --gres=gpu:1 \
    --job-name=$name -o slurm/logs/%x-%j.out \
    --wrap "cd ~/Projects/lacot && env $*" | awk '{print $4}'
}

echo "== SUB_M 8/16（每段候選數、test-time scaling）"
sub S-LM8  "OGBENCH_DATA_DIR=$ZDATA $BASE LACOT_ENV=$LENV LACOT_K=8 LACOT_TEACHER_MIX=0.5 LACOT_LOAD_CKPT=$LS5 $OFF $C2 LACOT_SUB_MAX_ARC=2 LACOT_SUB_M=8 $ZPY -u experiments/scratch_lacot_rollout.py"
sub S-MM8  "OGBENCH_DATA_DIR=$ZDATA $BASE LACOT_ENV=$MENV LACOT_K=8 LACOT_TEACHER_MIX=0.5 LACOT_LOAD_CKPT=$MS1 $OFF $C2 LACOT_SUB_MAX_ARC=2 LACOT_SUB_M=8 $ZPY -u experiments/scratch_lacot_rollout.py"
sub S-LM16 "OGBENCH_DATA_DIR=$ZDATA $BASE LACOT_ENV=$LENV LACOT_K=8 LACOT_TEACHER_MIX=0.5 LACOT_LOAD_CKPT=$LS5 $OFF $C2 LACOT_SUB_MAX_ARC=2 LACOT_SUB_M=16 $ZPY -u experiments/scratch_lacot_rollout.py"
sub S-MM16 "OGBENCH_DATA_DIR=$ZDATA $BASE LACOT_ENV=$MENV LACOT_K=8 LACOT_TEACHER_MIX=0.5 LACOT_LOAD_CKPT=$MS1 $OFF $C2 LACOT_SUB_MAX_ARC=2 LACOT_SUB_M=16 $ZPY -u experiments/scratch_lacot_rollout.py"

echo "== DELTA_SUB 5/10（供點基準距、ma2 聯動）"
sub S-LD5  "OGBENCH_DATA_DIR=$ZDATA $BASE LACOT_ENV=$LENV LACOT_K=8 LACOT_TEACHER_MIX=0.5 LACOT_LOAD_CKPT=$LS5 $OFF $C2 LACOT_SUB_MAX_ARC=2 LACOT_DELTA_SUB=5 $ZPY -u experiments/scratch_lacot_rollout.py"
sub S-MD5  "OGBENCH_DATA_DIR=$ZDATA $BASE LACOT_ENV=$MENV LACOT_K=8 LACOT_TEACHER_MIX=0.5 LACOT_LOAD_CKPT=$MS1 $OFF $C2 LACOT_SUB_MAX_ARC=2 LACOT_DELTA_SUB=5 $ZPY -u experiments/scratch_lacot_rollout.py"
sub S-LD10 "OGBENCH_DATA_DIR=$ZDATA $BASE LACOT_ENV=$LENV LACOT_K=8 LACOT_TEACHER_MIX=0.5 LACOT_LOAD_CKPT=$LS5 $OFF $C2 LACOT_SUB_MAX_ARC=2 LACOT_DELTA_SUB=10 $ZPY -u experiments/scratch_lacot_rollout.py"
sub S-MD10 "OGBENCH_DATA_DIR=$ZDATA $BASE LACOT_ENV=$MENV LACOT_K=8 LACOT_TEACHER_MIX=0.5 LACOT_LOAD_CKPT=$MS1 $OFF $C2 LACOT_SUB_MAX_ARC=2 LACOT_DELTA_SUB=10 $ZPY -u experiments/scratch_lacot_rollout.py"

echo "== ma 1.5（好顆綁更緊）＋ SUB_CAP 15（遠題段數、large）"
sub S-Lma15 "OGBENCH_DATA_DIR=$ZDATA $BASE LACOT_ENV=$LENV LACOT_K=8 LACOT_TEACHER_MIX=0.5 LACOT_LOAD_CKPT=$LS5 $OFF $C2 LACOT_SUB_MAX_ARC=1.5 $ZPY -u experiments/scratch_lacot_rollout.py"
sub S-Mma15 "OGBENCH_DATA_DIR=$ZDATA $BASE LACOT_ENV=$MENV LACOT_K=8 LACOT_TEACHER_MIX=0.5 LACOT_LOAD_CKPT=$MS1 $OFF $C2 LACOT_SUB_MAX_ARC=1.5 $ZPY -u experiments/scratch_lacot_rollout.py"
sub S-Lcap15 "OGBENCH_DATA_DIR=$ZDATA $BASE LACOT_ENV=$LENV LACOT_K=8 LACOT_TEACHER_MIX=0.5 LACOT_LOAD_CKPT=$LS5 $OFF $C2 LACOT_SUB_MAX_ARC=2 LACOT_SUB_CAP=15 $ZPY -u experiments/scratch_lacot_rollout.py"

echo "== 全部排入（11 支 eval-only）。"
