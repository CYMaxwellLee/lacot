#!/bin/bash
# 過夜排程 2026-08-31（主人 18:00「可以灑下去」；e giant 因 Berkeley 站掛、另掛背景重試）
# a. medium DS4 跨 seed 站穩（s0/s2 顆）＋DS3 拐點
# b. large EMA 農場：s3~s7 帶 EMA 重訓 → ema 臂官方（8-seed EMA 全景的後五格）
# c. 真獨立 GCBC 正式跑：medium s1＋large s0（BC_OWN、官方含 bc 臂）
# d. 自舉 v2 最便宜的藥：btit1 顆 eval 端 ma 收緊（1.5／1）補償信心膨脹
set -euo pipefail
cd ~/Projects/lacot

ZPY=$HOME/venvs/lacot-rocm/bin/python
APY=/archive/cymaxwelllee/LaCoT/.venv/bin/python
ZDATA=$HOME/data/ogbench
ADATA=/archive/cymaxwelllee/data/ogbench
BASE="MUJOCO_GL=osmesa LACOT_ENC_OBJ=recon_ictr LACOT_LEARNED_REFINE=0 LACOT_COND_DROP=0.1 LACOT_BC_INDEP=1"
OFF="LACOT_DEV_EVAL=0 LACOT_EVAL_RS=0 LACOT_EVAL_EPISODES=50"
C2MA="LACOT_SUBGOAL=conf2 LACOT_SUB_POLICY=bc LACOT_GRAD_REFINE=1 LACOT_GRAD_R=0 LACOT_SUB_MAX_ARC=2 LACOT_FINISH_R=2.0"
LENV=pointmaze-large-stitch-v0
MENV=pointmaze-medium-stitch-v0
TRAIN_BASE="$BASE LACOT_STEPS2=8000 LACOT_TEACHER_MIX=0.5 LACOT_EVAL_RS=0 LACOT_EVAL_EPISODES=2"
MK8S0=results/ckpt_medium-stitch_self_K8_c256_ch4_st8000_T128_ep2_gu_eorecon_ictr_tch0.5_norf_cd0.1_bci_s0.pt
MK8S1=results/ckpt_medium-stitch_self_K8_c256_ch4_st8000_T128_ep2_gu_eorecon_ictr_tch0.5_norf_cd0.1_bci_s1.pt
MK8S2=results/ckpt_medium-stitch_self_K8_c256_ch4_st8000_T128_ep2_gu_eorecon_ictr_tch0.5_norf_cd0.1_bci_s2.pt
BTIT1=results/ckpt_medium-stitch_self_K8_c256_ch4_st8000_T128_ep2_gu_eorecon_ictr_tch0.5_btit1_norf_cd0.1_bci_s1.pt

sub() {  # sub <node> <name> <deps-or-> <env+cmd string>
  local node=$1 name=$2 deps=$3; shift 3
  local depflag=""
  [ "$deps" != "-" ] && depflag="--dependency=afterok:$deps"
  sbatch -p admin -A it -q great-mage --time=24:00:00 --nodelist=$node --gres=gpu:1 \
    --job-name=$name -o slurm/logs/%x-%j.out $depflag \
    --wrap "cd ~/Projects/lacot && env $*" | awk '{print $4}'
}

echo "== a：medium DS4 跨 seed（s0/s2）＋DS3 拐點（zeldajr 立即）"
sub zeldajr N3-MD4s0 - "OGBENCH_DATA_DIR=$ZDATA $BASE LACOT_ENV=$MENV LACOT_K=8 LACOT_TEACHER_MIX=0.5 LACOT_LOAD_CKPT=$MK8S0 $OFF $C2MA LACOT_DELTA_SUB=4 $ZPY -u experiments/scratch_lacot_rollout.py"
sub zeldajr N3-MD4s2 - "OGBENCH_DATA_DIR=$ZDATA $BASE LACOT_ENV=$MENV LACOT_K=8 LACOT_TEACHER_MIX=0.5 LACOT_LOAD_CKPT=$MK8S2 $OFF $C2MA LACOT_DELTA_SUB=4 $ZPY -u experiments/scratch_lacot_rollout.py"
sub zeldajr N3-MD3 - "OGBENCH_DATA_DIR=$ZDATA $BASE LACOT_ENV=$MENV LACOT_K=8 LACOT_TEACHER_MIX=0.5 LACOT_LOAD_CKPT=$MK8S1 $OFF $C2MA LACOT_DELTA_SUB=3 $ZPY -u experiments/scratch_lacot_rollout.py"

echo "== d：自舉 v2 藥（btit1 顆 ma 收緊、zeldajr 立即）"
sub zeldajr N3-bt-ma15 - "OGBENCH_DATA_DIR=$ZDATA $BASE LACOT_ENV=$MENV LACOT_K=8 LACOT_TEACHER_MIX=0.5 LACOT_BOOT_TAG=it1 LACOT_LOAD_CKPT=$BTIT1 $OFF LACOT_SUBGOAL=conf2 LACOT_SUB_POLICY=bc LACOT_GRAD_REFINE=1 LACOT_GRAD_R=0 LACOT_SUB_MAX_ARC=1.5 LACOT_FINISH_R=2.0 $ZPY -u experiments/scratch_lacot_rollout.py"
sub zeldajr N3-bt-ma1 - "OGBENCH_DATA_DIR=$ZDATA $BASE LACOT_ENV=$MENV LACOT_K=8 LACOT_TEACHER_MIX=0.5 LACOT_BOOT_TAG=it1 LACOT_LOAD_CKPT=$BTIT1 $OFF LACOT_SUBGOAL=conf2 LACOT_SUB_POLICY=bc LACOT_GRAD_REFINE=1 LACOT_GRAD_R=0 LACOT_SUB_MAX_ARC=1 LACOT_FINISH_R=2.0 $ZPY -u experiments/scratch_lacot_rollout.py"

echo "== b：large EMA 農場 s3~s7（lady 訓 → zeldajr ema 臂官方）"
for S in 3 4 5 6 7; do
  J=$(sub lady N3-LE$S - "OGBENCH_DATA_DIR=$ADATA $TRAIN_BASE LACOT_ENV=$LENV LACOT_K=8 LACOT_SEED=$S LACOT_EMA_W=0.999 $APY -u experiments/scratch_lacot_rollout.py")
  EMCKPT=results/ckpt_large-stitch_self_K8_c256_ch4_st8000_T128_ep2_gu_eorecon_ictr_tch0.5_emw0.999_norf_cd0.1_bci_s$S.pt
  sub zeldajr N3-LE$S-ema $J "OGBENCH_DATA_DIR=$ZDATA $BASE LACOT_ENV=$LENV LACOT_K=8 LACOT_TEACHER_MIX=0.5 LACOT_LOAD_EMA=1 LACOT_LOAD_CKPT=$EMCKPT $OFF $C2MA $ZPY -u experiments/scratch_lacot_rollout.py"
done

echo "== c：真獨立 GCBC（moana 訓 → zeldajr 官方含 bc 臂）"
J=$(sub moana N3-Mbco - "OGBENCH_DATA_DIR=$ADATA $TRAIN_BASE LACOT_ENV=$MENV LACOT_K=8 LACOT_SEED=1 LACOT_BC_OWN=1 $APY -u experiments/scratch_lacot_rollout.py")
sub zeldajr N3-Mbco-off $J "OGBENCH_DATA_DIR=$ZDATA $BASE LACOT_ENV=$MENV LACOT_K=8 LACOT_TEACHER_MIX=0.5 LACOT_BC_OWN=1 LACOT_LOAD_CKPT=results/ckpt_medium-stitch_self_K8_c256_ch4_st8000_T128_ep2_gu_eorecon_ictr_tch0.5_bcown_norf_cd0.1_bci_s1.pt $OFF $C2MA $ZPY -u experiments/scratch_lacot_rollout.py"
J=$(sub moana N3-Lbco - "OGBENCH_DATA_DIR=$ADATA $TRAIN_BASE LACOT_ENV=$LENV LACOT_K=8 LACOT_SEED=0 LACOT_BC_OWN=1 $APY -u experiments/scratch_lacot_rollout.py")
sub zeldajr N3-Lbco-off $J "OGBENCH_DATA_DIR=$ZDATA $BASE LACOT_ENV=$LENV LACOT_K=8 LACOT_TEACHER_MIX=0.5 LACOT_BC_OWN=1 LACOT_LOAD_CKPT=results/ckpt_large-stitch_self_K8_c256_ch4_st8000_T128_ep2_gu_eorecon_ictr_tch0.5_bcown_norf_cd0.1_bci_s0.pt $OFF $C2MA $ZPY -u experiments/scratch_lacot_rollout.py"

echo "== 全部排入（a:3 d:2 b:10 c:4 ＝ 19 支）。e giant 等資料源復活另發。"
