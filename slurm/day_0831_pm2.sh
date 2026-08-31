#!/bin/bash
# 下午第二波 2026-08-31（主人 13:27「繼續關注實驗，灑吧」）
# D2. EMA 判決格：large K8 s0/s1/s2 帶 EMA 重訓 → 每顆 raw/ema 兩臂官方（同顆配對、6 支）
# B2. 挑顆補完：s3/s4/s7 官方（dev 判準已被推翻 ⇒ 三顆補上 = large 8-seed 完整官方分布）
# E2a. 自舉多樣性 probe：拿 btit1 顆再生成一次、看 pass@M 對照 0.496（驗「分布自我集中」假說）
# E2b. 自舉 frac 0.3：同一份 it1 npz、蒸餾佔比 0.5→0.3 重訓 → dev＋官方（治「供點加值歸零」）
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

BTIT1=results/ckpt_medium-stitch_self_K8_c256_ch4_st8000_T128_ep2_gu_eorecon_ictr_tch0.5_btit1_norf_cd0.1_bci_s1.pt
BOOTNPZ=results/boot/boot_medium_it1.npz

sub() {  # sub <node> <name> <deps-or-> <env+cmd string>
  local node=$1 name=$2 deps=$3; shift 3
  local depflag=""
  [ "$deps" != "-" ] && depflag="--dependency=afterok:$deps"
  sbatch -p admin -A it -q great-mage --time=24:00:00 --nodelist=$node --gres=gpu:1 \
    --job-name=$name -o slurm/logs/%x-%j.out $depflag \
    --wrap "cd ~/Projects/lacot && env $*" | awk '{print $4}'
}

echo "== B2：挑顆補完 s3/s4/s7 官方（顆已存在、zeldajr 立即）"
for S in 3 4 7; do
  sub zeldajr Q-LK8s$S-off - "OGBENCH_DATA_DIR=$ZDATA $BASE LACOT_ENV=$LENV LACOT_K=8 LACOT_TEACHER_MIX=0.5 LACOT_LOAD_CKPT=results/ckpt_large-stitch_self_K8_c256_ch4_st8000_T128_ep2_gu_eorecon_ictr_tch0.5_norf_cd0.1_bci_s$S.pt $OFF $C2MA $ZPY -u experiments/scratch_lacot_rollout.py"
done

echo "== E2a：自舉多樣性 probe（btit1 顆再生成、對照 pass@8=0.496）"
sub zeldajr Q-bootprobe - "OGBENCH_DATA_DIR=$ZDATA $BASE LACOT_ENV=$MENV LACOT_K=8 LACOT_TEACHER_MIX=0.5 LACOT_BOOT_TAG=it1 LACOT_LOAD_CKPT=$BTIT1 LACOT_BOOT_GEN=results/boot/boot_medium_it2probe.npz LACOT_BOOT_Q=512 LACOT_BOOT_M=8 LACOT_BOOT_RMIN=8 LACOT_BOOT_RMAX=30 $ZPY -u experiments/scratch_lacot_rollout.py"

echo "== E2b：frac 0.3 重蒸（同 it1 npz、moana 訓 → dev ＋ 官方）"
J=$(sub moana Q-bootf3 - "OGBENCH_DATA_DIR=$ADATA $TRAIN_BASE LACOT_ENV=$MENV LACOT_K=8 LACOT_SEED=1 LACOT_BOOT_DATA=$BOOTNPZ LACOT_BOOT_TAG=it1f3 LACOT_BOOT_FRAC=0.3 $APY -u experiments/scratch_lacot_rollout.py")
BF3CKPT=results/ckpt_medium-stitch_self_K8_c256_ch4_st8000_T128_ep2_gu_eorecon_ictr_tch0.5_btit1f3_norf_cd0.1_bci_s1.pt
sub zeldajr Q-bootf3-dv $J "OGBENCH_DATA_DIR=$ZDATA $BASE LACOT_ENV=$MENV LACOT_K=8 LACOT_TEACHER_MIX=0.5 LACOT_BOOT_TAG=it1f3 LACOT_LOAD_CKPT=$BF3CKPT $DEV $C2MA $ZPY -u experiments/scratch_lacot_rollout.py"
sub zeldajr Q-bootf3-off $J "OGBENCH_DATA_DIR=$ZDATA $BASE LACOT_ENV=$MENV LACOT_K=8 LACOT_TEACHER_MIX=0.5 LACOT_BOOT_TAG=it1f3 LACOT_LOAD_CKPT=$BF3CKPT $OFF $C2MA $ZPY -u experiments/scratch_lacot_rollout.py"

echo "== D2：EMA 判決格（large K8 三 seed 帶 EMA 重訓、lady → 每顆 raw/ema 兩臂官方）"
for S in 0 1 2; do
  J=$(sub lady Q-LE$S - "OGBENCH_DATA_DIR=$ADATA $TRAIN_BASE LACOT_ENV=$LENV LACOT_K=8 LACOT_SEED=$S LACOT_EMA_W=0.999 $APY -u experiments/scratch_lacot_rollout.py")
  EMCKPT=results/ckpt_large-stitch_self_K8_c256_ch4_st8000_T128_ep2_gu_eorecon_ictr_tch0.5_emw0.999_norf_cd0.1_bci_s$S.pt
  sub zeldajr Q-LE$S-raw $J "OGBENCH_DATA_DIR=$ZDATA $BASE LACOT_ENV=$LENV LACOT_K=8 LACOT_TEACHER_MIX=0.5 LACOT_LOAD_CKPT=$EMCKPT $OFF $C2MA $ZPY -u experiments/scratch_lacot_rollout.py"
  sub zeldajr Q-LE$S-ema $J "OGBENCH_DATA_DIR=$ZDATA $BASE LACOT_ENV=$LENV LACOT_K=8 LACOT_TEACHER_MIX=0.5 LACOT_LOAD_EMA=1 LACOT_LOAD_CKPT=$EMCKPT $OFF $C2MA $ZPY -u experiments/scratch_lacot_rollout.py"
done

echo "== 全部排入（B2:3 E2a:1 E2b:3 D2:9 ＝ 16 支）。squeue 看隊。"
