#!/bin/bash
# 2026-09-02 22:3x：hard v4 判負（eval .20/.22/.00）後的替代臂（目的不變、臂換成 soft 或原配方）
#   C'  soft ckpt × 守門 HG（綁定之後守門還開不開火）                8 eval
#   E'  新 held-out s28~35：soft 綁定（原配方臂 EB 已在跑）             8 train + 8 eval
#   F'  medium-stitch：原配方（V8 recipe）八顆 ⇒ medium 官方 8-seed     8 train + 8 eval
set -euo pipefail
cd ~/Projects/lacot
ZPY=$HOME/venvs/lacot-rocm/bin/python; APY=/archive/cymaxwelllee/LaCoT/.venv/bin/python
ZDATA=$HOME/data/ogbench; ADATA=/archive/cymaxwelllee/data/ogbench
BASE="MUJOCO_GL=osmesa LACOT_ENC_OBJ=recon_ictr LACOT_LEARNED_REFINE=0 LACOT_COND_DROP=0.1 LACOT_BC_INDEP=1"
TRAIN_BASE="$BASE LACOT_STEPS2=8000 LACOT_TEACHER_MIX=0.5 LACOT_EVAL_RS=0 LACOT_EVAL_EPISODES=2 LACOT_DIAG_TRAIN=1 LACOT_EMA_W=0.999 LACOT_WARMUP=500"
OFF="LACOT_DEV_EVAL=0 LACOT_EVAL_RS=0 LACOT_EVAL_EPISODES=50"
C2MA="LACOT_SUBGOAL=conf2 LACOT_SUB_POLICY=bc LACOT_GRAD_REFINE=1 LACOT_GRAD_R=0 LACOT_SUB_MAX_ARC=2 LACOT_FINISH_R=2.0"
LENV=pointmaze-large-stitch-v0; MENV=pointmaze-medium-stitch-v0
PREL=ckpt_large-stitch_self_K8_c256_ch4_st8000_T128_ep2_gu; PREM=ckpt_medium-stitch_self_K8_c256_ch4_st8000_T128_ep2_gu
mkdir -p slurm/logs
sub() { local node=$1 name=$2 deps=$3; shift 3; local depflag=""
  [ "$deps" != "-" ] && depflag="--dependency=afterok:$deps"
  sbatch -p admin -A it -q great-mage --time=24:00:00 --nodelist=$node --gres=gpu:1 --job-name=$name -o slurm/logs/%x-%j.out $depflag \
    --wrap "cd ~/Projects/lacot && env $*" | awk '{print $4}'; }
NODES=(lady moana pocahontas); i=0
pick() { local n=${NODES[$((i%3))]}; i=$((i+1)); echo $n; }
evnode() { if [ $(($1 % 2)) -eq 0 ]; then echo "zeldajr $ZPY $ZDATA"; else echo "$2 $APY $ADATA"; fi; }
echo "== C'：soft ckpt × HG"
OUTD=results/night_0902/dssoft_HG; mkdir -p "$OUTD"
for S in 20 21 22 23 24 25 26 27; do
  CK=results/${PREL}_eorecon_ictr_tch0.5_emw0.999_wu500_dssoft_norf_cd0.1_bci_s$S.pt; N=$(pick); read EN EP ED <<< "$(evnode $S $N)"
  sub $EN CSHG-s$S - "OGBENCH_DATA_DIR=$ED $BASE LACOT_ENV=$LENV LACOT_K=8 LACOT_TEACHER_MIX=0.5 LACOT_LOAD_EMA=1 LACOT_LOAD_CKPT=$CK LACOT_DEC_START=soft LACOT_SUB_HEADGUARD=3.0 LACOT_DIAG_DUMP=1 LACOT_OUT_DIR=$OUTD $OFF $C2MA $EP -u experiments/scratch_lacot_rollout.py" >/dev/null; echo "CSHG-s$S -> $EN"
done
echo "== E'：s28~35 soft 綁定"
OUTD=results/night_0902/heldout2_soft; mkdir -p "$OUTD"
for S in 28 29 30 31 32 33 34 35; do
  N=$(pick); J=$(sub $N ES-s$S - "OGBENCH_DATA_DIR=$ADATA $TRAIN_BASE LACOT_ENV=$LENV LACOT_K=8 LACOT_SEED=$S LACOT_DEC_START=soft $APY -u experiments/scratch_lacot_rollout.py")
  CK=results/${PREL}_eorecon_ictr_tch0.5_emw0.999_wu500_dssoft_norf_cd0.1_bci_s$S.pt; read EN EP ED <<< "$(evnode $S $N)"
  sub $EN ES-s$S-ema $J "OGBENCH_DATA_DIR=$ED $BASE LACOT_ENV=$LENV LACOT_K=8 LACOT_TEACHER_MIX=0.5 LACOT_LOAD_EMA=1 LACOT_LOAD_CKPT=$CK LACOT_DEC_START=soft LACOT_DIAG_DUMP=1 LACOT_OUT_DIR=$OUTD $OFF $C2MA $EP -u experiments/scratch_lacot_rollout.py" >/dev/null; echo "ES-s$S -> $N ($J); eval -> $EN"
done
echo "== F'：medium 原配方八顆"
OUTD=results/night_0902/medium_base; mkdir -p "$OUTD"
for S in 20 21 22 23 24 25 26 27; do
  N=$(pick); J=$(sub $N MB-s$S - "OGBENCH_DATA_DIR=$ADATA $TRAIN_BASE LACOT_ENV=$MENV LACOT_K=8 LACOT_SEED=$S $APY -u experiments/scratch_lacot_rollout.py")
  CK=results/${PREM}_eorecon_ictr_tch0.5_emw0.999_wu500_norf_cd0.1_bci_s$S.pt; read EN EP ED <<< "$(evnode $S $N)"
  sub $EN MB-s$S-ema $J "OGBENCH_DATA_DIR=$ED $BASE LACOT_ENV=$MENV LACOT_K=8 LACOT_TEACHER_MIX=0.5 LACOT_LOAD_EMA=1 LACOT_LOAD_CKPT=$CK LACOT_DIAG_DUMP=1 LACOT_OUT_DIR=$OUTD $OFF $C2MA $EP -u experiments/scratch_lacot_rollout.py" >/dev/null; echo "MB-s$S -> $N ($J); eval -> $EN"
done
