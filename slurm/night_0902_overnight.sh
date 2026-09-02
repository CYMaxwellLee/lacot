#!/bin/bash
# 2026-09-02 過夜（主人 20:04「看看有什麼實驗可以跑過夜跑到明天中午」；⛔ 等主人點頭再送）
# B  守門 HG／純錨定 DA 推到 V8 另外四顆（s20/21/22/24；eval only）                      8 eval
# C  hard 綁定 ckpt（批 A 的 DH-s*）× {HG, DA}（依賴 A 的訓練 job）                        16 eval
# E  新 held-out s28~s35：原配方（V8 recipe）vs hard 綁定                                  16 train + 16 eval
# F  medium-stitch：定版配方＋hard 綁定 八顆                                                8 train + 8 eval
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
  sbatch -p admin -A it -q great-mage --time=24:00:00 --nodelist=$node --gres=gpu:1 \
    --job-name=$name -o slurm/logs/%x-%j.out $depflag \
    --wrap "cd ~/Projects/lacot && env $*" | awk '{print $4}'; }
NODES=(lady moana pocahontas); i=0
pick() { local n=${NODES[$((i%3))]}; i=$((i+1)); echo $n; }
evnode() { if [ $(($1 % 2)) -eq 0 ]; then echo "zeldajr $ZPY $ZDATA"; else echo "$2 $APY $ADATA"; fi; }
SECT=${SECT:-"B C E F"}
if [[ " $SECT " == *" B "* ]]; then
  echo "== B：HG／DA 推到 s20/21/22/24（V8 ckpt、eval only）"
  for A in HG DA; do
    X=$([ $A = HG ] && echo "LACOT_SUB_HEADGUARD=3.0" || echo "LACOT_DEC_ANCHOR=1"); OUTD=results/night_0902/guard_$A; mkdir -p "$OUTD"
    for S in 20 21 22 24; do
      CK=results/${PREL}_eorecon_ictr_tch0.5_emw0.999_wu500_norf_cd0.1_bci_s$S.pt; N=$(pick); read EN EP ED <<< "$(evnode $S $N)"
      sub $EN $A-s$S - "OGBENCH_DATA_DIR=$ED $BASE LACOT_ENV=$LENV LACOT_K=8 LACOT_TEACHER_MIX=0.5 LACOT_LOAD_EMA=1 LACOT_LOAD_CKPT=$CK $X LACOT_DIAG_DUMP=1 LACOT_OUT_DIR=$OUTD $OFF $C2MA $EP -u experiments/scratch_lacot_rollout.py" >/dev/null; echo "$A-s$S -> $EN"
    done
  done
fi
if [[ " $SECT " == *" C "* ]]; then
  echo "== C：hard 綁定 ckpt × {HG, DA}（依賴 A 的 DH 訓練 job）"
  declare -A DH; for kv in $DH_IDS; do DH[${kv%%=*}]=${kv#*=}; done   # DH_IDS="DH-s20=22448 ..." 由環境傳入
  for A in HG DA; do
    X=$([ $A = HG ] && echo "LACOT_SUB_HEADGUARD=3.0" || echo "LACOT_DEC_ANCHOR=1"); OUTD=results/night_0902/dshard_$A; mkdir -p "$OUTD"
    for S in 20 21 22 23 24 25 26 27; do
      CK=results/${PREL}_eorecon_ictr_tch0.5_emw0.999_wu500_dshard_norf_cd0.1_bci_s$S.pt; N=$(pick); read EN EP ED <<< "$(evnode $S $N)"; DEP=${DH[DH-s$S]:--}
      sub $EN C$A-s$S $DEP "OGBENCH_DATA_DIR=$ED $BASE LACOT_ENV=$LENV LACOT_K=8 LACOT_TEACHER_MIX=0.5 LACOT_LOAD_EMA=1 LACOT_LOAD_CKPT=$CK LACOT_DEC_START=hard $X LACOT_DIAG_DUMP=1 LACOT_OUT_DIR=$OUTD $OFF $C2MA $EP -u experiments/scratch_lacot_rollout.py" >/dev/null; echo "C$A-s$S -> $EN (dep $DEP)"
    done
  done
fi
if [[ " $SECT " == *" E "* ]]; then
  echo "== E：新 held-out s28~s35：原配方 vs hard 綁定"
  for MODE in ${EMODES:-base hard}; do
    X=$([ $MODE = hard ] && echo "LACOT_DEC_START=hard" || echo ""); TAG=$([ $MODE = hard ] && echo "_dshard" || echo ""); OUTD=results/night_0902/heldout2_$MODE; mkdir -p "$OUTD"; A=$([ $MODE = hard ] && echo EH || echo EB)
    for S in 28 29 30 31 32 33 34 35; do
      N=$(pick); J=$(sub $N $A-s$S - "OGBENCH_DATA_DIR=$ADATA $TRAIN_BASE LACOT_ENV=$LENV LACOT_K=8 LACOT_SEED=$S $X $APY -u experiments/scratch_lacot_rollout.py")
      CK=results/${PREL}_eorecon_ictr_tch0.5_emw0.999_wu500${TAG}_norf_cd0.1_bci_s$S.pt; read EN EP ED <<< "$(evnode $S $N)"
      sub $EN $A-s$S-ema $J "OGBENCH_DATA_DIR=$ED $BASE LACOT_ENV=$LENV LACOT_K=8 LACOT_TEACHER_MIX=0.5 LACOT_LOAD_EMA=1 LACOT_LOAD_CKPT=$CK $X LACOT_DIAG_DUMP=1 LACOT_OUT_DIR=$OUTD $OFF $C2MA $EP -u experiments/scratch_lacot_rollout.py" >/dev/null; echo "$A-s$S -> $N ($J); eval -> $EN"
    done
  done
fi
if [[ " $SECT " == *" F "* ]]; then
  echo "== F：medium-stitch 定版配方＋hard 綁定 八顆"
  OUTD=results/night_0902/medium_hard; mkdir -p "$OUTD"
  for S in 20 21 22 23 24 25 26 27; do
    N=$(pick); J=$(sub $N MH-s$S - "OGBENCH_DATA_DIR=$ADATA $TRAIN_BASE LACOT_ENV=$MENV LACOT_K=8 LACOT_SEED=$S LACOT_DEC_START=hard $APY -u experiments/scratch_lacot_rollout.py")
    CK=results/${PREM}_eorecon_ictr_tch0.5_emw0.999_wu500_dshard_norf_cd0.1_bci_s$S.pt; read EN EP ED <<< "$(evnode $S $N)"
    sub $EN MH-s$S-ema $J "OGBENCH_DATA_DIR=$ED $BASE LACOT_ENV=$MENV LACOT_K=8 LACOT_TEACHER_MIX=0.5 LACOT_LOAD_EMA=1 LACOT_LOAD_CKPT=$CK LACOT_DEC_START=hard LACOT_DIAG_DUMP=1 LACOT_OUT_DIR=$OUTD $OFF $C2MA $EP -u experiments/scratch_lacot_rollout.py" >/dev/null; echo "MH-s$S -> $N ($J); eval -> $EN"
  done
fi
