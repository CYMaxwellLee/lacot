#!/bin/bash
# 2026-09-05 午批：intent-dropout 放量（先導儀器有效後的八顆；s40 用先導那顆併入＝8）
#   臂＝f27n 底座（凍 s27×ITE 無 FSQ）＋ LACOT_INTENT_DROP=0.3；每顆成對雙 eval（帶/零 intent）。
#   量：內化度 gap（帶 vs 零）×8 seed；對照 f27n .855±.029（無 drop）、s27 R0 .321。
#   ⛔ 帶 _bt 的批 CK 手寫全名（builder 順序 tch→bt→emw→wu→s1from→ite→idp→ds…）。
set -euo pipefail
cd ~/Projects/lacot
ZPY=$HOME/venvs/lacot-rocm/bin/python
APY=/archive/cymaxwelllee/LaCoT/.venv/bin/python
ZDATA=$HOME/data/ogbench
ADATA=/archive/cymaxwelllee/data/ogbench
BASE0="MUJOCO_GL=osmesa LACOT_ENC_OBJ=recon_ictr LACOT_LEARNED_REFINE=0 LACOT_BC_INDEP=1"
TRAIN0="LACOT_STEPS2=8000 LACOT_TEACHER_MIX=0.5 LACOT_EVAL_RS=0 LACOT_EVAL_EPISODES=2 LACOT_DIAG_TRAIN=1"
OFF="LACOT_DEV_EVAL=0 LACOT_EVAL_RS=0 LACOT_EVAL_EPISODES=50"
C2MA="LACOT_SUBGOAL=conf2 LACOT_SUB_POLICY=bc LACOT_GRAD_REFINE=1 LACOT_GRAD_R=0 LACOT_SUB_MAX_ARC=2 LACOT_FINISH_R=2.0"
LENV=pointmaze-large-stitch-v0
PRE=ckpt_large-stitch_self_K8_c256_ch4_st8000_T128_ep2_gu
SOFT27=results/${PRE}_eorecon_ictr_tch0.5_emw0.999_wu500_dssoft_norf_cd0.1_bci_s27.pt
O1=results/day_0905/idp8
O0=results/day_0905/idp8_zero
mkdir -p slurm/logs "$O1" "$O0"
[ -f "$SOFT27" ] || { echo "⛔ SOFT27 不在"; exit 1; }
sub() { local node=$1 name=$2 deps=$3; shift 3; local depflag=""
  [ "$deps" != "-" ] && depflag="--dependency=afterok:$deps"
  sbatch -p admin -A it -q great-mage --time=24:00:00 --nodelist=$node --gres=gpu:1 \
    --job-name=$name -o slurm/logs/%x-%j.out $depflag \
    --wrap "cd ~/Projects/lacot && env $*" | awk '{print $4}'; }
NODES=(jasmine lady moana pocahontas); i=0
for S in 41 42 43 44 45 46 47; do
  NODE=${NODES[$((i%4))]}; i=$((i+1))
  J=$(sub $NODE IDP8-s$S - "OGBENCH_DATA_DIR=$ADATA $BASE0 LACOT_COND_DROP=0.1 $TRAIN0 LACOT_ENV=$LENV LACOT_K=8 LACOT_SEED=$S LACOT_EMA_W=0.999 LACOT_WARMUP=500 LACOT_DEC_START=soft LACOT_S1_FROM=$SOFT27 LACOT_INTENT=embed LACOT_INTENT_DROP=0.3 LACOT_BOOT_TAG=idp1 $APY -u experiments/scratch_lacot_rollout.py")
  CK=results/${PRE}_eorecon_ictr_tch0.5_btidp1_emw0.999_wu500_s1from_ite_idp0.3_dssoft_norf_cd0.1_bci_s$S.pt
  EVBASE="OGBENCH_DATA_DIR=$ZDATA $BASE0 LACOT_COND_DROP=0.1 LACOT_ENV=$LENV LACOT_K=8 LACOT_TEACHER_MIX=0.5 LACOT_LOAD_EMA=1 LACOT_LOAD_CKPT=$CK LACOT_DEC_START=soft LACOT_INTENT=embed LACOT_INTENT_DROP=0.3 LACOT_BOOT_TAG=idp1"
  sub zeldajr IDP8-s$S-on $J "$EVBASE LACOT_OUT_DIR=$O1 $OFF $C2MA $ZPY -u experiments/scratch_lacot_rollout.py" >/dev/null
  sub zeldajr IDP8-s$S-z0 $J "$EVBASE LACOT_INTENT_ZERO=1 LACOT_OUT_DIR=$O0 $OFF $C2MA $ZPY -u experiments/scratch_lacot_rollout.py" >/dev/null
  echo "IDP8 s$S -> $NODE ($J)"
done
