#!/bin/bash
# 2026-09-04 intent 三接法先導（主人「開始吧」；設計 docs/DESIGN-2026-09-04-intent-three-wirings.md）
#   4 支、各 1 seed(s40)、large-stitch（對照 N3 .842 同 env）：
#   ITE  (i) embed   騎 N3 配方（凍s20+FSQ z 連續）
#   ITA  (ii) anchor 同上
#   ITR  (iii) residual ⛔ 不能 S1_FROM/FSQ（殘差語言重訓 stage1、無 FSQ）
#   ITE0 (i) embed 無FSQ+重訓版 —— ITR 的配對對照（同底、只差 target 變換）
#   先導判準：訓完不炸＋nll 形狀正常＋eval json 產出；⛔ 單顆分數只看方向不下結論。
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
SOFT20=results/${PRE}_eorecon_ictr_tch0.5_emw0.999_wu500_dssoft_norf_cd0.1_bci_s20.pt
FCK=results/night_0903/fsq/fsq_v2_d8L8_s20.pt
mkdir -p slurm/logs
sub() { local node=$1 name=$2 deps=$3; shift 3; local depflag=""
  [ "$deps" != "-" ] && depflag="--dependency=afterok:$deps"
  sbatch -p admin -A it -q great-mage --time=24:00:00 --nodelist=$node --gres=gpu:1 \
    --job-name=$name -o slurm/logs/%x-%j.out $depflag \
    --wrap "cd ~/Projects/lacot && env $*" | awk '{print $4}'; }
S=40
COMMON="OGBENCH_DATA_DIR=$ADATA $BASE0 LACOT_COND_DROP=0.1 $TRAIN0 LACOT_ENV=$LENV LACOT_K=8 LACOT_SEED=$S LACOT_EMA_W=0.999 LACOT_WARMUP=500 LACOT_DEC_START=soft"
FSQZ="LACOT_S1_FROM=$SOFT20 LACOT_FSQ_LOAD=$FCK LACOT_FSQ_SPACE=z LACOT_FSQ_TGT=snap LACOT_FSQ_ROUND=0"
EVC="$BASE0 LACOT_COND_DROP=0.1 LACOT_ENV=$LENV LACOT_K=8 LACOT_TEACHER_MIX=0.5 LACOT_LOAD_EMA=1 LACOT_DEC_START=soft"

# ITE / ITA：騎 N3 配方（jasmine / lady）
CK_ITE=results/${PRE}_eorecon_ictr_tch0.5_emw0.999_wu500_s1from_fsqzc8x8_ite_dssoft_norf_cd0.1_bci_s$S.pt
CK_ITA=results/${PRE}_eorecon_ictr_tch0.5_emw0.999_wu500_s1from_fsqzc8x8_ita_dssoft_norf_cd0.1_bci_s$S.pt
OUTD=results/day_0904/pilot_ite; mkdir -p "$OUTD"
J=$(sub jasmine IT-e - "$COMMON $FSQZ LACOT_INTENT=embed $APY -u experiments/scratch_lacot_rollout.py")
sub zeldajr IT-e-ema $J "OGBENCH_DATA_DIR=$ZDATA $EVC LACOT_LOAD_CKPT=$CK_ITE LACOT_FSQ_LOAD=$FCK LACOT_FSQ_SPACE=z LACOT_FSQ_TGT=snap LACOT_FSQ_ROUND=0 LACOT_INTENT=embed LACOT_OUT_DIR=$OUTD $OFF $C2MA $ZPY -u experiments/scratch_lacot_rollout.py" >/dev/null
echo "ITE -> jasmine ($J); eval -> zeldajr"
OUTD=results/day_0904/pilot_ita; mkdir -p "$OUTD"
J=$(sub lady IT-a - "$COMMON $FSQZ LACOT_INTENT=anchor $APY -u experiments/scratch_lacot_rollout.py")
sub zeldajr IT-a-ema $J "OGBENCH_DATA_DIR=$ZDATA $EVC LACOT_LOAD_CKPT=$CK_ITA LACOT_FSQ_LOAD=$FCK LACOT_FSQ_SPACE=z LACOT_FSQ_TGT=snap LACOT_FSQ_ROUND=0 LACOT_INTENT=anchor LACOT_OUT_DIR=$OUTD $OFF $C2MA $ZPY -u experiments/scratch_lacot_rollout.py" >/dev/null
echo "ITA -> lady ($J); eval -> zeldajr"

# ITR / ITE0：無 FSQ、重訓 stage1（moana / pocahontas）
CK_ITR=results/${PRE}_eorecon_ictr_tch0.5_emw0.999_wu500_itr_dssoft_norf_cd0.1_bci_s$S.pt
CK_ITE0=results/${PRE}_eorecon_ictr_tch0.5_emw0.999_wu500_ite_dssoft_norf_cd0.1_bci_s$S.pt
OUTD=results/day_0904/pilot_itr; mkdir -p "$OUTD"
J=$(sub moana IT-r - "$COMMON LACOT_INTENT=residual $APY -u experiments/scratch_lacot_rollout.py")
sub zeldajr IT-r-ema $J "OGBENCH_DATA_DIR=$ZDATA $EVC LACOT_LOAD_CKPT=$CK_ITR LACOT_INTENT=residual LACOT_OUT_DIR=$OUTD $OFF $C2MA $ZPY -u experiments/scratch_lacot_rollout.py" >/dev/null
echo "ITR -> moana ($J); eval -> zeldajr"
OUTD=results/day_0904/pilot_ite0; mkdir -p "$OUTD"
J=$(sub pocahontas IT-e0 - "$COMMON LACOT_INTENT=embed $APY -u experiments/scratch_lacot_rollout.py")
sub zeldajr IT-e0-ema $J "OGBENCH_DATA_DIR=$ZDATA $EVC LACOT_LOAD_CKPT=$CK_ITE0 LACOT_INTENT=embed LACOT_OUT_DIR=$OUTD $OFF $C2MA $ZPY -u experiments/scratch_lacot_rollout.py" >/dev/null
echo "ITE0 -> pocahontas ($J); eval -> zeldajr"
