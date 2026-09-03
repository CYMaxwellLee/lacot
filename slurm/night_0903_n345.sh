#!/bin/bash
# 2026-09-03 夜批 N3/N4/N5（主人 19:07「小Luna，跑」；計畫見 DESIGN-DRAFT-2026-09-03-discrete-two-level §四）
#   N3 只壓縮不離散：z甲 拿掉 round（FSQ_ROUND=0）×8 ⇒ 把 z甲 的 −.08 拆成壓縮費/離散費
#   N4 端到端腿加壓讀 u：z乙 × COND_DROP {0.3,0.5} ×8 ⇒ R0 .505 還能推多高（主人 9/2「加強讓模型讀u」）
#   N5 凍 stage1 穩健性：s26/s27 各凍 ×8 seed（無 FSQ）⇒ 防「s20 運氣好」；⛔ ckpt 用 _bt 段防撞凍 s20 批
#   N2 判定不重跑：sig16 bound 佔格分佈與 tanh 無實質差（learned projection 吸收 bound 形狀）、recon 同級。
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
NODES=(jasmine lady moana pocahontas); i=0
# ⛔ 別用 $(fn) 取節點——subshell 裡 i 遞增不回傳，40 支全擠第一台（9/3 夜踩過）

# ---- N3 只壓縮不離散（z 甲、ROUND=0、fsq v2）----
OUTD=results/night_0903/fsqz_cont; mkdir -p "$OUTD"
for S in ${SEEDS:-40 41 42 43 44 45 46 47}; do
  NODE=${NODES[$((i%4))]}; i=$((i+1))
  J=$(sub $NODE N3-s$S - "OGBENCH_DATA_DIR=$ADATA $BASE0 LACOT_COND_DROP=0.1 $TRAIN0 LACOT_ENV=$LENV LACOT_K=8 LACOT_SEED=$S LACOT_EMA_W=0.999 LACOT_WARMUP=500 LACOT_S1_FROM=$SOFT20 LACOT_DEC_START=soft LACOT_FSQ_LOAD=$FCK LACOT_FSQ_SPACE=z LACOT_FSQ_TGT=snap LACOT_FSQ_ROUND=0 $APY -u experiments/scratch_lacot_rollout.py")
  CK=results/${PRE}_eorecon_ictr_tch0.5_emw0.999_wu500_s1from_fsqzc8x8_dssoft_norf_cd0.1_bci_s$S.pt
  if [ $((S % 2)) -eq 0 ]; then EN=zeldajr; EP=$ZPY; ED=$ZDATA; else EN=$NODE; EP=$APY; ED=$ADATA; fi
  sub $EN N3-s$S-ema $J "OGBENCH_DATA_DIR=$ED $BASE0 LACOT_COND_DROP=0.1 LACOT_ENV=$LENV LACOT_K=8 LACOT_TEACHER_MIX=0.5 LACOT_LOAD_EMA=1 LACOT_LOAD_CKPT=$CK LACOT_DEC_START=soft LACOT_FSQ_LOAD=$FCK LACOT_FSQ_SPACE=z LACOT_FSQ_TGT=snap LACOT_FSQ_ROUND=0 LACOT_OUT_DIR=$OUTD $OFF $C2MA $EP -u experiments/scratch_lacot_rollout.py" >/dev/null
  echo "N3 s$S -> $NODE ($J); eval -> $EN"
done

# ---- N4 z乙 × COND_DROP {0.3,0.5}（fsq v2）----
for CD in 0.3 0.5; do
  CDTAG=${CD/0./}; OUTD=results/night_0903/fsqz_dq_cd$CDTAG; mkdir -p "$OUTD"
  for S in ${SEEDS:-40 41 42 43 44 45 46 47}; do
    NODE=${NODES[$((i%4))]}; i=$((i+1))
    J=$(sub $NODE N4c$CDTAG-s$S - "OGBENCH_DATA_DIR=$ADATA $BASE0 LACOT_COND_DROP=$CD $TRAIN0 LACOT_ENV=$LENV LACOT_K=8 LACOT_SEED=$S LACOT_EMA_W=0.999 LACOT_WARMUP=500 LACOT_S1_FROM=$SOFT20 LACOT_DEC_START=soft LACOT_FSQ_LOAD=$FCK LACOT_FSQ_SPACE=z LACOT_FSQ_TGT=dequant $APY -u experiments/scratch_lacot_rollout.py")
    CK=results/${PRE}_eorecon_ictr_tch0.5_emw0.999_wu500_s1from_fsqzd8x8_dssoft_norf_cd${CD}_bci_s$S.pt
    if [ $((S % 2)) -eq 0 ]; then EN=zeldajr; EP=$ZPY; ED=$ZDATA; else EN=$NODE; EP=$APY; ED=$ADATA; fi
    sub $EN N4c$CDTAG-s$S-ema $J "OGBENCH_DATA_DIR=$ED $BASE0 LACOT_COND_DROP=$CD LACOT_ENV=$LENV LACOT_K=8 LACOT_TEACHER_MIX=0.5 LACOT_LOAD_EMA=1 LACOT_LOAD_CKPT=$CK LACOT_DEC_START=soft LACOT_FSQ_LOAD=$FCK LACOT_FSQ_SPACE=z LACOT_FSQ_TGT=dequant LACOT_OUT_DIR=$OUTD $OFF $C2MA $EP -u experiments/scratch_lacot_rollout.py" >/dev/null
    echo "N4 cd$CD s$S -> $NODE ($J); eval -> $EN"
  done
done

# ---- N5 凍 s26/s27（無 FSQ；⛔ _bt 段防撞凍 s20 批的 ckpt）----
for SRC in 26 27; do
  SRCCK=results/${PRE}_eorecon_ictr_tch0.5_emw0.999_wu500_dssoft_norf_cd0.1_bci_s$SRC.pt
  [ -f "$SRCCK" ] || { echo "⛔ 凍源不在：$SRCCK"; exit 1; }
  OUTD=results/night_0903/dialect_s$SRC; mkdir -p "$OUTD"
  for S in ${SEEDS:-40 41 42 43 44 45 46 47}; do
    NODE=${NODES[$((i%4))]}; i=$((i+1))
    J=$(sub $NODE N5f$SRC-s$S - "OGBENCH_DATA_DIR=$ADATA $BASE0 LACOT_COND_DROP=0.1 $TRAIN0 LACOT_ENV=$LENV LACOT_K=8 LACOT_SEED=$S LACOT_EMA_W=0.999 LACOT_WARMUP=500 LACOT_S1_FROM=$SRCCK LACOT_DEC_START=soft LACOT_BOOT_TAG=f$SRC $APY -u experiments/scratch_lacot_rollout.py")
    CK=results/${PRE}_eorecon_ictr_tch0.5_btf${SRC}_emw0.999_wu500_s1from_dssoft_norf_cd0.1_bci_s$S.pt
    if [ $((S % 2)) -eq 0 ]; then EN=zeldajr; EP=$ZPY; ED=$ZDATA; else EN=$NODE; EP=$APY; ED=$ADATA; fi
    sub $EN N5f$SRC-s$S-ema $J "OGBENCH_DATA_DIR=$ED $BASE0 LACOT_COND_DROP=0.1 LACOT_ENV=$LENV LACOT_K=8 LACOT_TEACHER_MIX=0.5 LACOT_LOAD_EMA=1 LACOT_LOAD_CKPT=$CK LACOT_DEC_START=soft LACOT_BOOT_TAG=f$SRC LACOT_OUT_DIR=$OUTD $OFF $C2MA $EP -u experiments/scratch_lacot_rollout.py" >/dev/null
    echo "N5 凍s$SRC s$S -> $NODE ($J); eval -> $EN"
  done
done
