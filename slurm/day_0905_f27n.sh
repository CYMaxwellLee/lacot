#!/bin/bash
# 2026-09-05 早批（主人「灑吧」）：f27n＝凍 s27 stage1 × ITE intent、⛔ 無 FSQ × s40~47 = 8
#   問題：F27 已證 FSQ 有罪、intent 無罪 ⇒ s27(.857) 加 intent、拿掉 FSQ ＝ 新王座候補。
#   對照鏈：s27 .857（N5 凍）／E27 s27+FSQ+ITE .667／F27 s27+FSQ .673／f27n s27+ITE = ?
#   ⛔ 帶 _bt 的批不走 MID 模板，CK 手寫全名（9/4 E27 事故、決策 10）。
#   配方＝SCE（day_0904_intent_scale 的 ITE 臂）env 拿掉 FSQ 四個變數
#     ＋ LACOT_S1_FROM=SOFT27 ＋ LACOT_BOOT_TAG=f27n 防撞。
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
OUTD=results/day_0905/f27n
mkdir -p slurm/logs "$OUTD"
[ -f "$SOFT27" ] || { echo "⛔ SOFT27 不在：$SOFT27"; exit 1; }
sub() { local node=$1 name=$2 deps=$3; shift 3; local depflag=""
  [ "$deps" != "-" ] && depflag="--dependency=afterok:$deps"
  sbatch -p admin -A it -q great-mage --time=24:00:00 --nodelist=$node --gres=gpu:1 \
    --job-name=$name -o slurm/logs/%x-%j.out $depflag \
    --wrap "cd ~/Projects/lacot && env $*" | awk '{print $4}'; }
NODES=(jasmine lady moana pocahontas); i=0
for S in 40 41 42 43 44 45 46 47; do
  NODE=${NODES[$((i%4))]}; i=$((i+1))
  J=$(sub $NODE F27N-s$S - "OGBENCH_DATA_DIR=$ADATA $BASE0 LACOT_COND_DROP=0.1 $TRAIN0 LACOT_ENV=$LENV LACOT_K=8 LACOT_SEED=$S LACOT_EMA_W=0.999 LACOT_WARMUP=500 LACOT_DEC_START=soft LACOT_S1_FROM=$SOFT27 LACOT_INTENT=embed LACOT_BOOT_TAG=f27n $APY -u experiments/scratch_lacot_rollout.py")
  # ⛔ CK 手寫全名（builder 順序：tch→bt→emw→wu→s1from→[fsq 無]→ite→dssoft→norf→cd→bci；
  #    對過 builder code L2262-2309 ＋ E27/F27/ITE0 三批實檔）
  CK=results/${PRE}_eorecon_ictr_tch0.5_btf27n_emw0.999_wu500_s1from_ite_dssoft_norf_cd0.1_bci_s$S.pt
  sub zeldajr F27N-s$S-ema $J "OGBENCH_DATA_DIR=$ZDATA $BASE0 LACOT_COND_DROP=0.1 LACOT_ENV=$LENV LACOT_K=8 LACOT_TEACHER_MIX=0.5 LACOT_LOAD_EMA=1 LACOT_LOAD_CKPT=$CK LACOT_DEC_START=soft LACOT_INTENT=embed LACOT_BOOT_TAG=f27n LACOT_OUT_DIR=$OUTD $OFF $C2MA $ZPY -u experiments/scratch_lacot_rollout.py" >/dev/null
  echo "F27N s$S -> $NODE ($J)"
done
