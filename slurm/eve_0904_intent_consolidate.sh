#!/bin/bash
# 2026-09-04 傍晚鞏固批（主人「艦隊繼續把實驗跑下去」）：
#   ER  ITE×route 錨（訓推同分佈；部署形態測試）× s40~47 = 8
#   E27 ITE×s27 stage1（王座配方 × 最好 stage1；fsq_s27 已 fit recon .0064）× s40~47 = 8
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
SOFT27=results/${PRE}_eorecon_ictr_tch0.5_emw0.999_wu500_dssoft_norf_cd0.1_bci_s27.pt
FCK20=results/night_0903/fsq/fsq_v2_d8L8_s20.pt
FCK27=results/day_0904/fsq/fsq_v2_d8L8_s27.pt
mkdir -p slurm/logs
sub() { local node=$1 name=$2 deps=$3; shift 3; local depflag=""
  [ "$deps" != "-" ] && depflag="--dependency=afterok:$deps"
  sbatch -p admin -A it -q great-mage --time=24:00:00 --nodelist=$node --gres=gpu:1 \
    --job-name=$name -o slurm/logs/%x-%j.out $depflag \
    --wrap "cd ~/Projects/lacot && env $*" | awk '{print $4}'; }
NODES=(jasmine lady moana pocahontas); i=0
FSQC="LACOT_FSQ_SPACE=z LACOT_FSQ_TGT=snap LACOT_FSQ_ROUND=0"

run_it() { # $1=jobtag $2=S1_FROM $3=FCK $4=extra_train_env $5=ck_mid $6=outdir $7=extra_eval_env $8=boottag
  local JT=$1 S1=$2 FC=$3 XT=$4 MID=$5 OUTD=results/day_0904/$6 XE=$7 BT=$8
  mkdir -p "$OUTD"
  for S in 40 41 42 43 44 45 46 47; do
    local NODE=${NODES[$((i%4))]}; i=$((i+1))
    local J; J=$(sub $NODE $JT-s$S - "OGBENCH_DATA_DIR=$ADATA $BASE0 LACOT_COND_DROP=0.1 $TRAIN0 LACOT_ENV=$LENV LACOT_K=8 LACOT_SEED=$S LACOT_EMA_W=0.999 LACOT_WARMUP=500 LACOT_S1_FROM=$S1 LACOT_DEC_START=soft LACOT_FSQ_LOAD=$FC $FSQC LACOT_INTENT=embed $XT $BT $APY -u experiments/scratch_lacot_rollout.py")
    local CK=results/${PRE}_eorecon_ictr_tch0.5_emw0.999_wu500${MID}_s$S.pt
    sub zeldajr $JT-s$S-ema $J "OGBENCH_DATA_DIR=$ZDATA $BASE0 LACOT_COND_DROP=0.1 LACOT_ENV=$LENV LACOT_K=8 LACOT_TEACHER_MIX=0.5 LACOT_LOAD_EMA=1 LACOT_LOAD_CKPT=$CK LACOT_DEC_START=soft LACOT_FSQ_LOAD=$FC $FSQC LACOT_INTENT=embed $XE $BT LACOT_OUT_DIR=$OUTD $OFF $C2MA $ZPY -u experiments/scratch_lacot_rollout.py" >/dev/null
    echo "$JT s$S -> $NODE ($J)"
  done
}

# ER：ITE×route（凍 s20、fsq20；tag _iteR）
run_it ER "$SOFT20" "$FCK20" "LACOT_INTENT_SRC=route" "_s1from_fsqzc8x8_iteR_dssoft_norf_cd0.1_bci" pilot_iteR "LACOT_INTENT_SRC=route" ""
# E27：ITE×s27（凍 s27、fsq27；⛔ _bt 段防撞 s20 批 — 9/3 N5 同法）
# 🚨 9/4 首灑踩坑（決策 10 複發）：下行 MID 模板把 _btf27i 排在 wu500 後，但 builder 順序
#    是 tch→bt→emw→wu→s1from ⇒ 正確檔名 `_tch0.5_btf27i_emw0.999_wu500_s1from_...`。
#    當場 scancel 8 eval 重排（23753~23760、CK 整條寫死）。⛔ MID 模板容不下 bt 位 —
#    之後帶 _bt 的批別用 run_it 的 MID、CK 手寫全名。
run_it E27 "$SOFT27" "$FCK27" "" "_btf27i_s1from_fsqzc8x8_ite_dssoft_norf_cd0.1_bci" pilot_ite27 "" "LACOT_BOOT_TAG=f27i"
