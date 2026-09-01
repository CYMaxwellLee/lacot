#!/bin/bash
# 2026-09-01 夜：seed 病理並行治療（主人 18:13「並行吧，反正晚上都可以丟著跑」）
# 病理（當日驗屍、docs/FINDINGS 待補 0901 節）：
#   s1/s4＝encoder 起步卡死型（全油門起步）；s3＝末段震盪型（EMA 已治）；
#   s2＝表示局部畸形型（中部 U 形死區、決定論卡死；資料稀疏走廊＝BC 跨 seed 共通爛）。
# 治標(lady)：large 新 seed 農場 s8~s13 —— s8/9/10 原配方（純重抽爛顆率基線）
#   ＋ s11/12/13 LACOT_WARMUP=500（起步藥效直接對照）。全帶 EMA（default 配方）。
# 治本(moana/pocahontas)：s2 定點治療對決 ——
#   A(moana)      自舉荒漠補課：s2 顆 BOOT_GEN+DESERT 生成 → BOOT_DATA 蒸餾重訓（_btdz1）
#   B(pocahontas) 荒漠重採樣：LACOT_DATA_RESAMPLE=1 重訓（_rs）
# 評測(zeldajr)：全部 ema 臂官方；s2 兩帖加 LACOT_DIAG_DUMP=1（死區失敗 before/after，
#   before＝當日 results/diag_0901/ 的 s2 基線 0.632/91fail）。
#   ⛔ 全部 eval 產物走 LACOT_OUT_DIR=results/night_0901（不蓋 R2 正式檔）。
set -euo pipefail
cd ~/Projects/lacot
ZPY=$HOME/venvs/lacot-rocm/bin/python
APY=/archive/cymaxwelllee/LaCoT/.venv/bin/python
ZDATA=$HOME/data/ogbench
ADATA=/archive/cymaxwelllee/data/ogbench
BASE="MUJOCO_GL=osmesa LACOT_ENC_OBJ=recon_ictr LACOT_LEARNED_REFINE=0 LACOT_COND_DROP=0.1 LACOT_BC_INDEP=1"
TRAIN_BASE="$BASE LACOT_STEPS2=8000 LACOT_TEACHER_MIX=0.5 LACOT_EVAL_RS=0 LACOT_EVAL_EPISODES=2"
OFF="LACOT_DEV_EVAL=0 LACOT_EVAL_RS=0 LACOT_EVAL_EPISODES=50"
C2MA="LACOT_SUBGOAL=conf2 LACOT_SUB_POLICY=bc LACOT_GRAD_REFINE=1 LACOT_GRAD_R=0 LACOT_SUB_MAX_ARC=2 LACOT_FINISH_R=2.0"
LENV=pointmaze-large-stitch-v0
OUT="LACOT_OUT_DIR=results/night_0901"
mkdir -p results/night_0901 slurm/logs

sub() {  # sub <node> <name> <deps-or-> <env+cmd string>
  local node=$1 name=$2 deps=$3; shift 3
  local depflag=""
  [ "$deps" != "-" ] && depflag="--dependency=afterok:$deps"
  sbatch -p admin -A it -q great-mage --time=24:00:00 --nodelist=$node --gres=gpu:1 \
    --job-name=$name -o slurm/logs/%x-%j.out $depflag \
    --wrap "cd ~/Projects/lacot && env $*" | awk '{print $4}'
}

PRE=ckpt_large-stitch_self_K8_c256_ch4_st8000_T128_ep2_gu
S2EMW=results/${PRE}_eorecon_ictr_tch0.5_emw0.999_norf_cd0.1_bci_s2.pt

echo "== 治標：lady 農場 s8~s13（3 原配方 + 3 warmup500）→ zeldajr ema 臂"
for S in 8 9 10; do
  J=$(sub lady F-L$S - "OGBENCH_DATA_DIR=$ADATA $TRAIN_BASE LACOT_ENV=$LENV LACOT_K=8 LACOT_SEED=$S LACOT_EMA_W=0.999 $APY -u experiments/scratch_lacot_rollout.py")
  CK=results/${PRE}_eorecon_ictr_tch0.5_emw0.999_norf_cd0.1_bci_s$S.pt
  sub zeldajr F-L$S-ema $J "OGBENCH_DATA_DIR=$ZDATA $BASE LACOT_ENV=$LENV LACOT_K=8 LACOT_TEACHER_MIX=0.5 LACOT_LOAD_EMA=1 LACOT_LOAD_CKPT=$CK $OUT $OFF $C2MA $ZPY -u experiments/scratch_lacot_rollout.py"
done
for S in 11 12 13; do
  J=$(sub lady F-W$S - "OGBENCH_DATA_DIR=$ADATA $TRAIN_BASE LACOT_ENV=$LENV LACOT_K=8 LACOT_SEED=$S LACOT_EMA_W=0.999 LACOT_WARMUP=500 $APY -u experiments/scratch_lacot_rollout.py")
  CK=results/${PRE}_eorecon_ictr_tch0.5_emw0.999_wu500_norf_cd0.1_bci_s$S.pt
  sub zeldajr F-W$S-ema $J "OGBENCH_DATA_DIR=$ZDATA $BASE LACOT_ENV=$LENV LACOT_K=8 LACOT_TEACHER_MIX=0.5 LACOT_LOAD_EMA=1 LACOT_LOAD_CKPT=$CK $OUT $OFF $C2MA $ZPY -u experiments/scratch_lacot_rollout.py"
done

echo "== 治本A：s2 自舉荒漠補課鏈（moana 生成→重訓 → zeldajr ema+diag）"
J1=$(sub moana DZ-gen - "OGBENCH_DATA_DIR=$ADATA $BASE LACOT_ENV=$LENV LACOT_K=8 LACOT_TEACHER_MIX=0.5 LACOT_LOAD_CKPT=$S2EMW LACOT_BOOT_GEN=results/boot_s2_dz1.npz LACOT_BOOT_DESERT=1 LACOT_BOOT_Q=512 LACOT_BOOT_M=8 LACOT_BOOT_RMIN=8 LACOT_BOOT_RMAX=25 $APY -u experiments/scratch_lacot_rollout.py")
J2=$(sub moana DZ-train $J1 "OGBENCH_DATA_DIR=$ADATA $TRAIN_BASE LACOT_ENV=$LENV LACOT_K=8 LACOT_SEED=2 LACOT_EMA_W=0.999 LACOT_BOOT_DATA=results/boot_s2_dz1.npz LACOT_BOOT_TAG=dz1 $APY -u experiments/scratch_lacot_rollout.py")
CKA=results/${PRE}_eorecon_ictr_tch0.5_btdz1_emw0.999_norf_cd0.1_bci_s2.pt
sub zeldajr DZ-ema $J2 "OGBENCH_DATA_DIR=$ZDATA $BASE LACOT_ENV=$LENV LACOT_K=8 LACOT_TEACHER_MIX=0.5 LACOT_LOAD_EMA=1 LACOT_LOAD_CKPT=$CKA LACOT_DIAG_DUMP=1 $OUT $OFF $C2MA $ZPY -u experiments/scratch_lacot_rollout.py"

echo "== 治本B：s2 荒漠重採樣重訓（pocahontas → zeldajr ema+diag）"
J3=$(sub pocahontas RS-train - "OGBENCH_DATA_DIR=$ADATA $TRAIN_BASE LACOT_ENV=$LENV LACOT_K=8 LACOT_SEED=2 LACOT_EMA_W=0.999 LACOT_DATA_RESAMPLE=1 $APY -u experiments/scratch_lacot_rollout.py")
CKB=results/${PRE}_eorecon_ictr_tch0.5_emw0.999_rs_norf_cd0.1_bci_s2.pt
sub zeldajr RS-ema $J3 "OGBENCH_DATA_DIR=$ZDATA $BASE LACOT_ENV=$LENV LACOT_K=8 LACOT_TEACHER_MIX=0.5 LACOT_LOAD_EMA=1 LACOT_LOAD_CKPT=$CKB LACOT_DIAG_DUMP=1 $OUT $OFF $C2MA $ZPY -u experiments/scratch_lacot_rollout.py"

echo "== 全部排入：治標 6訓+6評、治本A 3鏈、治本B 2鏈 ＝ 17 支"
