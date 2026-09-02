#!/bin/bash
# 2026-09-02 白天（主人 11:11「治療 stability／處理翻車」）：兩批診斷、都不改配方
# A. V8 失敗地理：定版八顆裡最低三顆 s23/s25/s26＋最高 s27，官方 eval 加 LACOT_DIAG_DUMP=1
#    ⇒ 問「失敗是不是都聚在同一塊荒漠」（資料病 vs 初始化抽籤）。只 eval、不重訓。
# B. dz2 重現：boot_s2_dz2.npz 不動、LACOT_SEED=2 不動，只換資料流 seed d∈{2,12,22,32}
#    （d=2＝跟原 dz2 同 rng、當重跑對照；其餘三顆量「同題目不同抽樣」的散佈）
#    ⇒ 0.880 重現＝課程真、dz3 是真退步；散開＝階梯收回改講方差。
# ⛔ eval 端檔名不帶訓練 tag（9/1 兩次互蓋）⇒ B 的四支各自分目錄 dz2rep_d$D；A 四顆 seed 不同、共用一目錄。
set -euo pipefail
cd ~/Projects/lacot
ZPY=$HOME/venvs/lacot-rocm/bin/python
APY=/archive/cymaxwelllee/LaCoT/.venv/bin/python
ZDATA=$HOME/data/ogbench
ADATA=/archive/cymaxwelllee/data/ogbench
BASE="MUJOCO_GL=osmesa LACOT_ENC_OBJ=recon_ictr LACOT_LEARNED_REFINE=0 LACOT_COND_DROP=0.1 LACOT_BC_INDEP=1"
TRAIN_BASE="$BASE LACOT_STEPS2=8000 LACOT_TEACHER_MIX=0.5 LACOT_EVAL_RS=0 LACOT_EVAL_EPISODES=2 LACOT_DIAG_TRAIN=1"
OFF="LACOT_DEV_EVAL=0 LACOT_EVAL_RS=0 LACOT_EVAL_EPISODES=50"
C2MA="LACOT_SUBGOAL=conf2 LACOT_SUB_POLICY=bc LACOT_GRAD_REFINE=1 LACOT_GRAD_R=0 LACOT_SUB_MAX_ARC=2 LACOT_FINISH_R=2.0"
LENV=pointmaze-large-stitch-v0
PRE=ckpt_large-stitch_self_K8_c256_ch4_st8000_T128_ep2_gu
mkdir -p results/night_0902/v8diag slurm/logs

sub() { local node=$1 name=$2 deps=$3; shift 3; local depflag=""
  [ "$deps" != "-" ] && depflag="--dependency=afterok:$deps"
  sbatch -p admin -A it -q great-mage --time=24:00:00 --nodelist=$node --gres=gpu:1 \
    --job-name=$name -o slurm/logs/%x-%j.out $depflag \
    --wrap "cd ~/Projects/lacot && env $*" | awk '{print $4}'; }

echo "== A. V8 失敗地理（只 eval）"
for S in 23 25 26 27; do
  CK=results/${PRE}_eorecon_ictr_tch0.5_emw0.999_wu500_norf_cd0.1_bci_s$S.pt
  [ -f "$CK" ] || { echo "⛔ 缺 ckpt $CK"; exit 1; }
  J=$(sub zeldajr SD-s$S-diag - "OGBENCH_DATA_DIR=$ZDATA $BASE LACOT_ENV=$LENV LACOT_K=8 LACOT_TEACHER_MIX=0.5 LACOT_LOAD_EMA=1 LACOT_LOAD_CKPT=$CK LACOT_DIAG_DUMP=1 LACOT_OUT_DIR=results/night_0902/v8diag $OFF $C2MA $ZPY -u experiments/scratch_lacot_rollout.py")
  echo "SD-s$S-diag -> zeldajr ($J)"
done

echo "== B. dz2 重現（boot 不動、換資料流 seed）"
[ -f results/boot_s2_dz2.npz ] || { echo "⛔ 缺 boot_s2_dz2.npz"; exit 1; }
i=0
for D in 2 12 22 32; do
  NODE=$(echo lady moana pocahontas | cut -d" " -f$((i%3+1))); i=$((i+1))
  mkdir -p results/night_0902/dz2rep_d$D
  J=$(sub $NODE R2-d$D - "OGBENCH_DATA_DIR=$ADATA $TRAIN_BASE LACOT_ENV=$LENV LACOT_K=8 LACOT_SEED=2 LACOT_DATA_SEED=$D LACOT_EMA_W=0.999 LACOT_BOOT_DATA=results/boot_s2_dz2.npz LACOT_BOOT_TAG=dz2 $APY -u experiments/scratch_lacot_rollout.py")
  CK=results/${PRE}_eorecon_ictr_tch0.5_btdz2_emw0.999_dseed${D}_norf_cd0.1_bci_s2.pt
  sub zeldajr R2-d$D-ema $J "OGBENCH_DATA_DIR=$ZDATA $BASE LACOT_ENV=$LENV LACOT_K=8 LACOT_TEACHER_MIX=0.5 LACOT_LOAD_EMA=1 LACOT_LOAD_CKPT=$CK LACOT_DIAG_DUMP=1 LACOT_OUT_DIR=results/night_0902/dz2rep_d$D $OFF $C2MA $ZPY -u experiments/scratch_lacot_rollout.py" >/dev/null
  echo "R2-d$D -> $NODE ($J) ; eval -> zeldajr"
done
echo "== 共 12 支（A 4 eval；B 4 train + 4 eval）"
