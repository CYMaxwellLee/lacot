#!/bin/bash
# 2026-09-01 夜第二彈：主人裁的四項治療＋seed 病因四路診斷（主人 20:35「可以灑一些診斷實驗」）
# 治療（主人 20:29 裁）：
#   A(3鏈) warmup 配對救活：已知爛的 s14/15/16 同 seed 加 warmup 重訓 —— 因果級證據
#   B(1鏈) dz2 二輪補課：btdz1 顆再生成（curriculum：它現在更會荒漠）→ 攻 task2 最深處
#   C(2鏈) 病型特異性：s1/s4（起步卡死型）也補課 —— 預測無效或小效（病在表示不在教材）
#   D(1鏈) 安全性：好顆 s5 補課 —— 驗「補課對健康顆無害」
# 診斷（嫌疑人×4）：
#   E(2鏈) seed 拆分 2×2 交叉組：init s14×data s8、init s8×data s14 —— 壞 init vs 壞資料順序
#   F(2鏈) 好顆前300步逐幀對照：s8/s9 顯式 DATA_SEED=自己（行為不變、檔名隔離防蓋）
#   G(3鏈) lr 減半 vs warmup 對決：s14/15/16 × LR_SCALE=0.5（無 warmup）
# 全部訓練帶 LACOT_DIAG_TRAIN=1（前300步逐幀）；⛔ eval 一律 OUT_DIR=results/night_0901/p2 防互蓋
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
OUT="LACOT_OUT_DIR=results/night_0901/p2"
PRE=ckpt_large-stitch_self_K8_c256_ch4_st8000_T128_ep2_gu
mkdir -p results/night_0901/p2 slurm/logs

sub() { local node=$1 name=$2 deps=$3; shift 3; local depflag=""
  [ "$deps" != "-" ] && depflag="--dependency=afterok:$deps"
  sbatch -p admin -A it -q great-mage --time=24:00:00 --nodelist=$node --gres=gpu:1 \
    --job-name=$name -o slurm/logs/%x-%j.out $depflag \
    --wrap "cd ~/Projects/lacot && env $*" | awk '{print $4}'; }

EV() { # EV <deps> <name> <ckpt> [extra]
  local deps=$1 name=$2 ck=$3; shift 3
  sub zeldajr $name $deps "OGBENCH_DATA_DIR=$ZDATA $BASE LACOT_ENV=$LENV LACOT_K=8 LACOT_TEACHER_MIX=0.5 LACOT_LOAD_EMA=1 LACOT_LOAD_CKPT=$ck $* $OUT $OFF $C2MA $ZPY -u experiments/scratch_lacot_rollout.py" >/dev/null; }

echo "== A：warmup 配對救活 s14/15/16"
i=0
for S in 14 15 16; do
  NODE=$(echo lady moana pocahontas | cut -d" " -f$((i%3+1))); i=$((i+1))
  J=$(sub $NODE A-W$S - "OGBENCH_DATA_DIR=$ADATA $TRAIN_BASE LACOT_ENV=$LENV LACOT_K=8 LACOT_SEED=$S LACOT_EMA_W=0.999 LACOT_WARMUP=500 $APY -u experiments/scratch_lacot_rollout.py")
  EV $J A-W$S-ema results/${PRE}_eorecon_ictr_tch0.5_emw0.999_wu500_norf_cd0.1_bci_s$S.pt
  echo "A-W$S -> $NODE ($J)"
done

echo "== B：dz2 二輪補課（btdz1 顆再生成）"
DZ1=results/${PRE}_eorecon_ictr_tch0.5_btdz1_emw0.999_norf_cd0.1_bci_s2.pt
J1=$(sub moana B-gen2 - "OGBENCH_DATA_DIR=$ADATA $BASE LACOT_ENV=$LENV LACOT_K=8 LACOT_TEACHER_MIX=0.5 LACOT_LOAD_CKPT=$DZ1 LACOT_BOOT_GEN=results/boot_s2_dz2.npz LACOT_BOOT_DESERT=1 LACOT_BOOT_Q=512 LACOT_BOOT_M=8 LACOT_BOOT_RMIN=8 LACOT_BOOT_RMAX=25 $APY -u experiments/scratch_lacot_rollout.py")
J2=$(sub moana B-dz2 $J1 "OGBENCH_DATA_DIR=$ADATA $TRAIN_BASE LACOT_ENV=$LENV LACOT_K=8 LACOT_SEED=2 LACOT_EMA_W=0.999 LACOT_BOOT_DATA=results/boot_s2_dz2.npz LACOT_BOOT_TAG=dz2 $APY -u experiments/scratch_lacot_rollout.py")
EV $J2 B-dz2-ema results/${PRE}_eorecon_ictr_tch0.5_btdz2_emw0.999_norf_cd0.1_bci_s2.pt LACOT_DIAG_DUMP=1
echo "B chain: $J1 -> $J2"

echo "== C：病型特異性 s1/s4 補課（預測無效）＋ D：s5 安全性"
i=0
for S in 1 4 5; do
  NODE=$(echo lady pocahontas moana | cut -d" " -f$((i%3+1))); i=$((i+1))
  SRC=results/${PRE}_eorecon_ictr_tch0.5_emw0.999_norf_cd0.1_bci_s$S.pt
  Jg=$(sub $NODE CD-gen$S - "OGBENCH_DATA_DIR=$ADATA $BASE LACOT_ENV=$LENV LACOT_K=8 LACOT_TEACHER_MIX=0.5 LACOT_LOAD_CKPT=$SRC LACOT_BOOT_GEN=results/boot_s${S}_dz1.npz LACOT_BOOT_DESERT=1 LACOT_BOOT_Q=512 LACOT_BOOT_M=8 LACOT_BOOT_RMIN=8 LACOT_BOOT_RMAX=25 $APY -u experiments/scratch_lacot_rollout.py")
  Jt=$(sub $NODE CD-dz$S $Jg "OGBENCH_DATA_DIR=$ADATA $TRAIN_BASE LACOT_ENV=$LENV LACOT_K=8 LACOT_SEED=$S LACOT_EMA_W=0.999 LACOT_BOOT_DATA=results/boot_s${S}_dz1.npz LACOT_BOOT_TAG=dzs$S $APY -u experiments/scratch_lacot_rollout.py")
  EV $Jt CD-dz$S-ema results/${PRE}_eorecon_ictr_tch0.5_btdzs${S}_emw0.999_norf_cd0.1_bci_s$S.pt
  echo "CD s$S -> $NODE ($Jg -> $Jt)"
done

echo "== E：seed 拆分 2×2 交叉"
J=$(sub lady E-i14d8 - "OGBENCH_DATA_DIR=$ADATA $TRAIN_BASE LACOT_ENV=$LENV LACOT_K=8 LACOT_SEED=14 LACOT_DATA_SEED=8 LACOT_EMA_W=0.999 $APY -u experiments/scratch_lacot_rollout.py")
EV $J E-i14d8-ema results/${PRE}_eorecon_ictr_tch0.5_emw0.999_dseed8_norf_cd0.1_bci_s14.pt
J=$(sub pocahontas E-i8d14 - "OGBENCH_DATA_DIR=$ADATA $TRAIN_BASE LACOT_ENV=$LENV LACOT_K=8 LACOT_SEED=8 LACOT_DATA_SEED=14 LACOT_EMA_W=0.999 $APY -u experiments/scratch_lacot_rollout.py")
EV $J E-i8d14-ema results/${PRE}_eorecon_ictr_tch0.5_emw0.999_dseed14_norf_cd0.1_bci_s8.pt

echo "== F：好顆前300步逐幀對照 s8/s9（顯式 DATA_SEED=自己、檔名隔離）"
J=$(sub moana F-s8 - "OGBENCH_DATA_DIR=$ADATA $TRAIN_BASE LACOT_ENV=$LENV LACOT_K=8 LACOT_SEED=8 LACOT_DATA_SEED=8 LACOT_EMA_W=0.999 $APY -u experiments/scratch_lacot_rollout.py")
EV $J F-s8-ema results/${PRE}_eorecon_ictr_tch0.5_emw0.999_dseed8_norf_cd0.1_bci_s8.pt
J=$(sub lady F-s9 - "OGBENCH_DATA_DIR=$ADATA $TRAIN_BASE LACOT_ENV=$LENV LACOT_K=8 LACOT_SEED=9 LACOT_DATA_SEED=9 LACOT_EMA_W=0.999 $APY -u experiments/scratch_lacot_rollout.py")
EV $J F-s9-ema results/${PRE}_eorecon_ictr_tch0.5_emw0.999_dseed9_norf_cd0.1_bci_s9.pt

echo "== G：lr 減半對決 s14/15/16 × LR_SCALE=0.5"
i=0
for S in 14 15 16; do
  NODE=$(echo pocahontas lady moana | cut -d" " -f$((i%3+1))); i=$((i+1))
  J=$(sub $NODE G-L$S - "OGBENCH_DATA_DIR=$ADATA $TRAIN_BASE LACOT_ENV=$LENV LACOT_K=8 LACOT_SEED=$S LACOT_EMA_W=0.999 LACOT_LR_SCALE=0.5 $APY -u experiments/scratch_lacot_rollout.py")
  EV $J G-L$S-ema results/${PRE}_eorecon_ictr_tch0.5_emw0.999_lrs0.5_norf_cd0.1_bci_s$S.pt
  echo "G-L$S -> $NODE ($J)"
done

echo "== 全部排入：治療 A3+B1+C2+D1、診斷 E2+F2+G3 ＝ 14 鏈（訓14+gen4+eval14）"
