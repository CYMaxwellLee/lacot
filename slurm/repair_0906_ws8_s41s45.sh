#!/bin/bash
# 9/6 深夜（zeldajr reboot 後）：WS8 p=0.3 s41/s45 修復重交
# 背景：原訓練 24759(s41@alice)/24771(s45@quinn) 落在沒有 /archive venv 的節點，18:26 exit127 陣亡；
#   修復批 24990-25060 蓋了 f27nL/N5L/IDPXM16/WS8K 家族、獨漏 WS8 這兩顆（同分鐘陣亡、家族 grep 漏網）。
#   綁死它們的 eval 24884/24885/24886/24887＝DependencyNeverSatisfied（afterok 指著 failed job、永不跑），
#   由本腳本 scancel 後掛新訓練重交。八顆判決（day_0906_ws8_p1.sh 檔頭判讀①）缺這兩顆補不齊。
# 形：訓練＝day_0906_ws8_p1.sh train_ws 原 env 逐字（只換節點綁定 lady/moana）；
#   eval＝day_0906_ws8_eval_poca.sh ep 原 env 逐字（pocahontas、WSP-s{S}-{on,z0}、INTENT_ARMS=0）。
# 紀律：先 DRYRUN=1 跑過自檢每條 sbatch，再無 DRYRUN 真送。
set -euo pipefail
cd ~/Projects/lacot
APY=/archive/cymaxwelllee/LaCoT/.venv/bin/python
ADATA=/archive/cymaxwelllee/data/ogbench
BASE0="MUJOCO_GL=osmesa LACOT_ENC_OBJ=recon_ictr LACOT_LEARNED_REFINE=0 LACOT_BC_INDEP=1"
OFF="LACOT_DEV_EVAL=0 LACOT_EVAL_RS=0 LACOT_EVAL_EPISODES=50"
C2MA="LACOT_SUBGOAL=conf2 LACOT_SUB_POLICY=bc LACOT_GRAD_REFINE=1 LACOT_GRAD_R=0 LACOT_SUB_MAX_ARC=2 LACOT_FINISH_R=2.0"
LENV=pointmaze-large-stitch-v0
OUT03=results/day_0906/ws8_idp0.3
OUTZ03=results/day_0906/ws8_idp0.3_zero
mkdir -p slurm/logs "$OUT03" "$OUTZ03"
DRYRUN=${DRYRUN:-0}

sub() { local node=$1 name=$2 deps=$3; shift 3; local depflag=""
  [ "$deps" != "-" ] && depflag="--dependency=afterok:$deps"
  if [ "$DRYRUN" = "1" ]; then
    echo "[DRYRUN] sbatch -p admin -A it -q great-mage --time=24:00:00 --nodelist=$node --gres=gpu:1 --job-name=$name -o slurm/logs/%x-%j.out $depflag --wrap \"cd ~/Projects/lacot && env $*\"" >&2
    echo "DRY-$name"; return 0
  fi
  sbatch -p admin -A it -q great-mage --time=24:00:00 --nodelist=$node --gres=gpu:1 \
    --job-name=$name -o slurm/logs/%x-%j.out $depflag \
    --wrap "cd ~/Projects/lacot && env $*" | awk '{print $4}'; }

f27n_ck() { echo "results/ckpt_large-stitch_self_K8_c256_ch4_st8000_T128_ep2_gu_eorecon_ictr_tch0.5_btf27n_emw0.999_wu500_s1from_ite_dssoft_norf_cd0.1_bci_s$1.pt"; }
CKPRE="ckpt_large-stitch_self_K8_c256_ch4_st0_T128_ep2_gu_eorecon_ictr_tch0.5_btf27n_emw0.999_ite_idp0.3_ct4000_dssoft_norf_cd0.1_bci"

echo "=== 收掉死依賴 eval（24884/24885/24886/24887，DependencyNeverSatisfied）==="
if [ "$DRYRUN" = "1" ]; then echo "[DRYRUN] scancel 24884 24885 24886 24887" >&2; else scancel 24884 24885 24886 24887; fi

for S in 41 45; do
  FROM=$(f27n_ck $S)
  [ -f "$FROM" ] || { echo "⛔ 缺 s$S 起點 ckpt: $FROM" >&2; exit 1; }
  case $S in 41) NODE=lady;; 45) NODE=moana;; esac
  TJ=$(sub "$NODE" WS8-s$S-tr - \
    "OGBENCH_DATA_DIR=$ADATA $BASE0 LACOT_COND_DROP=0.1 LACOT_ENV=$LENV LACOT_K=8 LACOT_SEED=$S LACOT_TEACHER_MIX=0.5 LACOT_INTENT=embed LACOT_DEC_START=soft LACOT_BOOT_TAG=f27n LACOT_EMA_W=0.999 LACOT_LOAD_CKPT=$FROM LACOT_CONT_TRAIN=1 LACOT_STEPS2=4000 LACOT_INTENT_DROP=0.3 LACOT_DIV_LOG_EVERY=100 LACOT_EVAL_RS=0 LACOT_EVAL_EPISODES=2 LACOT_OUT_DIR=$OUT03 $APY -u experiments/scratch_lacot_rollout.py")
  echo "WS8 s$S 續訓 -> $NODE (train job=$TJ)"
  CK=$OUT03/${CKPRE}_s$S.pt
  EVB="$BASE0 OGBENCH_DATA_DIR=$ADATA LACOT_COND_DROP=0.1 LACOT_ENV=$LENV LACOT_K=8 LACOT_TEACHER_MIX=0.5 LACOT_LOAD_EMA=1 LACOT_LOAD_CKPT=$CK LACOT_DEC_START=soft LACOT_INTENT=embed LACOT_BOOT_TAG=f27n LACOT_INTENT_DROP=0.3 LACOT_INTENT_ARMS=0"
  JON=$(sub pocahontas WSP-s$S-on "$TJ" "$EVB LACOT_OUT_DIR=$OUT03 $OFF $C2MA $APY -u experiments/scratch_lacot_rollout.py")
  JZ=$(sub pocahontas WSP-s$S-z0 "$TJ" "$EVB LACOT_INTENT_ZERO=1 LACOT_OUT_DIR=$OUTZ03 $OFF $C2MA $APY -u experiments/scratch_lacot_rollout.py")
  echo "  eval WSP s$S -> pocahontas on=$JON z0=$JZ (afterok:$TJ)"
done
echo "=== 送出完畢（DRYRUN=$DRYRUN）==="
