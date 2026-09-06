#!/bin/bash
# 2026-09-06 深夜過批：ICLR 判決鏈四組（主人已核）
#   ① p=1 對照臂重發 — day_0906_ws8_p1.sh 的 WSP1 段原樣重送。
#      原 job 24780（WSP1-s40-tr）FAILED：assert 0<=drop<1.0 卡在收到 1.0（9秒陣亡，無殘留檔案，
#      results/day_0906/ws_p1{,_zero}/ 已核對是空的）。工作區版主檔（git 未 commit 的 M）已把
#      assert 上界改成 <=1.0（scratch_lacot_rollout.py:186），本批原樣重送即可過關。
#   ② ref 家族擴七顆×2 — f27nL / N5L 從 s40 擴到 s41~47。這兩族沒有既有腳本，env 用
#      sacct -j <job> --format=SubmitLine 逐字撈出（f27nL: job 24453；N5L: job 24472；
#      三族共用的 idpxm 起點 env 用 job 24347 對照，group③ 用）。
#   ③ idpxm 續練終態×8 — s40~47 全程從 SOFT27 重跑 STEPS2=16000（⛔ 不用 CONT_TRAIN，
#      乾淨可比性優先），env 同 idpxm 家族（job 24347）只換 STEPS2。
#   ④ WS 療程加倍×8 — s40~47 從 f27n st8000 續訓 8000 步（既有 ws8_idp0.3 只續 4000），
#      env 同 day_0906_ws8_p1.sh 的 train_ws()，只換 STEPS2=8000。
#
# ⛔ 全域鐵則：
#   - 絕不用 zeldajr GPU（sinfo 確認 State=drained, Reason=Kill task failed）。
#   - 訓練＋eval 全部 lady/moana/pocahontas 三台輪流（jasmine 今晚滿載中，不排它）。
#   - eve_0905_idp_dose.sh／day_0906_ws8_p1.sh 兩支模板原生把 eval 丟 zeldajr —— 本批 eval
#     全面改道：node 從 zeldajr 換成 Turing 三台、env 從 ZPY/ZDATA（本機 venv/data）換成
#     APY/ADATA（/archive venv/data，仿 day_0906_ws8_eval_poca.sh 的組合），並全部加
#     LACOT_INTENT_ARMS=0（主檔工作區帶三腿 patch、判決 eval 不吃加時腿）。
#
# 紀律：DRYRUN=1 先跑一遍，逐條 echo sbatch 全參數自檢（node 輪替／ckpt 全名／OUT_DIR／env
#   有沒有跟撈出來的 SubmitLine 對齊），過了才 DRYRUN=0 真送。
set -euo pipefail
cd ~/Projects/lacot

APY=/archive/cymaxwelllee/LaCoT/.venv/bin/python
ADATA=/archive/cymaxwelllee/data/ogbench
BASE0="MUJOCO_GL=osmesa LACOT_ENC_OBJ=recon_ictr LACOT_LEARNED_REFINE=0 LACOT_BC_INDEP=1"
OFF="LACOT_DEV_EVAL=0 LACOT_EVAL_RS=0 LACOT_EVAL_EPISODES=50"
C2MA="LACOT_SUBGOAL=conf2 LACOT_SUB_POLICY=bc LACOT_GRAD_REFINE=1 LACOT_GRAD_R=0 LACOT_SUB_MAX_ARC=2 LACOT_FINISH_R=2.0"
LENV=pointmaze-large-stitch-v0
SOFT27=results/ckpt_large-stitch_self_K8_c256_ch4_st8000_T128_ep2_gu_eorecon_ictr_tch0.5_emw0.999_wu500_dssoft_norf_cd0.1_bci_s27.pt

OUT_P1=results/day_0906/ws_p1;          OUTZ_P1=results/day_0906/ws_p1_zero
OUT_F27NL=results/day_0905/f27nL;       OUTZ_F27NL=results/day_0905/f27nL_zero
OUT_N5L=results/day_0905/n5L
OUT_IDPXM16=results/day_0906/idpxm16;   OUTZ_IDPXM16=results/day_0906/idpxm16_zero
OUT_WS8K=results/day_0906/ws8k_idp0.3;  OUTZ_WS8K=results/day_0906/ws8k_idp0.3_zero

mkdir -p slurm/logs "$OUT_P1" "$OUTZ_P1" "$OUT_F27NL" "$OUTZ_F27NL" "$OUT_N5L" \
         "$OUT_IDPXM16" "$OUTZ_IDPXM16" "$OUT_WS8K" "$OUTZ_WS8K"

echo "=== 起點 ckpt 存在性檢查 ==="
[ -f "$SOFT27" ] || { echo "⛔ 缺 SOFT27（group②③起點）: $SOFT27" >&2; exit 1; }
echo "✓ SOFT27 在"

f27n_ck() { echo "results/ckpt_large-stitch_self_K8_c256_ch4_st8000_T128_ep2_gu_eorecon_ictr_tch0.5_btf27n_emw0.999_wu500_s1from_ite_dssoft_norf_cd0.1_bci_s$1.pt"; }
for S in 40 41 42 43 44 45 46 47; do
  f=$(f27n_ck $S)
  [ -f "$f" ] || { echo "⛔ 缺 f27n s$S 起點 ckpt（group①④用）: $f" >&2; exit 1; }
done
echo "✓ f27n s40~47（8000 ckpt）8 顆都在"

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

# ⚠️ 兩條獨立輪替計數器 —— TI 只給訓練 job 用、EI 只給 eval job 用。
#    ⛔ 踩過的雷：一開始共用同一個計數器 i、每顆 seed 消耗「train+on+z0」固定 3 個名額，
#    3 跟三台節點數同餘 ⇒ i%3 每輪都繞回同一個相位，訓練永遠落在同一台（DRYRUN 撈出來的：
#    f27nL 七顆全部 lady、IDPXM16／WS8K 十六顆全部 pocahontas）。拆成 TI/EI 兩條獨立計數器，
#    保證同一個角色（訓練 or eval）在自己的迴圈裡確實逐台輪替，不受另一個角色的呼叫次數影響。
#    ⚠️ for 迴圈本身不 fork 子殼，TI/EI 的遞增都寫在本殼直接執行的敘述句裡（不包進 $(...)），
#    sub() 的呼叫可以放心包 $(...)——它只回傳 job id，不碰 TI/EI。
NODES=(lady moana pocahontas); TI=0; EI=0

echo
echo "=== ① p=1 對照臂重發（WSP1，s40，drop=1.0，CONT_TRAIN 續 4000 步）==="
F27N_S40=$(f27n_ck 40)
NODE=${NODES[$((TI%3))]}; TI=$((TI+1))
JOB_P1_TR=$(sub "$NODE" WSP1-s40-tr - \
  "OGBENCH_DATA_DIR=$ADATA $BASE0 LACOT_COND_DROP=0.1 LACOT_ENV=$LENV LACOT_K=8 LACOT_SEED=40 LACOT_TEACHER_MIX=0.5 LACOT_INTENT=embed LACOT_DEC_START=soft LACOT_BOOT_TAG=f27n LACOT_EMA_W=0.999 LACOT_LOAD_CKPT=$F27N_S40 LACOT_CONT_TRAIN=1 LACOT_STEPS2=4000 LACOT_INTENT_DROP=1.0 LACOT_DIV_LOG_EVERY=100 LACOT_EVAL_RS=0 LACOT_EVAL_EPISODES=2 LACOT_OUT_DIR=$OUT_P1 $APY -u experiments/scratch_lacot_rollout.py")
echo "WSP1 s40(drop=1.0) train -> $NODE ($JOB_P1_TR)"

CK_P1=$OUT_P1/ckpt_large-stitch_self_K8_c256_ch4_st0_T128_ep2_gu_eorecon_ictr_tch0.5_btf27n_emw0.999_ite_idp1_ct4000_dssoft_norf_cd0.1_bci_s40.pt
EVB_P1="$BASE0 OGBENCH_DATA_DIR=$ADATA LACOT_COND_DROP=0.1 LACOT_ENV=$LENV LACOT_K=8 LACOT_TEACHER_MIX=0.5 LACOT_LOAD_EMA=1 LACOT_LOAD_CKPT=$CK_P1 LACOT_DEC_START=soft LACOT_INTENT=embed LACOT_BOOT_TAG=f27n LACOT_INTENT_DROP=1.0 LACOT_INTENT_ARMS=0"
NODE=${NODES[$((EI%3))]}; EI=$((EI+1))
JOB_P1_ON=$(sub "$NODE" WSP1-s40-on "$JOB_P1_TR" "$EVB_P1 LACOT_OUT_DIR=$OUT_P1 $OFF $C2MA $APY -u experiments/scratch_lacot_rollout.py")
echo "  eval on -> $NODE ($JOB_P1_ON)"
NODE=${NODES[$((EI%3))]}; EI=$((EI+1))
JOB_P1_Z0=$(sub "$NODE" WSP1-s40-z0 "$JOB_P1_TR" "$EVB_P1 LACOT_INTENT_ZERO=1 LACOT_OUT_DIR=$OUTZ_P1 $OFF $C2MA $APY -u experiments/scratch_lacot_rollout.py")
echo "  eval z0 -> $NODE ($JOB_P1_Z0)"

echo
echo "=== ② ref 家族擴七顆×2（f27nL / N5L，s41~47）==="
echo "--- f27nL（有 intent、無 dropout、STEPS2=11429；env 對 job 24453）---"
for S in 41 42 43 44 45 46 47; do
  NODE=${NODES[$((TI%3))]}; TI=$((TI+1))
  JTR=$(sub "$NODE" f27nL-s$S-tr - \
    "OGBENCH_DATA_DIR=$ADATA $BASE0 LACOT_COND_DROP=0.1 LACOT_STEPS2=11429 LACOT_TEACHER_MIX=0.5 LACOT_EVAL_RS=0 LACOT_EVAL_EPISODES=2 LACOT_DIAG_TRAIN=1 LACOT_ENV=$LENV LACOT_K=8 LACOT_SEED=$S LACOT_EMA_W=0.999 LACOT_WARMUP=500 LACOT_DEC_START=soft LACOT_S1_FROM=$SOFT27 LACOT_INTENT=embed LACOT_BOOT_TAG=f27nL $APY -u experiments/scratch_lacot_rollout.py")
  echo "f27nL s$S train -> $NODE ($JTR)"
  CK=results/ckpt_large-stitch_self_K8_c256_ch4_st11429_T128_ep2_gu_eorecon_ictr_tch0.5_btf27nL_emw0.999_wu500_s1from_ite_dssoft_norf_cd0.1_bci_s$S.pt
  EVB="$BASE0 OGBENCH_DATA_DIR=$ADATA LACOT_COND_DROP=0.1 LACOT_ENV=$LENV LACOT_K=8 LACOT_TEACHER_MIX=0.5 LACOT_LOAD_EMA=1 LACOT_LOAD_CKPT=$CK LACOT_DEC_START=soft LACOT_INTENT=embed LACOT_BOOT_TAG=f27nL LACOT_INTENT_ARMS=0"
  NODE=${NODES[$((EI%3))]}; EI=$((EI+1))
  JON=$(sub "$NODE" f27nL-s$S-on "$JTR" "$EVB LACOT_OUT_DIR=$OUT_F27NL $OFF $C2MA $APY -u experiments/scratch_lacot_rollout.py")
  NODE2=${NODES[$((EI%3))]}; EI=$((EI+1))
  JZ0=$(sub "$NODE2" f27nL-s$S-z0 "$JTR" "$EVB LACOT_INTENT_ZERO=1 LACOT_OUT_DIR=$OUTZ_F27NL $OFF $C2MA $APY -u experiments/scratch_lacot_rollout.py")
  echo "  eval s$S -> on=$JON($NODE) z0=$JZ0($NODE2)"
done

echo "--- N5L（無 intent 基線、無 z0 腿、STEPS2=11429；env 對 job 24472）---"
for S in 41 42 43 44 45 46 47; do
  NODE=${NODES[$((TI%3))]}; TI=$((TI+1))
  JTR=$(sub "$NODE" N5L-s$S-tr - \
    "OGBENCH_DATA_DIR=$ADATA $BASE0 LACOT_COND_DROP=0.1 LACOT_STEPS2=11429 LACOT_TEACHER_MIX=0.5 LACOT_EVAL_RS=0 LACOT_EVAL_EPISODES=2 LACOT_DIAG_TRAIN=1 LACOT_ENV=$LENV LACOT_K=8 LACOT_SEED=$S LACOT_EMA_W=0.999 LACOT_WARMUP=500 LACOT_DEC_START=soft LACOT_S1_FROM=$SOFT27 LACOT_BOOT_TAG=n5L $APY -u experiments/scratch_lacot_rollout.py")
  echo "N5L s$S train -> $NODE ($JTR)"
  CK=results/ckpt_large-stitch_self_K8_c256_ch4_st11429_T128_ep2_gu_eorecon_ictr_tch0.5_btn5L_emw0.999_wu500_s1from_dssoft_norf_cd0.1_bci_s$S.pt
  EVB="$BASE0 OGBENCH_DATA_DIR=$ADATA LACOT_COND_DROP=0.1 LACOT_ENV=$LENV LACOT_K=8 LACOT_TEACHER_MIX=0.5 LACOT_LOAD_EMA=1 LACOT_LOAD_CKPT=$CK LACOT_DEC_START=soft LACOT_BOOT_TAG=n5L LACOT_INTENT_ARMS=0"
  NODE=${NODES[$((EI%3))]}; EI=$((EI+1))
  JEV=$(sub "$NODE" N5L-s$S-ev "$JTR" "$EVB LACOT_OUT_DIR=$OUT_N5L $OFF $C2MA $APY -u experiments/scratch_lacot_rollout.py")
  echo "  eval s$S -> $JEV ($NODE)"
done

echo
echo "=== ③ idpxm 續練終態×8（s40~47，全程重跑 STEPS2=16000，不用 CONT_TRAIN；env 對 job 24347）==="
for S in 40 41 42 43 44 45 46 47; do
  NODE=${NODES[$((TI%3))]}; TI=$((TI+1))
  JTR=$(sub "$NODE" IDPXM16-s$S-tr - \
    "OGBENCH_DATA_DIR=$ADATA $BASE0 LACOT_COND_DROP=0.1 LACOT_STEPS2=16000 LACOT_TEACHER_MIX=0.5 LACOT_EVAL_RS=0 LACOT_EVAL_EPISODES=2 LACOT_DIAG_TRAIN=1 LACOT_ENV=$LENV LACOT_K=8 LACOT_SEED=$S LACOT_EMA_W=0.999 LACOT_WARMUP=500 LACOT_DEC_START=soft LACOT_S1_FROM=$SOFT27 LACOT_INTENT=embed LACOT_INTENT_DROP=0.3 LACOT_BOOT_TAG=idpxm $APY -u experiments/scratch_lacot_rollout.py")
  echo "IDPXM16 s$S train -> $NODE ($JTR)"
  CK=results/ckpt_large-stitch_self_K8_c256_ch4_st16000_T128_ep2_gu_eorecon_ictr_tch0.5_btidpxm_emw0.999_wu500_s1from_ite_idp0.3_dssoft_norf_cd0.1_bci_s$S.pt
  EVB="$BASE0 OGBENCH_DATA_DIR=$ADATA LACOT_COND_DROP=0.1 LACOT_ENV=$LENV LACOT_K=8 LACOT_TEACHER_MIX=0.5 LACOT_LOAD_EMA=1 LACOT_LOAD_CKPT=$CK LACOT_DEC_START=soft LACOT_INTENT=embed LACOT_INTENT_DROP=0.3 LACOT_BOOT_TAG=idpxm LACOT_INTENT_ARMS=0"
  NODE=${NODES[$((EI%3))]}; EI=$((EI+1))
  JON=$(sub "$NODE" IDPXM16-s$S-on "$JTR" "$EVB LACOT_OUT_DIR=$OUT_IDPXM16 $OFF $C2MA $APY -u experiments/scratch_lacot_rollout.py")
  NODE2=${NODES[$((EI%3))]}; EI=$((EI+1))
  JZ0=$(sub "$NODE2" IDPXM16-s$S-z0 "$JTR" "$EVB LACOT_INTENT_ZERO=1 LACOT_OUT_DIR=$OUTZ_IDPXM16 $OFF $C2MA $APY -u experiments/scratch_lacot_rollout.py")
  echo "  eval s$S -> on=$JON($NODE) z0=$JZ0($NODE2)"
done

echo
echo "=== ④ WS 療程加倍×8（s40~47，f27n st8000 起續訓 8000 步，drop=0.3；env 對 day_0906_ws8_p1.sh train_ws()）==="
for S in 40 41 42 43 44 45 46 47; do
  FROM=$(f27n_ck $S)
  NODE=${NODES[$((TI%3))]}; TI=$((TI+1))
  JTR=$(sub "$NODE" WS8K-s$S-tr - \
    "OGBENCH_DATA_DIR=$ADATA $BASE0 LACOT_COND_DROP=0.1 LACOT_ENV=$LENV LACOT_K=8 LACOT_SEED=$S LACOT_TEACHER_MIX=0.5 LACOT_INTENT=embed LACOT_DEC_START=soft LACOT_BOOT_TAG=f27n LACOT_EMA_W=0.999 LACOT_LOAD_CKPT=$FROM LACOT_CONT_TRAIN=1 LACOT_STEPS2=8000 LACOT_INTENT_DROP=0.3 LACOT_DIV_LOG_EVERY=100 LACOT_EVAL_RS=0 LACOT_EVAL_EPISODES=2 LACOT_OUT_DIR=$OUT_WS8K $APY -u experiments/scratch_lacot_rollout.py")
  echo "WS8K s$S train -> $NODE ($JTR)"
  CK=$OUT_WS8K/ckpt_large-stitch_self_K8_c256_ch4_st0_T128_ep2_gu_eorecon_ictr_tch0.5_btf27n_emw0.999_ite_idp0.3_ct8000_dssoft_norf_cd0.1_bci_s$S.pt
  EVB="$BASE0 OGBENCH_DATA_DIR=$ADATA LACOT_COND_DROP=0.1 LACOT_ENV=$LENV LACOT_K=8 LACOT_TEACHER_MIX=0.5 LACOT_LOAD_EMA=1 LACOT_LOAD_CKPT=$CK LACOT_DEC_START=soft LACOT_INTENT=embed LACOT_BOOT_TAG=f27n LACOT_INTENT_DROP=0.3 LACOT_INTENT_ARMS=0"
  NODE=${NODES[$((EI%3))]}; EI=$((EI+1))
  JON=$(sub "$NODE" WS8K-s$S-on "$JTR" "$EVB LACOT_OUT_DIR=$OUT_WS8K $OFF $C2MA $APY -u experiments/scratch_lacot_rollout.py")
  NODE2=${NODES[$((EI%3))]}; EI=$((EI+1))
  JZ0=$(sub "$NODE2" WS8K-s$S-z0 "$JTR" "$EVB LACOT_INTENT_ZERO=1 LACOT_OUT_DIR=$OUTZ_WS8K $OFF $C2MA $APY -u experiments/scratch_lacot_rollout.py")
  echo "  eval s$S -> on=$JON($NODE) z0=$JZ0($NODE2)"
done

echo
echo "=== 送出完畢（DRYRUN=$DRYRUN）：①3 job（1train+2eval）＋②35 job（14train+21eval）＋③24 job（8train+16eval）＋④24 job（8train+16eval）＝86 job ==="
