#!/bin/bash
# 2026-09-06 深夜過批 v2 —— night_0906_overnight.sh 死亡 job 搶修重交
#
# 事故經過：對 pending 訓練 job 誤跑 `scontrol update ReqNodeList=`（想清空節點綁定讓
# slurm 自由排），結果 slurm 把這些 pending 訓練直接判 FAILED（sacct Reason 多半留下
# 舊值 ReqNodeNotAvail，NodeList 欄位是 quinn/rapunzel/snowwhite/alice 這些從沒排過本批
# 任何 job 的節點 —— 判斷是 slurm 內部重新評估排程時的殘影，不是 job payload 真的跑過去）。
# 依賴它們的 eval（afterok 綁 tr job id）在盤點當下多半還顯示 PENDING（Reason 陸續從
# None 轉成 Dependency，尚未被 slurm 掃到判 DependencyNeverSatisfied），但依賴的訓練
# 已經 FAILED、這些 eval 注定跑不了，一併當死亡處理：先 scancel 掉舊的殘影 job，
# 新訓練 job 出爐後用新 job id 重接 afterok。
#
# 節點政策（主人 09-06 深夜兩次裁定，以後者為準）：
#   ⛔ 全程不用 zeldajr（drain）、不用 bocchi（Pro 6000，主人未授權）、不用 alice。
#   ✅ 訓練：原本四台輪流 —— jasmine／lady／moana／pocahontas（jasmine 主人明確說可以用，
#      它忙就讓 job 排隊沒關係，不用刻意閃開）。
#   ✅ eval：維持 Turing 三台 —— lady／moana／pocahontas（不含 jasmine）。
#   都是「明確 --nodelist 綁定＋輪流」，不是 --exclude 讓 slurm 自由排。
#
# 只重交死亡的：
#   - f27nL：s42／s45／s46（其餘 s41/43/44/47 訓練還 RUNNING，活的別碰）
#   - N5L：s41~47 全部（7 顆訓練全 FAILED，包含 sacct 顯示 Reason=None 的 s43／job 24921）
#   - IDPXM16：s40~47 全部（8 顆訓練全 FAILED）
#   - WS8K：s40~47 全部（8 顆訓練全 FAILED）
#   - group①（WSP1-s40，job 24893/24894/24895）全部活著，本檔不碰，不出現在下面
#
# 紀律：DRYRUN=1 先跑一遍自檢（node 是否只落在許可清單、seed 清單對不對、ckpt 路徑對不對），
#   過了才 DRYRUN=0 真送。
set -euo pipefail
cd ~/Projects/lacot

APY=/archive/cymaxwelllee/LaCoT/.venv/bin/python
ADATA=/archive/cymaxwelllee/data/ogbench
BASE0="MUJOCO_GL=osmesa LACOT_ENC_OBJ=recon_ictr LACOT_LEARNED_REFINE=0 LACOT_BC_INDEP=1"
OFF="LACOT_DEV_EVAL=0 LACOT_EVAL_RS=0 LACOT_EVAL_EPISODES=50"
C2MA="LACOT_SUBGOAL=conf2 LACOT_SUB_POLICY=bc LACOT_GRAD_REFINE=1 LACOT_GRAD_R=0 LACOT_SUB_MAX_ARC=2 LACOT_FINISH_R=2.0"
LENV=pointmaze-large-stitch-v0
SOFT27=results/ckpt_large-stitch_self_K8_c256_ch4_st8000_T128_ep2_gu_eorecon_ictr_tch0.5_emw0.999_wu500_dssoft_norf_cd0.1_bci_s27.pt

OUT_F27NL=results/day_0905/f27nL;       OUTZ_F27NL=results/day_0905/f27nL_zero
OUT_N5L=results/day_0905/n5L
OUT_IDPXM16=results/day_0906/idpxm16;   OUTZ_IDPXM16=results/day_0906/idpxm16_zero
OUT_WS8K=results/day_0906/ws8k_idp0.3;  OUTZ_WS8K=results/day_0906/ws8k_idp0.3_zero

mkdir -p slurm/logs "$OUT_F27NL" "$OUTZ_F27NL" "$OUT_N5L" \
         "$OUT_IDPXM16" "$OUTZ_IDPXM16" "$OUT_WS8K" "$OUTZ_WS8K"

echo "=== 起點 ckpt 存在性檢查 ==="
[ -f "$SOFT27" ] || { echo "⛔ 缺 SOFT27（group②③起點）: $SOFT27" >&2; exit 1; }
echo "✓ SOFT27 在"

f27n_ck() { echo "results/ckpt_large-stitch_self_K8_c256_ch4_st8000_T128_ep2_gu_eorecon_ictr_tch0.5_btf27n_emw0.999_wu500_s1from_ite_dssoft_norf_cd0.1_bci_s$1.pt"; }
for S in 40 41 42 43 44 45 46 47; do
  f=$(f27n_ck $S)
  [ -f "$f" ] || { echo "⛔ 缺 f27n s$S 起點 ckpt（group④用）: $f" >&2; exit 1; }
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

# TI：訓練用，4 台輪（jasmine／lady／moana／pocahontas）。EI：eval 用，3 台輪（Turing：
# lady／moana／pocahontas，不含 jasmine）。兩條計數器互不干擾（原始批次踩過共用計數器
# 跟節點數同餘、永遠繞回同一相位的雷，這裡延續分開的作法）。
NODES_TR=(jasmine lady moana pocahontas); TI=0
NODES_EV=(lady moana pocahontas); EI=0

echo
echo "=== ② ref 家族重交 —— f27nL 死亡 seed（s42／s45／s46；s41/43/44/47 訓練活的不動）==="
for S in 42 45 46; do
  NODE=${NODES_TR[$((TI%4))]}; TI=$((TI+1))
  JTR=$(sub "$NODE" f27nL-s$S-tr - \
    "OGBENCH_DATA_DIR=$ADATA $BASE0 LACOT_COND_DROP=0.1 LACOT_STEPS2=11429 LACOT_TEACHER_MIX=0.5 LACOT_EVAL_RS=0 LACOT_EVAL_EPISODES=2 LACOT_DIAG_TRAIN=1 LACOT_ENV=$LENV LACOT_K=8 LACOT_SEED=$S LACOT_EMA_W=0.999 LACOT_WARMUP=500 LACOT_DEC_START=soft LACOT_S1_FROM=$SOFT27 LACOT_INTENT=embed LACOT_BOOT_TAG=f27nL $APY -u experiments/scratch_lacot_rollout.py")
  echo "f27nL s$S train -> $NODE ($JTR)"
  CK=results/ckpt_large-stitch_self_K8_c256_ch4_st11429_T128_ep2_gu_eorecon_ictr_tch0.5_btf27nL_emw0.999_wu500_s1from_ite_dssoft_norf_cd0.1_bci_s$S.pt
  EVB="$BASE0 OGBENCH_DATA_DIR=$ADATA LACOT_COND_DROP=0.1 LACOT_ENV=$LENV LACOT_K=8 LACOT_TEACHER_MIX=0.5 LACOT_LOAD_EMA=1 LACOT_LOAD_CKPT=$CK LACOT_DEC_START=soft LACOT_INTENT=embed LACOT_BOOT_TAG=f27nL LACOT_INTENT_ARMS=0"
  NODE=${NODES_EV[$((EI%3))]}; EI=$((EI+1))
  JON=$(sub "$NODE" f27nL-s$S-on "$JTR" "$EVB LACOT_OUT_DIR=$OUT_F27NL $OFF $C2MA $APY -u experiments/scratch_lacot_rollout.py")
  NODE2=${NODES_EV[$((EI%3))]}; EI=$((EI+1))
  JZ0=$(sub "$NODE2" f27nL-s$S-z0 "$JTR" "$EVB LACOT_INTENT_ZERO=1 LACOT_OUT_DIR=$OUTZ_F27NL $OFF $C2MA $APY -u experiments/scratch_lacot_rollout.py")
  echo "  eval s$S -> on=$JON($NODE) z0=$JZ0($NODE2)"
done

echo "--- N5L（無 intent 基線、無 z0 腿；s41~47 全部重交）---"
for S in 41 42 43 44 45 46 47; do
  NODE=${NODES_TR[$((TI%4))]}; TI=$((TI+1))
  JTR=$(sub "$NODE" N5L-s$S-tr - \
    "OGBENCH_DATA_DIR=$ADATA $BASE0 LACOT_COND_DROP=0.1 LACOT_STEPS2=11429 LACOT_TEACHER_MIX=0.5 LACOT_EVAL_RS=0 LACOT_EVAL_EPISODES=2 LACOT_DIAG_TRAIN=1 LACOT_ENV=$LENV LACOT_K=8 LACOT_SEED=$S LACOT_EMA_W=0.999 LACOT_WARMUP=500 LACOT_DEC_START=soft LACOT_S1_FROM=$SOFT27 LACOT_BOOT_TAG=n5L $APY -u experiments/scratch_lacot_rollout.py")
  echo "N5L s$S train -> $NODE ($JTR)"
  CK=results/ckpt_large-stitch_self_K8_c256_ch4_st11429_T128_ep2_gu_eorecon_ictr_tch0.5_btn5L_emw0.999_wu500_s1from_dssoft_norf_cd0.1_bci_s$S.pt
  EVB="$BASE0 OGBENCH_DATA_DIR=$ADATA LACOT_COND_DROP=0.1 LACOT_ENV=$LENV LACOT_K=8 LACOT_TEACHER_MIX=0.5 LACOT_LOAD_EMA=1 LACOT_LOAD_CKPT=$CK LACOT_DEC_START=soft LACOT_BOOT_TAG=n5L LACOT_INTENT_ARMS=0"
  NODE=${NODES_EV[$((EI%3))]}; EI=$((EI+1))
  JEV=$(sub "$NODE" N5L-s$S-ev "$JTR" "$EVB LACOT_OUT_DIR=$OUT_N5L $OFF $C2MA $APY -u experiments/scratch_lacot_rollout.py")
  echo "  eval s$S -> $JEV ($NODE)"
done

echo
echo "=== ③ idpxm 續練終態×8 重交 —— s40~47 全部（8 顆訓練全 FAILED）==="
for S in 40 41 42 43 44 45 46 47; do
  NODE=${NODES_TR[$((TI%4))]}; TI=$((TI+1))
  JTR=$(sub "$NODE" IDPXM16-s$S-tr - \
    "OGBENCH_DATA_DIR=$ADATA $BASE0 LACOT_COND_DROP=0.1 LACOT_STEPS2=16000 LACOT_TEACHER_MIX=0.5 LACOT_EVAL_RS=0 LACOT_EVAL_EPISODES=2 LACOT_DIAG_TRAIN=1 LACOT_ENV=$LENV LACOT_K=8 LACOT_SEED=$S LACOT_EMA_W=0.999 LACOT_WARMUP=500 LACOT_DEC_START=soft LACOT_S1_FROM=$SOFT27 LACOT_INTENT=embed LACOT_INTENT_DROP=0.3 LACOT_BOOT_TAG=idpxm $APY -u experiments/scratch_lacot_rollout.py")
  echo "IDPXM16 s$S train -> $NODE ($JTR)"
  CK=results/ckpt_large-stitch_self_K8_c256_ch4_st16000_T128_ep2_gu_eorecon_ictr_tch0.5_btidpxm_emw0.999_wu500_s1from_ite_idp0.3_dssoft_norf_cd0.1_bci_s$S.pt
  EVB="$BASE0 OGBENCH_DATA_DIR=$ADATA LACOT_COND_DROP=0.1 LACOT_ENV=$LENV LACOT_K=8 LACOT_TEACHER_MIX=0.5 LACOT_LOAD_EMA=1 LACOT_LOAD_CKPT=$CK LACOT_DEC_START=soft LACOT_INTENT=embed LACOT_INTENT_DROP=0.3 LACOT_BOOT_TAG=idpxm LACOT_INTENT_ARMS=0"
  NODE=${NODES_EV[$((EI%3))]}; EI=$((EI+1))
  JON=$(sub "$NODE" IDPXM16-s$S-on "$JTR" "$EVB LACOT_OUT_DIR=$OUT_IDPXM16 $OFF $C2MA $APY -u experiments/scratch_lacot_rollout.py")
  NODE2=${NODES_EV[$((EI%3))]}; EI=$((EI+1))
  JZ0=$(sub "$NODE2" IDPXM16-s$S-z0 "$JTR" "$EVB LACOT_INTENT_ZERO=1 LACOT_OUT_DIR=$OUTZ_IDPXM16 $OFF $C2MA $APY -u experiments/scratch_lacot_rollout.py")
  echo "  eval s$S -> on=$JON($NODE) z0=$JZ0($NODE2)"
done

echo
echo "=== ④ WS 療程加倍×8 重交 —— s40~47 全部（8 顆訓練全 FAILED）==="
for S in 40 41 42 43 44 45 46 47; do
  FROM=$(f27n_ck $S)
  NODE=${NODES_TR[$((TI%4))]}; TI=$((TI+1))
  JTR=$(sub "$NODE" WS8K-s$S-tr - \
    "OGBENCH_DATA_DIR=$ADATA $BASE0 LACOT_COND_DROP=0.1 LACOT_ENV=$LENV LACOT_K=8 LACOT_SEED=$S LACOT_TEACHER_MIX=0.5 LACOT_INTENT=embed LACOT_DEC_START=soft LACOT_BOOT_TAG=f27n LACOT_EMA_W=0.999 LACOT_LOAD_CKPT=$FROM LACOT_CONT_TRAIN=1 LACOT_STEPS2=8000 LACOT_INTENT_DROP=0.3 LACOT_DIV_LOG_EVERY=100 LACOT_EVAL_RS=0 LACOT_EVAL_EPISODES=2 LACOT_OUT_DIR=$OUT_WS8K $APY -u experiments/scratch_lacot_rollout.py")
  echo "WS8K s$S train -> $NODE ($JTR)"
  CK=$OUT_WS8K/ckpt_large-stitch_self_K8_c256_ch4_st0_T128_ep2_gu_eorecon_ictr_tch0.5_btf27n_emw0.999_ite_idp0.3_ct8000_dssoft_norf_cd0.1_bci_s$S.pt
  EVB="$BASE0 OGBENCH_DATA_DIR=$ADATA LACOT_COND_DROP=0.1 LACOT_ENV=$LENV LACOT_K=8 LACOT_TEACHER_MIX=0.5 LACOT_LOAD_EMA=1 LACOT_LOAD_CKPT=$CK LACOT_DEC_START=soft LACOT_INTENT=embed LACOT_BOOT_TAG=f27n LACOT_INTENT_DROP=0.3 LACOT_INTENT_ARMS=0"
  NODE=${NODES_EV[$((EI%3))]}; EI=$((EI+1))
  JON=$(sub "$NODE" WS8K-s$S-on "$JTR" "$EVB LACOT_OUT_DIR=$OUT_WS8K $OFF $C2MA $APY -u experiments/scratch_lacot_rollout.py")
  NODE2=${NODES_EV[$((EI%3))]}; EI=$((EI+1))
  JZ0=$(sub "$NODE2" WS8K-s$S-z0 "$JTR" "$EVB LACOT_INTENT_ZERO=1 LACOT_OUT_DIR=$OUTZ_WS8K $OFF $C2MA $APY -u experiments/scratch_lacot_rollout.py")
  echo "  eval s$S -> on=$JON($NODE) z0=$JZ0($NODE2)"
done

echo
echo "=== 重交完畢（DRYRUN=$DRYRUN）：②f27nL 3train+6eval ＋ N5L 7train+7eval ＋ ③IDPXM16 8train+16eval ＋ ④WS8K 8train+16eval ＝ 26train+45eval=71 job ==="
