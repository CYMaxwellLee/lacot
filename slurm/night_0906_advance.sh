#!/bin/bash
# 2026-09-06 深夜追加批：理論推進兩組（主人裁示，經協調者中途換單 —— 原⑥⑦⑧三組作廢未交，
#   換這兩組；本批完全獨立於 24893 起的過夜批＋修復使魔的重交批，不碰、只加）。
#
#   A. GRPO rung1 pilot ×2 — 從 idpxm 家族 s40/s41（st11429，zero 分佈內）續訓 4000 步、開 GRPO。
#      env 照 WS8K 組（sacct 撈 24955 SubmitLine）改 LOAD_CKPT 換成 idpxm 起點、STEPS2=8000→4000、
#      BOOT_TAG f27n→idpxm（反映真實續訓來源）、加 GRPO_W=1 G=8 BQ=4 EVERY=1（其餘 GRPO 參數用
#      主檔預設：WARM=500 STDNORM=1）。AMP 不開（預設本來就是 0，且主檔 assert 擋 AMP×GRPO）。
#      ⚠️ 檔名 grpo 段：已對照【當下】_tag_extra 原樣算過 —— GRPO_W=1 且其餘 GRPO 參數全預設
#      ⇒ 只加 `_grpo1`（G/EVERY/BQ/WARM/STDNORM 都在預設值，不會再加後綴）。
#
#   B. 第二環境 medium-stitch ×8（f27nM ×4 + idpxmM ×4，s40~43）——
#      ⚠️ S1_FROM 跨環境判斷（grep experiments/scratch_lacot_rollout.py 現在的版本得出）：
#        stage 1（traj_enc + e_pooler，line ~641）只吃【正規化後的 2D 軌跡點】
#        （line ~576：traj=(OBS[...]-mu)/sd，mu/sd 是【每次執行】從當次 LACOT_ENV 自己的
#        資料集現算，line 37），佔據圖（_GOCC/_EOCC/_tocc 等）是另外一段、只餵 BFS routing／
#        GRPO reward／teacher 軌跡，不進 traj_enc 的 forward。
#        ⇒ 「stage1 直接讀占據圖」這個假設【沒有】在目前程式碼裡成立（不是硬架構耦合，
#          不會 shape mismatch、也不會 crash）。
#        ⇒ 但這不代表可以放心沿用 SOFT27：正規化只對齊一階/二階矩，₩並不能保證 medium 迷宮
#          在正規化座標下的「路徑形狀」分布（轉彎頻率、走廊長度、牆型）跟 large 一致；
#          load 端（line ~914-930）對 S1_FROM 也【完全沒有】environment 一致性檢查，純粹
#          放行。SOFT27 目前只在同一個 large-stitch 環境內被重複使用過，從沒跨過 maze size。
#        ⇒ 判斷：medium 兩個家族都【不給 S1_FROM／不給 SOFT27】，stage 1 從頭練
#          （STEPS1 用主檔預設 1500，未覆寫）。這是既有、經過驗證的路徑（9/2 SOFT27
#          出現前，所有 large-stitch 家族本來就是這樣練出來的），不是新賭注。
#        ⚠️ 代價：medium 家族訓練時間 = stage1(1500步，從頭) + stage2(8000/11429步)，
#          比同款 STEPS2 的 large 家族（用 S1_FROM 跳過 stage1）多，時間估計見腳本外的回報。
#      f27nM：env 照 f27nL 組（sacct 24896／night_0906_overnight.sh ②段）只換
#        ENV=medium-stitch、STEPS2=8000、BOOT_TAG=f27nM、拿掉 S1_FROM。
#      idpxmM：env 照 IDPXM16 組（sacct 24931／night_0906_overnight.sh ③段）只換
#        ENV=medium-stitch、STEPS2=11429（不是 16000）、BOOT_TAG=idpxmM、拿掉 S1_FROM。
#      ⚠️ 資料先驗過（本次已用 srun 在 jasmine/lady/moana/pocahontas 四台各自的 /archive
#        確認 pointmaze-medium-stitch-v0.npz + -val.npz 都在，四台都有，不是只有訓練目標台）。
#
# ⛔ 全域鐵則（跟過夜批一致）：
#   - 絕不用 zeldajr／bocchi／alice。
#   - 訓練 jasmine/lady/moana/pocahontas 四台輪流；eval 只在 Turing 三台
#     （lady/moana/pocahontas）輪流。TI／EI 兩條獨立計數器（過夜批踩過共用計數器導致
#     同相位鎖死同一台的雷，這裡直接照抄拆開的寫法）。
#   - eval 全部帶 LACOT_INTENT_ARMS=0（本批沒有三腿組，全部標準 eval）。
#   - 完全不動 24893 起的過夜批／修復使魔重交批的任何 job，只【新增】。
#
# ⚠️ 已知風險（誠實揭露，不是自己能修的範圍）：main 檔 experiments/scratch_lacot_rollout.py
#   現在是【工作區未 commit、修復使魔正在改】的活狀態（git diff 536+/94- 行；本次驗證過程中
#   親眼看到同一個 grep pattern 兩次呼叫之間行號整段偏移 ~187 行）。已對照 diff hunk 位置
#   確認這波在動的是【新功能】：推論期 BoN、訓練側 LACOT_AMP/LACOT_COMPILE 加速
#   （兩個都預設關、本批完全沒用到），GRPO 與 S1_FROM 那兩段是這波 diff 之前就存在、
#   不在这次改動範圍內。但 job 真正執行是排到才跑（現在佇列很滿），屆時 repo 版本可能又
#   跟現在不同 —— 跟過夜批／修復批共用同一個曝險，不是本批獨有。若 eval 端因為檔名段
#   算法又變而找不到 ckpt，會是【吵的】failure（file not found），不是靜默錯誤。
#
# 紀律：DRYRUN=1 先跑一遍，逐條 echo sbatch 全參數自檢，過了才 DRYRUN=0 真送。
set -euo pipefail
cd ~/Projects/lacot

APY=/archive/cymaxwelllee/LaCoT/.venv/bin/python
ADATA=/archive/cymaxwelllee/data/ogbench
BASE0="MUJOCO_GL=osmesa LACOT_ENC_OBJ=recon_ictr LACOT_LEARNED_REFINE=0 LACOT_BC_INDEP=1"
OFF="LACOT_DEV_EVAL=0 LACOT_EVAL_RS=0 LACOT_EVAL_EPISODES=50"
C2MA="LACOT_SUBGOAL=conf2 LACOT_SUB_POLICY=bc LACOT_GRAD_REFINE=1 LACOT_GRAD_R=0 LACOT_SUB_MAX_ARC=2 LACOT_FINISH_R=2.0"
LENV=pointmaze-large-stitch-v0
MENV=pointmaze-medium-stitch-v0

OUT_GRPO1=results/day_0906/grpo1;   OUTZ_GRPO1=results/day_0906/grpo1_zero
OUT_F27NM=results/day_0906/f27nM;   OUTZ_F27NM=results/day_0906/f27nM_zero
OUT_IDPXMM=results/day_0906/idpxmM; OUTZ_IDPXMM=results/day_0906/idpxmM_zero

mkdir -p slurm/logs "$OUT_GRPO1" "$OUTZ_GRPO1" "$OUT_F27NM" "$OUTZ_F27NM" "$OUT_IDPXMM" "$OUTZ_IDPXMM"

echo "=== 起點 ckpt 存在性檢查（A 組）==="
idpxm_ck() { echo "results/ckpt_large-stitch_self_K8_c256_ch4_st11429_T128_ep2_gu_eorecon_ictr_tch0.5_btidpxm_emw0.999_wu500_s1from_ite_idp0.3_dssoft_norf_cd0.1_bci_s$1.pt"; }
for S in 40 41; do
  f=$(idpxm_ck $S)
  [ -f "$f" ] || { echo "⛔ 缺 idpxm s$S 起點 ckpt（A 組用）: $f" >&2; exit 1; }
done
echo "✓ idpxm s40/s41（st11429）起點都在"

echo "=== B 組檔名段互蓋自檢 ==="
echo "  f27nM 用新 BOOT_TAG=f27nM + ENV 前綴 medium-stitch（既有檔全是 large-stitch）⇒ 不可能撞現有檔"
echo "  idpxmM 用新 BOOT_TAG=idpxmM，同理 ⇒ 不可能撞現有 btidpxm（無 M）家族"
[ -d results/day_0906/grpo1 ] && [ -z "$(ls -A results/day_0906/grpo1 2>/dev/null)" ] || echo "  （grpo1 目錄剛建或已空，OK）"

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

# ⚠️ TI／EI 兩條獨立輪替計數器（過夜批的教訓：共用同一條計數器、角色數跟節點數同餘會
#    造成同一角色永遠落在同一台）。TNODES 訓練 4 台、ENODES eval 只 Turing 3 台。
TNODES=(jasmine lady moana pocahontas); ENODES=(lady moana pocahontas); TI=0; EI=0

echo
echo "=== A. GRPO rung1 pilot ×2（idpxm s40/s41 st11429 起，CONT_TRAIN 4000 步，開 GRPO）==="
for S in 40 41; do
  FROM=$(idpxm_ck $S)
  NODE=${TNODES[$((TI%4))]}; TI=$((TI+1))
  JTR=$(sub "$NODE" GRPO1-s$S-tr - \
    "OGBENCH_DATA_DIR=$ADATA $BASE0 LACOT_COND_DROP=0.1 LACOT_ENV=$LENV LACOT_K=8 LACOT_SEED=$S LACOT_TEACHER_MIX=0.5 LACOT_INTENT=embed LACOT_DEC_START=soft LACOT_BOOT_TAG=idpxm LACOT_EMA_W=0.999 LACOT_LOAD_CKPT=$FROM LACOT_CONT_TRAIN=1 LACOT_STEPS2=4000 LACOT_INTENT_DROP=0.3 LACOT_GRPO_W=1 LACOT_GRPO_G=8 LACOT_GRPO_BQ=4 LACOT_GRPO_EVERY=1 LACOT_DIV_LOG_EVERY=100 LACOT_EVAL_RS=0 LACOT_EVAL_EPISODES=2 LACOT_OUT_DIR=$OUT_GRPO1 $APY -u experiments/scratch_lacot_rollout.py")
  echo "GRPO1 s$S train -> $NODE ($JTR)"
  CK=$OUT_GRPO1/ckpt_large-stitch_self_K8_c256_ch4_st0_T128_ep2_gu_eorecon_ictr_tch0.5_btidpxm_emw0.999_ite_idp0.3_ct4000_grpo1_dssoft_norf_cd0.1_bci_s$S.pt
  EVB="$BASE0 OGBENCH_DATA_DIR=$ADATA LACOT_COND_DROP=0.1 LACOT_ENV=$LENV LACOT_K=8 LACOT_TEACHER_MIX=0.5 LACOT_LOAD_EMA=1 LACOT_LOAD_CKPT=$CK LACOT_DEC_START=soft LACOT_INTENT=embed LACOT_BOOT_TAG=idpxm LACOT_INTENT_DROP=0.3 LACOT_INTENT_ARMS=0"
  NODE=${ENODES[$((EI%3))]}; EI=$((EI+1))
  JON=$(sub "$NODE" GRPO1-s$S-on "$JTR" "$EVB LACOT_OUT_DIR=$OUT_GRPO1 $OFF $C2MA $APY -u experiments/scratch_lacot_rollout.py")
  NODE2=${ENODES[$((EI%3))]}; EI=$((EI+1))
  JZ0=$(sub "$NODE2" GRPO1-s$S-z0 "$JTR" "$EVB LACOT_INTENT_ZERO=1 LACOT_OUT_DIR=$OUTZ_GRPO1 $OFF $C2MA $APY -u experiments/scratch_lacot_rollout.py")
  echo "  eval s$S -> on=$JON($NODE) z0=$JZ0($NODE2)"
done

echo
echo "=== B. 第二環境 medium ×8（f27nM ×4 + idpxmM ×4，stage1 從頭練、不給 S1_FROM）==="
echo "--- f27nM（無 intent dropout，STEPS2=8000，s40~43；env 對 f27nL/job 24896，拿掉 S1_FROM）---"
for S in 40 41 42 43; do
  NODE=${TNODES[$((TI%4))]}; TI=$((TI+1))
  JTR=$(sub "$NODE" f27nM-s$S-tr - \
    "OGBENCH_DATA_DIR=$ADATA $BASE0 LACOT_COND_DROP=0.1 LACOT_STEPS2=8000 LACOT_TEACHER_MIX=0.5 LACOT_EVAL_RS=0 LACOT_EVAL_EPISODES=2 LACOT_DIAG_TRAIN=1 LACOT_ENV=$MENV LACOT_K=8 LACOT_SEED=$S LACOT_EMA_W=0.999 LACOT_WARMUP=500 LACOT_DEC_START=soft LACOT_INTENT=embed LACOT_BOOT_TAG=f27nM $APY -u experiments/scratch_lacot_rollout.py")
  echo "f27nM s$S train -> $NODE ($JTR)"
  CK=results/ckpt_medium-stitch_self_K8_c256_ch4_st8000_T128_ep2_gu_eorecon_ictr_tch0.5_btf27nM_emw0.999_wu500_ite_dssoft_norf_cd0.1_bci_s$S.pt
  EVB="$BASE0 OGBENCH_DATA_DIR=$ADATA LACOT_COND_DROP=0.1 LACOT_ENV=$MENV LACOT_K=8 LACOT_TEACHER_MIX=0.5 LACOT_LOAD_EMA=1 LACOT_LOAD_CKPT=$CK LACOT_DEC_START=soft LACOT_INTENT=embed LACOT_BOOT_TAG=f27nM LACOT_INTENT_ARMS=0"
  NODE=${ENODES[$((EI%3))]}; EI=$((EI+1))
  JON=$(sub "$NODE" f27nM-s$S-on "$JTR" "$EVB LACOT_OUT_DIR=$OUT_F27NM $OFF $C2MA $APY -u experiments/scratch_lacot_rollout.py")
  NODE2=${ENODES[$((EI%3))]}; EI=$((EI+1))
  JZ0=$(sub "$NODE2" f27nM-s$S-z0 "$JTR" "$EVB LACOT_INTENT_ZERO=1 LACOT_OUT_DIR=$OUTZ_F27NM $OFF $C2MA $APY -u experiments/scratch_lacot_rollout.py")
  echo "  eval s$S -> on=$JON($NODE) z0=$JZ0($NODE2)"
done

echo "--- idpxmM（intent dropout=0.3，STEPS2=11429，s40~43；env 對 IDPXM16/job 24931，拿掉 S1_FROM）---"
for S in 40 41 42 43; do
  NODE=${TNODES[$((TI%4))]}; TI=$((TI+1))
  JTR=$(sub "$NODE" idpxmM-s$S-tr - \
    "OGBENCH_DATA_DIR=$ADATA $BASE0 LACOT_COND_DROP=0.1 LACOT_STEPS2=11429 LACOT_TEACHER_MIX=0.5 LACOT_EVAL_RS=0 LACOT_EVAL_EPISODES=2 LACOT_DIAG_TRAIN=1 LACOT_ENV=$MENV LACOT_K=8 LACOT_SEED=$S LACOT_EMA_W=0.999 LACOT_WARMUP=500 LACOT_DEC_START=soft LACOT_INTENT=embed LACOT_INTENT_DROP=0.3 LACOT_BOOT_TAG=idpxmM $APY -u experiments/scratch_lacot_rollout.py")
  echo "idpxmM s$S train -> $NODE ($JTR)"
  CK=results/ckpt_medium-stitch_self_K8_c256_ch4_st11429_T128_ep2_gu_eorecon_ictr_tch0.5_btidpxmM_emw0.999_wu500_ite_idp0.3_dssoft_norf_cd0.1_bci_s$S.pt
  EVB="$BASE0 OGBENCH_DATA_DIR=$ADATA LACOT_COND_DROP=0.1 LACOT_ENV=$MENV LACOT_K=8 LACOT_TEACHER_MIX=0.5 LACOT_LOAD_EMA=1 LACOT_LOAD_CKPT=$CK LACOT_DEC_START=soft LACOT_INTENT=embed LACOT_INTENT_DROP=0.3 LACOT_BOOT_TAG=idpxmM LACOT_INTENT_ARMS=0"
  NODE=${ENODES[$((EI%3))]}; EI=$((EI+1))
  JON=$(sub "$NODE" idpxmM-s$S-on "$JTR" "$EVB LACOT_OUT_DIR=$OUT_IDPXMM $OFF $C2MA $APY -u experiments/scratch_lacot_rollout.py")
  NODE2=${ENODES[$((EI%3))]}; EI=$((EI+1))
  JZ0=$(sub "$NODE2" idpxmM-s$S-z0 "$JTR" "$EVB LACOT_INTENT_ZERO=1 LACOT_OUT_DIR=$OUTZ_IDPXMM $OFF $C2MA $APY -u experiments/scratch_lacot_rollout.py")
  echo "  eval s$S -> on=$JON($NODE) z0=$JZ0($NODE2)"
done

echo
echo "=== 送出完畢（DRYRUN=$DRYRUN）：A 6 job（2train+4eval）＋B 24 job（8train+16eval）＝30 job ==="
