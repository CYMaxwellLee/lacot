#!/bin/bash
# 2026-09-06 加速 sanity（Patch A 驗完後提交、⛔ 不等結果）：
#   同一組【正式訓練配置】（S1_FROM=SOFT27 那組、STEPS2=300、seed 40）跑兩顆：
#     SANITY-base ＝ AMP=0            （fp32 基準）
#     SANITY-fast ＝ AMP=1 COMPILE=1  （fp16 autocast＋GradScaler、flow.nll 過 torch.compile）
#   要的只有兩件事：wall-clock 差多少、l_nf/l_anchor 的收斂曲線有沒有分岔。
#   ⛔ 不是要它出成績（STEPS2=300 的顆不進任何表）。
# ⚠️ 兩顆【序列】跑（fast 掛 afterany:base）—— 同時跑在同一台會互相搶，wall-clock 的比較就白做了。
# ⚠️ 產物：results/day_0906/sanity_amp/。檔名差在 _amp_cmp 段（_tag_extra 9/6 新增）
#    ⇒ ⛔ 兩顆不會互蓋。
set -euo pipefail
cd ~/Projects/lacot
APY=/archive/cymaxwelllee/LaCoT/.venv/bin/python
ADATA=/archive/cymaxwelllee/data/ogbench
BASE0="MUJOCO_GL=osmesa LACOT_ENC_OBJ=recon_ictr LACOT_LEARNED_REFINE=0 LACOT_BC_INDEP=1"
LENV=pointmaze-large-stitch-v0
SOFT27=results/ckpt_large-stitch_self_K8_c256_ch4_st8000_T128_ep2_gu_eorecon_ictr_tch0.5_emw0.999_wu500_dssoft_norf_cd0.1_bci_s27.pt
TRAIN0="LACOT_STEPS2=300 LACOT_TEACHER_MIX=0.5 LACOT_EVAL_RS=0 LACOT_EVAL_EPISODES=2 LACOT_DIAG_TRAIN=1"
OUT=results/day_0906/sanity_amp
mkdir -p slurm/logs "$OUT"

sub() { local name=$1 deps=$2; shift 2; local depflag=""
  [ "$deps" != "-" ] && depflag="--dependency=afterok:$deps"
  sbatch -p admin -A it -q great-mage --time=24:00:00 --nodelist=moana --gres=gpu:1 \
    --job-name=$name -o slurm/logs/%x-%j.out $depflag \
    --wrap "cd ~/Projects/lacot && env $*" | awk '{print $4}'; }

COMMON="OGBENCH_DATA_DIR=$ADATA $BASE0 LACOT_COND_DROP=0.1 $TRAIN0 LACOT_ENV=$LENV LACOT_K=8 \
LACOT_SEED=40 LACOT_EMA_W=0.999 LACOT_WARMUP=500 LACOT_DEC_START=soft LACOT_S1_FROM=$SOFT27 \
LACOT_INTENT=embed LACOT_INTENT_DROP=0.3 LACOT_LOG_EVERY=25 LACOT_BOOT_TAG=sanity LACOT_OUT_DIR=$OUT"

# ⭐ GATE=<jobid>：等 Patch A 的 AMP smoke 過了再跑（⛔ 別讓 sanity 變成第一個踩雷的）
JB=$(sub SANITY-base "${GATE:--}" "$COMMON $APY -u experiments/scratch_lacot_rollout.py")
JF=$(sub SANITY-fast "$JB" "$COMMON LACOT_AMP=1 LACOT_COMPILE=1 $APY -u experiments/scratch_lacot_rollout.py")
echo "SANITY-base $JB  (moana)"
echo "SANITY-fast $JF  (moana, afterok:$JB)"
