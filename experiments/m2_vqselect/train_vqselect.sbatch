#!/usr/bin/env bash
#SBATCH -J M2-vqsel-tr
#SBATCH -p admin
#SBATCH -A it
#SBATCH -q great-mage
#SBATCH --nodelist=zeldajr
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=8:00:00
#SBATCH -o /home/cymaxwelllee/Projects/lacot-m2/experiments/m2_vqselect/logs/%x-%j.out
# M2 寫譜頭訓練。⭐ 咒語【逐項照 lo 頭 v1 那支】（job 25775，ckpt results/ant_pilot/lo_100k），
#   唯一差別＝ LACOT_LO_W=1 換成 LACOT_VQSEL_W=1。
# ⇒ 主模型（計畫棧）在兩顆之間應該【逐位元相同】—— fork_rng ＋ 專用 RNG ＋ 獨立 optimizer
#   三道隔離兌現的話。收工要拿 l_nf/l_anchor 這幾行去對 slurm/logs/ANTG1-lo-tr-25775.out。
# ⛔ 施工在 worktree（~/Projects/lacot-m2），主 repo main 不動。⛔ GPU 一律走 slurm。
set -eu
WT=/home/cymaxwelllee/Projects/lacot-m2
PY=/home/cymaxwelllee/venvs/lacot-rocm/bin/python
S1=/home/cymaxwelllee/Projects/lacot/results/ant_pilot/idp03/ckpt_antmaze-medium-stitch_self_K8_c256_ch4_st8000_T128_ep2_gu_eorecon_ictr_tch0.5_emw0.999_wu500_ite_idp0.3_dssoft_norf_cd0.1_bci_s40.pt
DICT=${VQSEL_DICT:-/home/cymaxwelllee/Projects/lacot/experiments/walk_verify/results/p0_dict_v1_L4K32_50k.pt}
OUT=${OUT:-$WT/experiments/m2_vqselect/ckpt}
STEPS2=${STEPS2:-100000}

cd "$WT"
echo "worktree: $(git -C "$WT" rev-parse --abbrev-ref HEAD) $(git -C "$WT" rev-parse HEAD)"
echo "S1_FROM: $S1"
echo "dict   : $DICT"
echo "steps2 : $STEPS2   out: $OUT"
df -h / | tail -1

env OGBENCH_DATA_DIR=/home/cymaxwelllee/.ogbench/data MUJOCO_GL=osmesa \
    LACOT_ENC_OBJ=recon_ictr LACOT_LEARNED_REFINE=0 LACOT_BC_INDEP=1 LACOT_COND_DROP=0.1 \
    LACOT_ENV=antmaze-medium-stitch-v0 LACOT_K=8 LACOT_SEED=40 LACOT_TEACHER_MIX=0.5 \
    LACOT_EMA_W=0.999 LACOT_WARMUP=500 LACOT_DEC_START=soft LACOT_INTENT=embed \
    LACOT_INTENT_DROP=0.3 LACOT_TCAP=128 LACOT_S1_FROM="$S1" \
    LACOT_STEPS2="$STEPS2" LACOT_EVAL_EPISODES=2 LACOT_EVAL_RS=0 LACOT_DEV_EVAL=0 \
    LACOT_VQSEL_W=1 LACOT_VQSEL_CKPT="$DICT" \
    LACOT_OUT_DIR="$OUT" \
    "$PY" -u experiments/scratch_lacot_rollout.py
