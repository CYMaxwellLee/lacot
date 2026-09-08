#!/usr/bin/env bash
#SBATCH -J M2-vqsel-ev
#SBATCH -p admin
#SBATCH -A it
#SBATCH -q great-mage
#SBATCH --nodelist=zeldajr
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH -o /home/cymaxwelllee/Projects/lacot-m2/experiments/m2_vqselect/logs/%x-%j.out
# M2 寫譜頭 eval。設定逐項照 job 25987（ANTG2-mainseg，主頭分段 .448／per-leg .320 那一支）
# 與 P2 vq_oracle（job 26202），唯一差別＝ LACOT_SUB_POLICY=vq_select ＋ ckpt 換成 M2 自己那顆。
# ⛔ 沒有 LACOT_LO_W —— M2 那顆 ckpt 沒有 lo 段（開了會當場 assert，那是對的）。
set -eu
WT=/home/cymaxwelllee/Projects/lacot-m2
PY=/home/cymaxwelllee/venvs/lacot-rocm/bin/python
DICT=${VQSEL_DICT:-/home/cymaxwelllee/Projects/lacot/experiments/walk_verify/results/p0_dict_v1_L4K32_50k.pt}
CK=${CK:-$(ls $WT/experiments/m2_vqselect/ckpt/ckpt_*_vqsW1_s40.pt | head -1)}
EPS=${EPS:-50}
OUT=${OUT:-$WT/experiments/m2_vqselect/results/eval}
TRACE=${TRACE:-$WT/experiments/m2_vqselect/traces}

cd "$WT"
echo "worktree: $(git -C "$WT" rev-parse --abbrev-ref HEAD) $(git -C "$WT" rev-parse HEAD)"
echo "ckpt: $CK"
echo "dict: $DICT"
echo "trace_dir: $TRACE   episodes: $EPS"
df -h / | tail -1

env OGBENCH_DATA_DIR=/home/cymaxwelllee/.ogbench/data MUJOCO_GL=osmesa \
    LACOT_ENC_OBJ=recon_ictr LACOT_LEARNED_REFINE=0 LACOT_BC_INDEP=1 LACOT_COND_DROP=0.1 \
    LACOT_ENV=antmaze-medium-stitch-v0 LACOT_K=8 LACOT_SEED=40 LACOT_TEACHER_MIX=0.5 \
    LACOT_LOAD_EMA=1 LACOT_LOAD_CKPT="$CK" LACOT_DEC_START=soft LACOT_INTENT=embed \
    LACOT_INTENT_DROP=0.3 LACOT_DEV_EVAL=0 LACOT_EVAL_RS=0 LACOT_EVAL_EPISODES=$EPS \
    LACOT_SUBGOAL=conf2 LACOT_LEG_STATS=1 LACOT_GRAD_REFINE=1 LACOT_GRAD_R=0 \
    LACOT_SUB_MAX_ARC=2 LACOT_FINISH_R=2.0 LACOT_OUT_DIR="$OUT" \
    LACOT_SUB_POLICY=vq_select LACOT_VQSEL_CKPT="$DICT" LACOT_VQSEL_TRACE_DIR="$TRACE" \
    "$PY" -u experiments/scratch_lacot_rollout.py
