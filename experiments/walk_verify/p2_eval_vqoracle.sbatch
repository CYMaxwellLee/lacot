#!/usr/bin/env bash
#SBATCH -p admin
#SBATCH -A it
#SBATCH -q great-mage
#SBATCH --nodelist=zeldajr
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH -o /home/cymaxwelllee/Projects/lacot/experiments/walk_verify/logs/%x-%j.out
# P2：vq_oracle 作弊選字上界。設定逐項照 job 25987（ANTG2-mainseg，主頭分段 .448／
# per-leg .320 的那一支），唯一差別＝LACOT_SUB_POLICY=vq_oracle ＋ 從 worktree 跑。
# ⛔ 施工在 worktree（~/Projects/lacot-vqoracle），主 repo main 不動。
set -eu
WT=/home/cymaxwelllee/Projects/lacot-vqoracle
PY=/home/cymaxwelllee/venvs/lacot-rocm/bin/python
DICT=${VQO_DICT:-/home/cymaxwelllee/Projects/lacot/experiments/walk_verify/results/p0_dict_v1_L4K32_50k.pt}
EPS=${EPS:-50}
MAXH_ARG=${MAXH_ARG:-}
OUT=${OUT:-results/ant_pilot/eval_vqo}
TRACE=${TRACE:-/home/cymaxwelllee/Projects/lacot/experiments/walk_verify/results/p2_traces}

cd "$WT"
LCK=$(ls /home/cymaxwelllee/Projects/lacot/results/ant_pilot/lo_100k/ckpt_*_s40.pt | head -1)
echo "worktree: $(git -C "$WT" rev-parse --abbrev-ref HEAD) $(git -C "$WT" rev-parse HEAD)"
echo "ckpt: $LCK"
echo "dict: $DICT"
echo "trace_dir: $TRACE   episodes: $EPS   extra: ${MAXH_ARG}"

env OGBENCH_DATA_DIR=/home/cymaxwelllee/.ogbench/data MUJOCO_GL=osmesa \
    LACOT_ENC_OBJ=recon_ictr LACOT_LEARNED_REFINE=0 LACOT_BC_INDEP=1 LACOT_COND_DROP=0.1 \
    LACOT_ENV=antmaze-medium-stitch-v0 LACOT_K=8 LACOT_SEED=40 LACOT_TEACHER_MIX=0.5 \
    LACOT_LOAD_EMA=1 LACOT_LOAD_CKPT=$LCK LACOT_DEC_START=soft LACOT_INTENT=embed \
    LACOT_INTENT_DROP=0.3 LACOT_DEV_EVAL=0 LACOT_EVAL_RS=0 LACOT_EVAL_EPISODES=$EPS \
    LACOT_SUBGOAL=conf2 LACOT_LO_W=1 LACOT_LEG_STATS=1 LACOT_GRAD_REFINE=1 LACOT_GRAD_R=0 \
    LACOT_SUB_MAX_ARC=2 LACOT_FINISH_R=2.0 LACOT_OUT_DIR="$OUT" \
    LACOT_SUB_POLICY=vq_oracle LACOT_VQO_CKPT="$DICT" LACOT_VQO_TRACE_DIR="$TRACE" \
    ${MAXH_ARG} \
    "$PY" -u experiments/scratch_lacot_rollout.py
