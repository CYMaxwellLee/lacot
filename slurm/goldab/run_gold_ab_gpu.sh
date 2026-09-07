#!/usr/bin/env bash
# GPU 版零差單跑：$1=主檔路徑 $2=OUT_DIR。
# 跟 run_pm2.sh 同配置（9/6 正式配方計畫棧、重裝），差別只有：GPU 可見、cd 到該檔的 repo root。
set -u
SP=~/Projects/lacot/slurm/goldab
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export MUJOCO_GL=osmesa
export OGBENCH_DATA_DIR=/home/cymaxwelllee/.ogbench/data
export LACOT_ENV=pointmaze-medium-stitch-v0
export LACOT_SEED=0 LACOT_K=8
export LACOT_ENC_OBJ=recon_ictr LACOT_LEARNED_REFINE=0 LACOT_BC_INDEP=1
export LACOT_TEACHER_MIX=0.5 LACOT_EMA_W=0.999 LACOT_WARMUP=5
export LACOT_DEC_START=soft LACOT_COND_DROP=0.1
export LACOT_INTENT=embed
export LACOT_SUBGOAL=conf2 LACOT_SUB_POLICY=bc LACOT_GRAD_REFINE=1 LACOT_GRAD_R=0
export LACOT_SUB_MAX_ARC=2 LACOT_FINISH_R=2.0
export LACOT_BON_N=4 LACOT_BON_CALIB_N=32
export LACOT_GRPO_W=1.0 LACOT_GRPO_WARM=2 LACOT_GRPO_G=2 LACOT_GRPO_BQ=2
export LACOT_DIV_W=0.1 LACOT_DIV_LOG_EVERY=10 LACOT_INTENT_DROP=0.3
export LACOT_TCAP=64
export LACOT_STEPS1=150 LACOT_STEPS2=20
export LACOT_EVAL_EPISODES=1 LACOT_EVAL_MAXH=20 LACOT_EVAL_RS=0
export LACOT_LOG_EVERY=5
export LACOT_OUT_DIR="$2"
ROOT="$(dirname "$(dirname "$1")")"
cd "$ROOT"
exec ~/venvs/lacot-rocm/bin/python -u "$SP/detrun.py" "$1"
