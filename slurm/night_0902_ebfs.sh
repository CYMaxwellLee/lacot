#!/bin/bash
# 2026-09-02 下午（草稿、⛔ 未經主人點頭不送）：ebfs 分辨器 —— 餵 E 圖搜索的正確路標、用同一顆 ckpt 的執行通道
# 問：低分顆（s23/s25/s26）的路口失敗是「計畫給錯路標」還是「執行通道走不到」。s27 當對照（好顆 ebfs 應 ~0.97）。
# 只 eval、不重訓；產物走 results/night_0902/ebfs/（seed 不同、不互蓋）。
set -euo pipefail
cd ~/Projects/lacot
ZPY=$HOME/venvs/lacot-rocm/bin/python
ZDATA=$HOME/data/ogbench
BASE="MUJOCO_GL=osmesa LACOT_ENC_OBJ=recon_ictr LACOT_LEARNED_REFINE=0 LACOT_COND_DROP=0.1 LACOT_BC_INDEP=1"
OFF="LACOT_DEV_EVAL=0 LACOT_EVAL_RS=0 LACOT_EVAL_EPISODES=50"
EB="LACOT_SUBGOAL=ebfs LACOT_SUB_POLICY=bc LACOT_GRAD_REFINE=1 LACOT_FINISH_R=2.0"
LENV=pointmaze-large-stitch-v0
PRE=ckpt_large-stitch_self_K8_c256_ch4_st8000_T128_ep2_gu
mkdir -p results/night_0902/ebfs slurm/logs
for S in 23 25 26 27; do
  CK=results/${PRE}_eorecon_ictr_tch0.5_emw0.999_wu500_norf_cd0.1_bci_s$S.pt
  [ -f "$CK" ] || { echo "⛔ 缺 ckpt $CK"; exit 1; }
  J=$(sbatch -p admin -A it -q great-mage --time=24:00:00 --nodelist=zeldajr --gres=gpu:1 --job-name=EB-s$S -o slurm/logs/%x-%j.out \
      --wrap "cd ~/Projects/lacot && env OGBENCH_DATA_DIR=$ZDATA $BASE LACOT_ENV=$LENV LACOT_K=8 LACOT_TEACHER_MIX=0.5 LACOT_LOAD_EMA=1 LACOT_LOAD_CKPT=$CK LACOT_DIAG_DUMP=1 LACOT_OUT_DIR=results/night_0902/ebfs $OFF $EB $ZPY -u experiments/scratch_lacot_rollout.py" | awk '{print $4}')
  echo "EB-s$S -> zeldajr ($J)"
done
echo "== 共 4 支 eval"
