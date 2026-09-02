#!/bin/bash
# 2026-09-02 20:0x（草稿、⛔ 等主人點頭）：計畫開頭不在起點 ⇒ 三帖 eval-only 修法，低三高一各三臂＝12 支（zeldajr）
#   HG   開頭守門 LACOT_SUB_HEADGUARD=3.0（黏住整集、近終點放行）
#   DA   推論錨定 LACOT_DEC_ANCHOR=1（計畫平移到第 0 點＝現在位置；純 latent、不搜索）
#   HGS  守門＋路標吸附 LACOT_SUB_HEADGUARD=3.0 LACOT_SUB_SNAP=1
set -euo pipefail
cd ~/Projects/lacot
ZPY=$HOME/venvs/lacot-rocm/bin/python; ZDATA=$HOME/data/ogbench
BASE="MUJOCO_GL=osmesa LACOT_ENC_OBJ=recon_ictr LACOT_LEARNED_REFINE=0 LACOT_COND_DROP=0.1 LACOT_BC_INDEP=1"
OFF="LACOT_DEV_EVAL=0 LACOT_EVAL_RS=0 LACOT_EVAL_EPISODES=50"
C2MA="LACOT_SUBGOAL=conf2 LACOT_SUB_POLICY=bc LACOT_GRAD_REFINE=1 LACOT_GRAD_R=0 LACOT_SUB_MAX_ARC=2 LACOT_FINISH_R=2.0"
PRE=ckpt_large-stitch_self_K8_c256_ch4_st8000_T128_ep2_gu
declare -A ARM=( [HG]="LACOT_SUB_HEADGUARD=3.0" [DA]="LACOT_DEC_ANCHOR=1" [HGS]="LACOT_SUB_HEADGUARD=3.0 LACOT_SUB_SNAP=1" )
for A in HG DA HGS; do
  OUTD=results/night_0902/guard_$A; mkdir -p "$OUTD"
  for S in 23 25 26 27; do
    CK=results/${PRE}_eorecon_ictr_tch0.5_emw0.999_wu500_norf_cd0.1_bci_s$S.pt
    sbatch -p admin -A it -q great-mage --time=24:00:00 --nodelist=zeldajr --gres=gpu:1 --job-name=$A-s$S -o slurm/logs/%x-%j.out \
      --wrap "cd ~/Projects/lacot && env OGBENCH_DATA_DIR=$ZDATA $BASE LACOT_ENV=pointmaze-large-stitch-v0 LACOT_K=8 LACOT_TEACHER_MIX=0.5 LACOT_LOAD_EMA=1 LACOT_LOAD_CKPT=$CK LACOT_DIAG_DUMP=1 LACOT_OUT_DIR=$OUTD ${ARM[$A]} $OFF $C2MA $ZPY -u experiments/scratch_lacot_rollout.py" | awk '{print "'$A'-s'$S' -> " $4}'
  done
done
