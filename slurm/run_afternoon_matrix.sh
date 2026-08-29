#!/bin/bash
# 下午自動實驗矩陣（主人 2026-08-29 搭機前交辦：「排一個script，下午安排實驗下去跑」）
#
# 設計：把病一/病二兩支快篩補全成 2×2 因子（w_len=0 × 終局接管 × {flat,S0,S1}）
#       ＋ conf（信心選點）首跑兩支。⇒ 主效應與交互作用一次判讀，歸因乾淨。
# 已在跑的不重發：Q-len=S1_wl0(20911)、Q-fin=S0_fin(20912)；基準 flat/S0/S1 = G3。
# 全部 eval-only（載同一顆 A2 ckpt）、tier2 100 題、24h 上限、四台輪配。
set -euo pipefail
cd "$HOME/Projects/lacot"
CK=results/ckpt_medium-stitch_self_K4_c256_ch4_st8000_T128_ep50_gu_eorecon_ictr_norf_cd0.1_bci_s0.pt
COMMON='OGBENCH_DATA_DIR=/archive/cymaxwelllee/data/ogbench MUJOCO_GL=osmesa
 LACOT_ENV=pointmaze-medium-stitch-v0 LACOT_ENC_OBJ=recon_ictr LACOT_LEARNED_REFINE=0
 LACOT_COND_DROP=0.1 LACOT_BC_INDEP=1 LACOT_DEV_EVAL=1 LACOT_DEV_TIERS=2
 LACOT_DEV_PER_TIER=100 LACOT_GRAD_REFINE=1'
COMMON=$(echo $COMMON)   # 摺成一行

launch () {  # launch <name> <node> <extra-env...>
    local name=$1 node=$2; shift 2
    sbatch -p admin -A it -q great-mage --time=24:00:00 --nodelist="$node" --gres=gpu:1 \
        --job-name="$name" -o slurm/logs/%x-%j.out \
        --wrap "cd ~/Projects/lacot && env $COMMON $* LACOT_LOAD_CKPT=$CK /archive/cymaxwelllee/LaCoT/.venv/bin/python -u experiments/scratch_lacot_rollout.py"
}

# ── 因子矩陣補全（9 支）────────────────────────────────────────────
launch M-fl-wl   pocahontas 'LACOT_W_LEN=0'
launch M-fl-fin  pocahontas 'LACOT_FINISH_R=2.0'
launch M-fl-wf   jasmine    'LACOT_W_LEN=0 LACOT_FINISH_R=2.0'
launch M-S0-wl   jasmine    'LACOT_SUBGOAL=bfs LACOT_W_LEN=0'
launch M-S0-wf   moana      'LACOT_SUBGOAL=bfs LACOT_W_LEN=0 LACOT_FINISH_R=2.0'
launch M-S1-fin  moana      'LACOT_SUBGOAL=latent LACOT_FINISH_R=2.0'
launch M-S1-wf   lady       'LACOT_SUBGOAL=latent LACOT_W_LEN=0 LACOT_FINISH_R=2.0'
# ── conf（信心選點，主人核准後的首跑）──────────────────────────────
launch M-conf    lady       'LACOT_SUBGOAL=conf'
launch M-conf-wf pocahontas 'LACOT_SUBGOAL=conf LACOT_W_LEN=0 LACOT_FINISH_R=2.0'

squeue -u "$USER" -o '%i %j %T %l %N'
