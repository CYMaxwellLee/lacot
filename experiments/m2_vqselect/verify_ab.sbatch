#!/usr/bin/env bash
#SBATCH -J M2-verify-ab
#SBATCH -p admin
#SBATCH -A it
#SBATCH -q great-mage
#SBATCH --nodelist=zeldajr
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=2:00:00
#SBATCH -o /home/cymaxwelllee/Projects/lacot-m2/experiments/m2_vqselect/logs/%x-%j.out
# M2 施工驗收（兩關一起跑，共用同一張卡）：
#   ① golden A/A/B（pointmaze、GPU）：main vs vq-select-v1 worktree 零差
#   ② ant 側旗關閉零差：goldab 只證 pointmaze，這支補 antmaze 這條路徑
# ⛔ GOLD_OUT／輸出目錄一律指到【新目錄】—— gold_ab_wrap.sh 開頭會 rm -rf 輸出目錄，
#    用預設值會刪掉 lo-v1／vq-oracle-v1 那兩次的既有證據。
set -u
WT=/home/cymaxwelllee/Projects/lacot-m2
PY=/home/cymaxwelllee/venvs/lacot-rocm/bin/python

echo "===================== ① goldab（pointmaze、GPU）====================="
export BRANCH_ROOT=$WT
export GOLD_OUT=/home/cymaxwelllee/Projects/lacot/results/goldab_vqselect
bash /home/cymaxwelllee/Projects/lacot/slurm/goldab/gold_ab_wrap.sh

echo
echo "===================== ② ant 旗關閉零差 ============================="
G=/home/cymaxwelllee/Projects/lacot/results/flagoff_ant_ab_m2
rm -rf "$G"; mkdir -p "$G/A" "$G/B"
LCK=$(ls /home/cymaxwelllee/Projects/lacot/results/ant_pilot/lo_100k/ckpt_*_s40.pt | head -1)

run_one () {   # $1=repo root  $2=out dir
  cd "$1"
  env OGBENCH_DATA_DIR=/home/cymaxwelllee/.ogbench/data MUJOCO_GL=osmesa \
      LACOT_ENC_OBJ=recon_ictr LACOT_LEARNED_REFINE=0 LACOT_BC_INDEP=1 LACOT_COND_DROP=0.1 \
      LACOT_ENV=antmaze-medium-stitch-v0 LACOT_K=8 LACOT_SEED=40 LACOT_TEACHER_MIX=0.5 \
      LACOT_LOAD_EMA=1 LACOT_LOAD_CKPT=$LCK LACOT_DEC_START=soft LACOT_INTENT=embed \
      LACOT_INTENT_DROP=0.3 LACOT_DEV_EVAL=0 LACOT_EVAL_RS=0 LACOT_EVAL_EPISODES=1 \
      LACOT_EVAL_MAXH=120 LACOT_SUBGOAL=conf2 LACOT_LO_W=1 LACOT_GRAD_REFINE=1 LACOT_GRAD_R=0 \
      LACOT_SUB_MAX_ARC=2 LACOT_FINISH_R=2.0 LACOT_OUT_DIR="$2" \
      "$PY" -u /home/cymaxwelllee/Projects/lacot/slurm/goldab/detrun.py \
            "$1/experiments/scratch_lacot_rollout.py"
}

echo "== git =="
git -C /home/cymaxwelllee/Projects/lacot rev-parse --abbrev-ref HEAD HEAD
git -C "$WT"                             rev-parse --abbrev-ref HEAD HEAD
echo "== A (main, flag off) =="
run_one /home/cymaxwelllee/Projects/lacot "$G/A" > "$G/A.log" 2>&1; echo "A rc=$?"
echo "== B (worktree, flag off) =="
run_one "$WT"                             "$G/B" > "$G/B.log" 2>&1; echo "B rc=$?"

echo "== sha256 =="
for d in A B; do (cd "$G/$d" && sha256sum rollout_*.json 2>/dev/null | sed "s|^|$d |"); done
SA=$( (cd "$G/A" && sha256sum rollout_*.json 2>/dev/null | awk '{print $1}') )
SB=$( (cd "$G/B" && sha256sum rollout_*.json 2>/dev/null | awk '{print $1}') )
echo "== 檔名比對 =="
(cd "$G/A" && ls) ; (cd "$G/B" && ls)
echo "== stdout diff (A vs B，路徑字串會不同、其餘應全同) =="
diff <(sed "s|$G/A|OUT|g" "$G/A.log") <(sed "s|$G/B|OUT|g" "$G/B.log") | head -20
if [ -n "$SA" ] && [ "$SA" == "$SB" ]; then
  echo "FLAGOFF VERDICT: PASS — ant 路徑旗關閉零差（rollout json sha256 全同）"
else
  echo "FLAGOFF VERDICT: FAIL — ant 路徑旗關閉有差（A=$SA B=$SB）"
fi
