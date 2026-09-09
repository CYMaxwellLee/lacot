#!/usr/bin/env bash
#SBATCH -p admin
#SBATCH -A it
#SBATCH -q great-mage
#SBATCH --nodelist=zeldajr
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH -o /home/cymaxwelllee/Projects/lacot-dyn/experiments/gen_sft/logs/%x-%j.out
# 主人裁示①②：動態隨機起點 + 可選漂移擾動 —— 六顆訓練（3 seed x {無擾動,輕擾動}）。
# ⛔ 新檔（dyn 系列），不改動既有 train_gen_sft.sbatch / train_gen_sft_aug.sbatch /
#    train_gen_sft.py / train_gen_sft_dyn.py。
# ⛔ 施工只在 worktree ~/Projects/lacot-dyn（分支 rewrite-dyn）—— ⚠️ 這份 sbatch 是
#    照 train_gen_sft_aug.sbatch 抄過來的骨架，那份原檔 WT 寫的是 lacot-rewrite
#    （並行隊友的施工目錄，本工單完全不碰！）——已改成 lacot-dyn，提交前有重新核對。
#
# 沿用既有 GPU-visibility-guard 除錯結論（2026-09-09, job 26819/21/23/25 FAILED 根因，
# 見 train_gen_sft.py / train_gen_sft_dyn.py 檔頭）：這裡的重試迴圈只當防禦。
set -eu
WT=/home/cymaxwelllee/Projects/lacot-dyn
PY=/home/cymaxwelllee/venvs/lacot-rocm/bin/python
CORPUS=${CORPUS:?must set CORPUS}
SEED=${SEED:?must set SEED}
TAG=${TAG:?must set TAG}
STEPS=${STEPS:-30000}
BATCH=${BATCH:-256}
PERTURB=${PERTURB:-0}

cd "$WT"
echo "worktree: $(git -C "$WT" rev-parse --abbrev-ref HEAD) $(git -C "$WT" rev-parse HEAD)"
echo "node: $(hostname)  corpus=$CORPUS seed=$SEED steps=$STEPS tag=$TAG batch=$BATCH perturb=$PERTURB"
echo "env(visible/gpu): $(env | grep -iE 'visible|rocm|hip|cuda|gpu' | sort | tr '\n' ' ')"
rocm-smi --showproductname 2>&1 | head -6 || true

ok=0
for attempt in 1 2 3 4 5; do
  if "$PY" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
    ok=1
    echo "torch.cuda.is_available()=True on attempt $attempt"
    break
  fi
  echo "attempt $attempt: torch.cuda.is_available()=False, sleep 10s and retry"
  sleep 10
done
if [ "$ok" -ne 1 ]; then
  echo "⛔ 5 次重試後 torch.cuda.is_available() 仍是 False，不是暫態問題，停手回報"
  rocm-smi 2>&1 || true
  exit 1
fi

PERTURB_FLAG=""
if [ "$PERTURB" = "1" ]; then PERTURB_FLAG="--perturb"; fi

env OGBENCH_DATA_DIR=/home/cymaxwelllee/.ogbench/data \
    "$PY" -u experiments/gen_sft/train_gen_sft_dyn.py \
    --corpus "$CORPUS" --device cuda --steps "$STEPS" --batch-size "$BATCH" \
    --seed "$SEED" --tag "$TAG" $PERTURB_FLAG \
    --out-dir experiments/gen_sft/results
