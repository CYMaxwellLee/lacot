#!/usr/bin/env bash
#SBATCH -p admin
#SBATCH -A it
#SBATCH -q great-mage
#SBATCH --nodelist=zeldajr
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH -o /home/cymaxwelllee/Projects/lacot-rewrite/experiments/gen_sft/logs/%x-%j.out
# rewrite-v1 步2：鐘形定讞 + 半路起點增強 —— 六顆訓練（3 seed x {原語料,增強語料}）。
# ⛔ 新檔（v3/aug 系列），不改動既有 train_gen_sft.sbatch / train_gen_sft.py。
# ⛔ 施工只在 worktree ~/Projects/lacot-rewrite（分支 rewrite-v1），main 不動，
#    lacot-gensft 的既有 ckpt 唯讀不碰。
#
# 沿用既有 train_gen_sft.sbatch 的 GPU-visibility-guard 除錯結論（2026-09-09,
# job 26819/21/23/25 連續 FAILED 的根因）：train_gen_sft.py 內部在 import
# gen_sft_common 之前已經呼叫一次 torch.cuda.is_available() 把偵測結果快取住
# （見該檔案內建 guard），這裡的重試迴圈只當防禦（真的卡忙的暫態問題還有用）。
set -eu
WT=/home/cymaxwelllee/Projects/lacot-rewrite
PY=/home/cymaxwelllee/venvs/lacot-rocm/bin/python
CORPUS=${CORPUS:?must set CORPUS}
SEED=${SEED:?must set SEED}
TAG=${TAG:?must set TAG}
STEPS=${STEPS:-30000}
BATCH=${BATCH:-256}

cd "$WT"
echo "worktree: $(git -C "$WT" rev-parse --abbrev-ref HEAD) $(git -C "$WT" rev-parse HEAD)"
echo "node: $(hostname)  corpus=$CORPUS seed=$SEED steps=$STEPS tag=$TAG batch=$BATCH"
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

env OGBENCH_DATA_DIR=/home/cymaxwelllee/.ogbench/data \
    "$PY" -u experiments/gen_sft/train_gen_sft.py \
    --corpus "$CORPUS" --device cuda --steps "$STEPS" --batch-size "$BATCH" \
    --seed "$SEED" --tag "$TAG" \
    --out-dir experiments/gen_sft/results
