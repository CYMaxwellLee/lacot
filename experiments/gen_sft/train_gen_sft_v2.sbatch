#!/usr/bin/env bash
#SBATCH -p admin
#SBATCH -A it
#SBATCH -q great-mage
#SBATCH --nodelist=zeldajr
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH -o /home/cymaxwelllee/Projects/lacot-gensft/experiments/gen_sft/logs/%x-%j.out
# gen-sft-v2：訓練 u 的容量掃/加權教材各版本（train_gen_sft_v2.py）。GPU only。
# ⛔ 施工在 worktree（~/Projects/lacot-gensft，分支 gen-sft-v1），主 repo main 不動。
# 跟 train_gen_sft.sbatch 的差異：呼叫 v2 訓練腳本、加 $EXTRA passthrough（跟
# eval_gen_sft.sbatch 同慣例）方便同一支 sbatch 檔用不同 --d-model/--n-layers/
# --n-heads/--weight-mode 等參數發射容量掃 3 格 + 加權 2 顆共 5 個 job。
set -eu
WT=/home/cymaxwelllee/Projects/lacot-gensft
PY=/home/cymaxwelllee/venvs/lacot-rocm/bin/python
STEPS=${STEPS:-30000}
TAG=${TAG:?must set TAG}
BATCH=${BATCH:-256}
EXTRA=${EXTRA:-}

cd "$WT"
echo "worktree: $(git -C "$WT" rev-parse --abbrev-ref HEAD) $(git -C "$WT" rev-parse HEAD)"
echo "node: $(hostname)  steps=$STEPS tag=$TAG batch=$BATCH extra=[$EXTRA]"
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
    "$PY" -u experiments/gen_sft/train_gen_sft_v2.py \
    --device cuda --steps "$STEPS" --batch-size "$BATCH" --tag "$TAG" \
    --out-dir experiments/gen_sft/results $EXTRA
