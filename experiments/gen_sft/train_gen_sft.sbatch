#!/usr/bin/env bash
#SBATCH -p admin
#SBATCH -A it
#SBATCH -q great-mage
#SBATCH --nodelist=zeldajr
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH -o /home/cymaxwelllee/Projects/lacot-gensft/experiments/gen_sft/logs/%x-%j.out
# gen-sft-v1：訓練 u（(s0,路標序列)->code 字串自回歸生成）。GPU only。
# ⛔ 施工在 worktree（~/Projects/lacot-gensft，分支 gen-sft-v1），主 repo main 不動。
#
# 除錯記錄（2026-09-09，真正原因）：job 26819/21/23/25 連續 torch.cuda.is_available()
# =False FAILED。真正原因不是這層 sbatch/slurm：是 train_gen_sft.py import
# gen_sft_common -> run_teacher_relay 時，後者檔頭 `os.environ.setdefault(
# "HIP_VISIBLE_DEVICES","")`（它自己是 CPU-only 腳本的自保寫法）在「slurm 從沒設過
# HIP_VISIBLE_DEVICES」的前提下真的生效，把它釘成空字串，蓋過 CUDA_VISIBLE_DEVICES=0
# —— ROCm 把「HIP 可見裝置=空字串」讀成看不到卡。修法在 train_gen_sft.py 內：import
# gen_sft_common 之前先呼叫一次 torch.cuda.is_available()，把偵測結果快取住。這裡的
# 重試迴圈是繞路中間產物、不是真正修法，留著只當防禦（若真的遇到卡忙的暫態問題還是
# 有用），不能拿掉本行程式碼裡的那個 guard。
set -eu
WT=/home/cymaxwelllee/Projects/lacot-gensft
PY=/home/cymaxwelllee/venvs/lacot-rocm/bin/python
STEPS=${STEPS:-60000}
TAG=${TAG:-gensft_v1}
BATCH=${BATCH:-256}

cd "$WT"
echo "worktree: $(git -C "$WT" rev-parse --abbrev-ref HEAD) $(git -C "$WT" rev-parse HEAD)"
echo "node: $(hostname)  steps=$STEPS tag=$TAG batch=$BATCH"
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
    --device cuda --steps "$STEPS" --batch-size "$BATCH" --tag "$TAG" \
    --out-dir experiments/gen_sft/results
