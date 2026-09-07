#!/usr/bin/env bash
# M0 偵察：跑 6 格 VQ-VAE 步法字典（seg_len {4,8} × K {8,16,32}）。
# ⛔ CPU-only、不經 slurm（規模小，CPU 直跑符合主人給的規則）。
# 為了在共用 gateway（zeldajr）上當個好鄰居：nice + 限制 thread 數、六格依序跑（不搶機器）。
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=/home/cymaxwelllee/venvs/lacot-rocm/bin/python
export OGBENCH_DATA_DIR=/home/cymaxwelllee/.ogbench/data
export CUDA_VISIBLE_DEVICES=""
export HIP_VISIBLE_DEVICES=""
export MUJOCO_GL=osmesa
export OMP_NUM_THREADS=4

mkdir -p "$HERE/results" "$HERE/logs"

run_one() {
  local seg_len=$1
  local k=$2
  local extra=${3:-}
  local tag="L${seg_len}_K${k}"
  echo "[run_all] starting $tag $extra"
  nice -n 15 "$PY" "$HERE/run_config.py" \
    --seg-len "$seg_len" --k "$k" --threads 4 --out-dir "$HERE" --data-dir "$OGBENCH_DATA_DIR" \
    $extra > "$HERE/logs/${tag}.log" 2>&1
  echo "[run_all] finished $tag (exit $?)"
}

for seg_len in 4 8; do
  for k in 8 16 32; do
    if [ "$seg_len" = "4" ] && [ "$k" = "16" ]; then
      run_one "$seg_len" "$k" "--visualize"
    else
      run_one "$seg_len" "$k"
    fi
  done
done

echo "[run_all] all 6 configs done"
