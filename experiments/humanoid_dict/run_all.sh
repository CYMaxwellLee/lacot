#!/usr/bin/env bash
# 跑滿 26 格：24 主格（G∈{1,2,4,8,16,32} × K∈{8,16,32,64}，段長 4，跟 ant 完全同格）
# ＋ 2 段長 8 對照格（G16K16、G1K32）。CPU-only、8-way 平行（每 proc 4 threads
# = 32 threads，留一半機器給別人，這台是共用 gateway，跟 gk_scan/run_all.sh 同款設定）。
set -uo pipefail
cd "$(dirname "$0")"

DATA_DIR="${OGBENCH_DATA_DIR:-/home/cymaxwelllee/.ogbench/data}"
PY=/home/cymaxwelllee/venvs/lacot-rocm/bin/python
PARALLEL=8
THREADS=4

mkdir -p logs results ckpt

GS=(1 2 4 8 16 32)
KS=(8 16 32 64)
# 段長 8 對照格（任務指定：G16K16、G1K32）
EXTRA_CELLS=("16 16 8" "1 32 8")

pids=()
running=0

launch_cell () {
  local G="$1" K="$2" L="$3"
  local tag="G${G}_K${K}"
  if [ "$L" != "4" ]; then
    tag="G${G}_K${K}_L${L}"
  fi
  (
    MUJOCO_GL=osmesa OGBENCH_DATA_DIR="$DATA_DIR" \
    "$PY" run_cell.py --g "$G" --k "$K" --seg-len "$L" --threads "$THREADS" --data-dir "$DATA_DIR" \
      > "logs/${tag}.log" 2>&1
    echo "EXIT=$? tag=${tag}" >> "logs/_exit_codes.log"
  ) &
  pids+=($!)
  running=$((running+1))
  if [ "$running" -ge "$PARALLEL" ]; then
    wait -n
    running=$((running-1))
  fi
}

for G in "${GS[@]}"; do
  for K in "${KS[@]}"; do
    launch_cell "$G" "$K" "4"
  done
done
for cell in "${EXTRA_CELLS[@]}"; do
  read -r G K L <<< "$cell"
  launch_cell "$G" "$K" "$L"
done

wait

echo "=== run_all.sh done: $(date) ==="
echo "cells expected: 26, jsons found: $(ls results/G*.json 2>/dev/null | wc -l)"
grep -c '^EXIT=0' logs/_exit_codes.log 2>/dev/null | xargs -I{} echo "successful exits: {}"
