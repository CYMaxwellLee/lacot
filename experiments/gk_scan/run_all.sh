#!/usr/bin/env bash
# 跑滿 24 格（G∈{1,2,4,8,16,32} × K∈{8,16,32,64}），CPU-only、8-way 平行（每 proc 4 threads
# = 32 threads，留一半機器給別人，這台是共用 gateway）。
set -uo pipefail
cd "$(dirname "$0")"

DATA_DIR="${OGBENCH_DATA_DIR:-/home/cymaxwelllee/.ogbench/data}"
PY=/home/cymaxwelllee/venvs/lacot-rocm/bin/python
PARALLEL=8
THREADS=4

mkdir -p logs results ckpt

GS=(1 2 4 8 16 32)
KS=(8 16 32 64)

pids=()
running=0
for G in "${GS[@]}"; do
  for K in "${KS[@]}"; do
    tag="G${G}_K${K}"
    (
      MUJOCO_GL=osmesa OGBENCH_DATA_DIR="$DATA_DIR" \
      "$PY" run_cell.py --g "$G" --k "$K" --threads "$THREADS" --data-dir "$DATA_DIR" \
        > "logs/${tag}.log" 2>&1
      echo "EXIT=$? tag=${tag}" >> "logs/_exit_codes.log"
    ) &
    pids+=($!)
    running=$((running+1))
    if [ "$running" -ge "$PARALLEL" ]; then
      wait -n
      running=$((running-1))
    fi
  done
done
wait

echo "=== run_all.sh done: $(date) ==="
echo "cells expected: 24, jsons found: $(ls results/G*_K*.json 2>/dev/null | wc -l)"
grep -c '^EXIT=0' logs/_exit_codes.log 2>/dev/null | xargs -I{} echo "successful exits: {}"
