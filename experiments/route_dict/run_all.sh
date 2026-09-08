#!/usr/bin/env bash
# 跑滿 3 格（K∈{8,16,32}），CPU-only，3 格同時跑（量很小，不需分批）。
set -uo pipefail
cd "$(dirname "$0")"

DATA_DIR="${OGBENCH_DATA_DIR:-/home/cymaxwelllee/.ogbench/data}"
PY=/home/cymaxwelllee/venvs/lacot-rocm/bin/python
THREADS=4

mkdir -p logs results

for K in 8 16 32; do
  (
    MUJOCO_GL=osmesa OGBENCH_DATA_DIR="$DATA_DIR" \
    "$PY" run_cell.py --k "$K" --threads "$THREADS" --data-dir "$DATA_DIR" \
      > "logs/K${K}.log" 2>&1
    echo "EXIT=$? tag=K${K}" >> "logs/_exit_codes.log"
  ) &
done
wait

echo "=== run_all.sh done: $(date) ==="
echo "cells expected: 3, jsons found: $(ls results/K*.json 2>/dev/null | wc -l)"
grep -c '^EXIT=0' logs/_exit_codes.log 2>/dev/null | xargs -I{} echo "successful exits: {}"
