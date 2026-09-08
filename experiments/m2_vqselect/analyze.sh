#!/usr/bin/env bash
# M2 四關全表 ＋ 6 支 gif。⭐ 直接沿用 P2 的量尺包分析器（experiments/walk_verify/p2_analyze_traces.py）
#   —— 同一支 code、同一把尺 ⇒ 跟 vq_oracle 那張表【可以直接並排比】，⛔ 不自己重寫一把尺。
# ⛔ CPU-only（osmesa 軟體算圖），⛔ 不上 GPU。
set -eu
WT=/home/cymaxwelllee/Projects/lacot-m2
PY=/home/cymaxwelllee/venvs/lacot-rocm/bin/python
TRACE=${TRACE:-$WT/experiments/m2_vqselect/traces}
OUT=${OUT:-$WT/experiments/m2_vqselect/results}
GIF=${GIF:-$OUT/m2_gifs}

cd "$WT/experiments/walk_verify"
env OGBENCH_DATA_DIR=/home/cymaxwelllee/.ogbench/data OMP_NUM_THREADS=8 \
    "$PY" -u p2_analyze_traces.py \
      --trace-dir "$TRACE" \
      --ruler /home/cymaxwelllee/Projects/lacot/experiments/walk_verify/results/ruler_pack.json \
      --out-dir "$OUT" --gif-dir "$GIF" --tag m2_vqsel \
      --n-gif-success 3 --n-gif-fail 3
