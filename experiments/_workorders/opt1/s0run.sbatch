#!/bin/bash
#SBATCH --partition=admin
#SBATCH --account=it
#SBATCH --qos=great-mage
#SBATCH --nodelist=lady
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --job-name=opt1-s0
#SBATCH --output=/home/cymaxwelllee/Projects/lacot/results/opt1/s0/logs/s0-%j.out
# F4/S0 首航 v3（兩輪 F5 預檢後）。變更帳：
# v2→v3（S0 預檢官 NO-GO 五條全處置）：
#  1) 隱形有效預設顯式凍結：LACOT_W_ICTR=0.2 / LACOT_ICTR_SIGMA=0.05 / LACOT_DEC_START_W=1.0
#     （鍵名 lead grep :142-143/:224 親驗）。
#  2) guard 全 fail-closed：query 失敗即停（去 || true）、MEMUSED 非整數即停。
#  3) SM 閘換 nvidia-smi --query-gpu=utilization.gpu 迴圈 ×10（cgroup 內只回本卡、與 UUID 一致；
#     dmon 全機混卡風險由此消除。lady dmon 實樣=兩行表頭、$2=sm，格式假設成立但不再依賴它）。
#  4) interlock：S0_RUNS 環境變數控段——先發 S0_RUNS="00"（r00 暖機兼 smoke），lead 人工驗
#     r00 產物（JSON 有 ema_m、tcap=128、episodes=10、ckpt cfg 帶 EMA_W 與 ema、恰一 JSON 一 PT、
#     time.txt 外推 6 跑 <8h）後才發 S0_RUNS="01 02 03 04 05"。⛔ 不再一發跑六。
#  5) schema 裁定（card 修訂）：EMA_W 以 ckpt cfg 為唯一來源；JSON 驗 ema_m 存在即可。
# lead 已驗（9/28）：lady=3×2080Ti、marl2026 佔 GPU0（sm 34%）、閒 2 卡；npz 19MB+import 鏈全通；
# /home NFS 剩 1.3T；/usr/bin/time 在；MAX_TRAIN_T>=128 由既有 ckpt 名帶 _T128 佐證。
set -euo pipefail
cd /home/cymaxwelllee/Projects/lacot
while IFS='=' read -r KEY _; do
  case "$KEY" in LACOT_*) unset "$KEY" ;; esac
done < <(env)
PY=/archive/cymaxwelllee/LaCoT/.venv/bin/python
DATA=/archive/cymaxwelllee/data/ogbench/pointmaze-large-stitch-v0.npz
ROOT=results/opt1/s0
RUNS="${S0_RUNS:-00}"
mkdir -p "$ROOT/logs"
test -x "$PY"
test -f "$DATA"
"$PY" -B -c 'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))'
nvidia-smi -L > "$ROOT/gpu-list.txt"
NVIS=$(wc -l < "$ROOT/gpu-list.txt")
if [ "$NVIS" -ne 1 ]; then
  echo "job 內可見 GPU 數=$NVIS（預期 1）。本站 cgroup 未按預期隔離——停止並回報 lead，勿放寬" >&2
  exit 20
fi
GPU_UUID=$(nvidia-smi --query-gpu=uuid --format=csv,noheader | head -1 | tr -d '[:space:]')
test -n "$GPU_UUID"
echo "assigned GPU UUID=$GPU_UUID SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}" | tee "$ROOT/gpu-assignment.txt"
nvidia-smi topo -m > "$ROOT/topo.txt" || echo "topo unavailable in cgroup" >> "$ROOT/topo.txt"
guard_gate() {  # $1=階段標籤；fail-closed：任何查詢失敗=停
  local TAG=$1
  nvidia-smi --query-compute-apps=gpu_uuid,pid,used_gpu_memory --format=csv > "$ROOT/gpu-procs-$TAG.csv"
  if grep -Fq "$GPU_UUID" "$ROOT/gpu-procs-$TAG.csv"; then
    echo "$TAG：本卡有其他 compute PID，停止" >&2; exit 21
  fi
  local MEMUSED
  MEMUSED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d '[:space:]')
  case "$MEMUSED" in
    ''|*[!0-9.]*) echo "$TAG：VRAM 讀值非數字（'$MEMUSED'），停止" >&2; exit 22 ;;
  esac
  if [ "${MEMUSED%%.*}" -ge 512 ]; then echo "$TAG：VRAM ${MEMUSED}MiB ≥512，停止" >&2; exit 22; fi
  local BUSY=0 N=0 U
  for _ in 1 2 3 4 5 6 7 8 9 10; do
    U=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1 | tr -d '[:space:]')
    case "$U" in ''|*[!0-9]*) echo "$TAG：SM 讀值非數字（'$U'），停止" >&2; exit 23 ;; esac
    N=$((N+1)); [ "$U" -ge 5 ] && BUSY=$((BUSY+1)) || true
    sleep 1
  done
  echo "$TAG: sm-samples=$N busy(>=5%)=$BUSY" >> "$ROOT/sm-gate.log"
  if [ "$N" -ne 10 ] || [ "$BUSY" -gt 2 ]; then echo "$TAG：SM 閘未過（$BUSY/10 ≥5%），停止" >&2; exit 23; fi
}
guard_gate "before-$RUNS"
sha256sum "$DATA" > "$ROOT/dataset.sha256"
git rev-parse HEAD > "$ROOT/git-head.txt"
for R in $RUNS; do
  OUT="$ROOT/r$R"
  if [ -d "$OUT" ] && [ -n "$(ls -A "$OUT" 2>/dev/null)" ]; then
    echo "r$R 目錄已有既存檔案（stale），停止——人工清點後重發" >&2; exit 24
  fi
  guard_gate "pre-r$R"
  mkdir -p "$OUT"
  env OGBENCH_DATA_DIR=/archive/cymaxwelllee/data/ogbench MUJOCO_GL=osmesa PYTHONUNBUFFERED=1 \
    LACOT_ENV=pointmaze-large-stitch-v0 LACOT_SEED=33 LACOT_DATA_SEED=-1 \
    LACOT_CONS=self LACOT_K=8 LACOT_COND=256 LACOT_CHUNK=4 LACOT_TCAP=128 \
    LACOT_STEPS1=1500 LACOT_STEPS2=2000 LACOT_ENC_OBJ=recon_ictr \
    LACOT_W_ICTR=0.2 LACOT_ICTR_SIGMA=0.05 \
    LACOT_TEACHER_MIX=0.5 LACOT_EMA_W=0.999 LACOT_WARMUP=500 \
    LACOT_DEC_START=soft LACOT_DEC_START_W=1.0 \
    LACOT_LEARNED_REFINE=0 LACOT_COND_DROP=0.1 LACOT_BC_INDEP=1 \
    LACOT_AMP=0 LACOT_COMPILE=0 LACOT_LOAD_CKPT= LACOT_CONT_TRAIN=0 LACOT_S1_FROM= \
    LACOT_GRPO_W=0 LACOT_DEV_EVAL=0 LACOT_PREREQ=0 LACOT_FLOW_PROBE=0 \
    LACOT_DIAG_DUMP=0 LACOT_DIAG_TRAIN=0 LACOT_BON_N=0 \
    LACOT_EVAL_RS=0 LACOT_EVAL_EPISODES=2 LACOT_LOG_EVERY=1000 \
    LACOT_OUT_DIR="$OUT" \
    /usr/bin/time -f 'wall_s=%e user_s=%U sys_s=%S maxrss_kb=%M' -o "$OUT/time.txt" \
    "$PY" -u experiments/scratch_lacot_rollout.py > "$OUT/stdout.log" 2> "$OUT/stderr.log"
  sha256sum "$OUT"/rollout_*.json "$OUT"/ckpt_*.pt > "$OUT/artifacts.sha256"
  JN=$(ls "$OUT"/rollout_*.json | wc -l); PN=$(ls "$OUT"/ckpt_*.pt | wc -l)
  if [ "$JN" -ne 1 ] || [ "$PN" -ne 1 ]; then echo "r$R 產物計數異常 json=$JN pt=$PN，停止" >&2; exit 25; fi
  nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv > "$OUT/gpu-after.csv"
done
echo "S0_DONE runs=[$RUNS]"
