#!/bin/bash
# 把 anti-collapse 的 15 個變體用 job array 送出去，一格一張 GPU。
#
# 配置（主人 2026-08-23 裁示，見 memory cluster-layout-code-here-data-there）：
#   code   -> ~/Projects/lacot            一份，走 home（每台都掛得到）
#   venv   -> /archive/cymaxwelllee/LaCoT/.venv   各台一份，⛔ 不走 NFS
#   資料   -> /archive/cymaxwelllee/data/ogbench  各台一份
#
# ⚠️ TIME 預設 40 分鐘，⛔ 不要設成 partition 上限。
#    理由（2026-08-09 踩過）：Slurm 的 backfill 假設 job 會用滿宣告時間，
#    宣告 12 小時就等於告訴它「12 小時內沒有空 GPU」，連兩分鐘的診斷都
#    插不進空檔，全部排在隊伍後面。宣告貼近實際用量，backfill 才會動。
#
# 用法：
#   ./slurm/submit_variants.sh                 # 全部 15 個
#   VARIANTS="byol barlow" ./slurm/submit_variants.sh   # 只跑指定的
#   PARTITION=turing ./slurm/submit_variants.sh
set -euo pipefail

PROJ="$HOME/Projects/lacot"
ALL="byol_h32 byol_h64 byol_h128 byol_h256 byol_h1024 ema_m99 ema_m996 ema_m999 cema_m99 cema_m996 cema_m999"
VARIANTS="${VARIANTS:-$ALL}"
PARTITION="${PARTITION:-turing,ada-lite}"
TIME="${TIME:-00:40:00}"
CPUS="${CPUS:-4}"

mkdir -p "$PROJ/slurm_outputs" "$PROJ/results"

# ⛔ 只送到「本機已經有 venv 和資料」的節點 —— 送到沒有的會直接失敗浪費排程
echo "檢查各節點就緒狀態…"
READY=()
for h in jasmine lady pocahontas moana; do
    # ⚠️ 一定要真的 import torch —— 只檢查 python 檔在不在會給假綠燈：
    #    `uv venv` 第一步就建好 bin/python 的 symlink，套件卻還在下載。
    #    2026-08-23 就是這樣四台全報 ready，其實三台的 venv 只有 88K。
    ok=$(timeout 60 ssh -o ConnectTimeout=6 -o BatchMode=yes "$h" \
        '/archive/cymaxwelllee/LaCoT/.venv/bin/python -c "import torch, ogbench" >/dev/null 2>&1 \
         && [ -f /archive/cymaxwelllee/data/ogbench/pointmaze-medium-navigate-v0.npz ] && echo yes' 2>/dev/null || true)
    if [ "$ok" = "yes" ]; then READY+=("$h"); printf "  %-11s ready\n" "$h"
    else printf "  %-11s 還沒好，跳過\n" "$h"; fi
done
[ ${#READY[@]} -eq 0 ] && { echo "⛔ 沒有任何節點就緒"; exit 1; }

# ⚠️ 一個 job 不能同時跨 partition 指定 nodelist（sbatch 會回
#    "Requested node configuration is not available"）。jasmine 在 ada-lite、
#    其餘在 turing，所以要按 partition 分組，變體輪流丟到各組。
declare -A NODE_PART=( [jasmine]=ada-lite [lady]=turing [pocahontas]=turing [moana]=turing )
declare -A GROUP
for h in "${READY[@]}"; do
    pt="${NODE_PART[$h]}"
    GROUP[$pt]="${GROUP[$pt]:+${GROUP[$pt]},}$h"
done
# ⚠️ 分配要按【GPU 張數】加權，⛔ 不是每個 partition 輪一次。
#    2026-08-23 踩過：jasmine（ada-lite）只有 3 張、turing 那邊有 8 張，
#    平均輪流等於把一半的 job 塞進 3 張卡，另外 8 張在旁邊閒著。
declare -A NGPU
for h in "${READY[@]}"; do
    g=$(sinfo -h -n "$h" -o "%G" 2>/dev/null | head -1 | grep -oE 'gpu:[^,]*' | grep -oE '[0-9]+$' | head -1)
    NGPU[${NODE_PART[$h]}]=$(( ${NGPU[${NODE_PART[$h]}]:-0} + ${g:-1} ))
done
PARTS=()
for pt in "${!GROUP[@]}"; do
    for ((i = 0; i < ${NGPU[$pt]:-1}; i++)); do PARTS+=("$pt"); done
done
echo "  分組：$(for pt in "${!GROUP[@]}"; do printf "%s=[%s]x%s " "$pt" "${GROUP[$pt]}" "${NGPU[$pt]:-?}"; done)"

n=0
for v in $VARIANTS; do
    pt="${PARTS[$((n % ${#PARTS[@]}))]}"      # 變體輪流分配到各 partition
    jid=$(sbatch --parsable \
        --job-name="lacot-$v" \
        --partition="$pt" \
        --nodelist="${GROUP[$pt]}" \
        --nodes=1 --ntasks=1 \
        --gres=gpu:1 \
        --cpus-per-task="$CPUS" \
        --time="$TIME" \
        --output="$PROJ/slurm_outputs/%j-$v.out" \
        --error="$PROJ/slurm_outputs/%j-$v.err" \
        --export=ALL,VARIANT="$v" \
        "$PROJ/slurm/run_variant.sh")
    printf "  送出 %-12s job %-8s (%s)\n" "$v" "$jid" "$pt"
    n=$((n + 1))
done
echo
echo "共送出 $n 個，節點：$(IFS=,; echo "${READY[*]}")"
echo "看進度：squeue -u \$USER    |    收結果：python3 $PROJ/slurm/collect.py"
