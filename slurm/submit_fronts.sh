#!/bin/bash
# 把 exp_three_fronts.py 的若干格 × 若干 seed 送到叢集，一格一張 GPU。
#
# ⚠️ TIME 預設 15 分鐘，⛔ 不要設成 partition 上限（實測一格 2.5~3.5 分）。
#    理由（2026-08-09 踩過）：Slurm 的 backfill 假設 job 會用滿宣告時間，
#    宣告越久越插不進空檔。宣告貼近實際用量，backfill 才會動。
#
# ⚠️ 一格一個 seed —— ⛔ 不要只跑一個 seed 就下結論（同一個病 08-07 一晚咬四次）。
#
# 用法：
#   MODES="tok_000 tok_r00" SEEDS="0 1 2" ./slurm/submit_fronts.sh
set -euo pipefail

PROJ="$HOME/Projects/lacot"
MODES="${MODES:-}"
SEEDS="${SEEDS:-0 1 2}"
TIME="${TIME:-00:15:00}"
CPUS="${CPUS:-4}"

mkdir -p "$PROJ/slurm_outputs" "$PROJ/results"

# ⛔ 只送到「本機已經有 venv 和資料」的節點
echo "檢查各節點就緒狀態…"
READY=()
for h in jasmine lady pocahontas moana; do
    # ⚠️ 一定要真的 import torch —— 只檢查 python 檔在不在會給假綠燈
    #    （`uv venv` 第一步就建好 symlink，套件還在下載）。2026-08-23 踩過。
    ok=$(timeout 60 ssh -o ConnectTimeout=6 -o BatchMode=yes "$h" \
        '/archive/cymaxwelllee/LaCoT/.venv/bin/python -c "import torch, ogbench" >/dev/null 2>&1 \
         && [ -f /archive/cymaxwelllee/data/ogbench/pointmaze-medium-navigate-v0.npz ] && echo yes' 2>/dev/null || true)
    if [ "$ok" = "yes" ]; then READY+=("$h"); printf "  %-11s ready\n" "$h"
    else printf "  %-11s 還沒好，跳過\n" "$h"; fi
done
[ ${#READY[@]} -eq 0 ] && { echo "⛔ 沒有任何節點就緒"; exit 1; }

# ⚠️ 一個 job 不能同時跨 partition 指定 nodelist。jasmine 在 ada-lite、其餘在 turing。
# 🚨 --nodelist 是「這些節點【全部】都要進 allocation」，⛔ 不是「從裡面挑一台」。
#    2026-08-23 踩到：nodelist=lady,moana,pocahontas ＋ --gres=gpu:1（＝per node）
#    ⇒ 一個 job 佔【三台各一張 GPU】、開三個 task，而 batch script 只在第一台跑，
#    另外兩張卡整段閒著被鎖住。要「從裡面挑一台」必須配 --nodes=1 --ntasks=1。
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

# 要補跑時（有些格已經有結果）用 PAIRS 指定，一行一個 "mode seed"；
# 沒給就跑 MODES × SEEDS 的全部組合。
if [ -n "${PAIRS:-}" ]; then
    mapfile -t CELLS < <(printf '%s\n' "$PAIRS" | sed '/^[[:space:]]*$/d')
else
    CELLS=()
    for m in $MODES; do for sd in $SEEDS; do CELLS+=("$m $sd"); done; done
fi

n=0
for cell in "${CELLS[@]}"; do
    read -r m sd <<<"$cell"
    pt="${PARTS[$((n % ${#PARTS[@]}))]}"
    jid=$(sbatch --parsable \
        --job-name="fr-$m-s$sd" \
        --partition="$pt" \
        --nodelist="${GROUP[$pt]}" \
        --nodes=1 --ntasks=1 \
        --gres=gpu:1 \
        --cpus-per-task="$CPUS" \
        --time="$TIME" \
        --output="$PROJ/slurm_outputs/%j-$m-s$sd.out" \
        --error="$PROJ/slurm_outputs/%j-$m-s$sd.err" \
        --export=ALL,MODE="$m",LACOT_SEED="$sd" \
        "$PROJ/slurm/run_fronts.sh")
    printf "  送出 %-10s seed %s  job %-8s (%s)\n" "$m" "$sd" "$jid" "$pt"
    n=$((n + 1))
done
echo
echo "共送出 $n 個。看進度：squeue -u \$USER"
