#!/usr/bin/env bash
# GPU golden A/A/B：A=main、B=受測分支的 worktree、同一張卡序跑。
# A1 vs A2 = 這張卡的 run-to-run 決定性（不成立則整個比對無效、直接 INCONCLUSIVE）
# A1 vs B  = 分支零差判決
# 用法：BRANCH_ROOT=<受測 worktree 路徑> [GOLD_OUT=<輸出目錄>] bash gold_ab_wrap.sh
#   （先 git worktree add <路徑> <分支>；⛔ 別在主 repo 切分支 — 排隊 job runtime 讀主 repo）
set -u
SP=$HOME/Projects/lacot/slurm/goldab
G_OUT=${GOLD_OUT:-$HOME/Projects/lacot/results/goldab}
G="$G_OUT"; rm -rf "$G"; mkdir -p "$G/A1" "$G/A2" "$G/B"
MAIN=$HOME/Projects/lacot/experiments/scratch_lacot_rollout.py
BRANCH_ROOT=${BRANCH_ROOT:?⛔ 要給 BRANCH_ROOT=受測 worktree 路徑}
BRANCH=$BRANCH_ROOT/experiments/scratch_lacot_rollout.py

echo "== git 狀態快照 =="
git -C "$HOME/Projects/lacot" rev-parse --abbrev-ref HEAD HEAD
git -C "$BRANCH_ROOT" rev-parse --abbrev-ref HEAD HEAD

bash "$SP/run_gold_ab_gpu.sh" "$MAIN"   "$G/A1" > "$G/A1.log" 2>&1; echo "A1 rc=$?"
bash "$SP/run_gold_ab_gpu.sh" "$MAIN"   "$G/A2" > "$G/A2.log" 2>&1; echo "A2 rc=$?"
bash "$SP/run_gold_ab_gpu.sh" "$BRANCH" "$G/B"  > "$G/B.log"  2>&1; echo "B  rc=$?"

echo "== ckpt/json sha256 =="
for d in A1 A2 B; do (cd "$G/$d" && sha256sum ckpt_*.pt rollout_*.json 2>/dev/null | sed "s|^|$d |"); done

sha_of(){ (cd "$G/$1" && sha256sum ckpt_*.pt 2>/dev/null | awk '{print $1}' | sort | tr '\n' '+'); }
J_of(){ (cd "$G/$1" && sha256sum rollout_*.json 2>/dev/null | awk '{print $1}' | sort | tr '\n' '+'); }
A1=$(sha_of A1); A2=$(sha_of A2); B=$(sha_of B)
JA1=$(J_of A1); JA2=$(J_of A2); JB=$(J_of B)

echo "== stdout diff（A1 vs A2）=="; diff -q "$G/A1.log" "$G/A2.log" && echo "A1==A2 stdout: SAME" || echo "A1!=A2 stdout: DIFFER"
echo "== stdout diff（A1 vs B）=="; diff "$G/A1.log" "$G/B.log" | head -20

if [ -z "$A1" ] || [ "$A1" != "$A2" ] || [ "$JA1" != "$JA2" ]; then
  echo "VERDICT: INCONCLUSIVE — 這張卡 run-to-run 不決定（A1 vs A2 不同或缺檔），改 moana 重跑"
elif [ "$A1" == "$B" ] && [ "$JA1" == "$JB" ]; then
  echo "VERDICT: PASS — 受測分支對 pointmaze 於 GPU 亦零差（ckpt+json 全同）"
else
  echo "VERDICT: FAIL — 受測分支在 GPU 路徑上與 main 有差，禁 merge、回設計"
fi
