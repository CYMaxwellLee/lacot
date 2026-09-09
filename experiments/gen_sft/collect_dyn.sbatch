#!/usr/bin/env bash
#SBATCH -p admin
#SBATCH -A it
#SBATCH -q great-mage
#SBATCH --nodelist=zeldajr
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH -o /home/cymaxwelllee/Projects/lacot-dyn/experiments/gen_sft/logs/%x-%j.out
# 主人裁示①②：Slurm dependency 鏈最後一棒——等全部 6 顆訓練 + 6 顆 eval 都
# afterok 完工才跑，純讀 JSON 收表算判準（analyze_dyn.py），不需要任何運算資源。
# 提交時掛 --dependency=afterok:<全部12個jobid逗號分隔>，用 sbatch --wait 拿到
# 「全部完工」的單一阻塞點，不在前景寫 squeue 輪詢迴圈。
set -eu
WT=/home/cymaxwelllee/Projects/lacot-dyn
PY=/home/cymaxwelllee/venvs/lacot-rocm/bin/python
cd "$WT"
echo "worktree: $(git -C "$WT" rev-parse --abbrev-ref HEAD) $(git -C "$WT" rev-parse HEAD)"
echo "node: $(hostname)  collecting dyn results"
"$PY" -u experiments/gen_sft/analyze_dyn.py
