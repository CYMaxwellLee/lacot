#!/usr/bin/env python
"""段集可比性自檢（工單：G16K16 vs K32 單段保真度對比，2026-09-11）。

⛔ 不修改任何既有檔案，只 import：
  - wv_common.py（load_npz，純函式）
  - teacher_relay/run_teacher_relay.py（trk32，G16K16 側用它的 build_tasks）
  - teacher_relay/run_teacher_relay_byleg.py（trb，K32/drift_analysis 側實際用的 build_tasks
    —— drift_analysis/run_drift_analysis.py 自己 `import run_teacher_relay_byleg as trb` 後呼叫
    `trb.build_tasks(...)`，見該檔第 60 行 `sys.path.insert(0, TR_DIR)` + 第 309 行呼叫）

目的：直接跑兩邊實際呼叫的 build_tasks（不是靠讀 code 保證「應該一樣」），
用同一份 raw / ruler / seed=20260908 / n_traj=200 / rho=1.875，逐 task 比對
episode / s0 / e0 / M / wp_xy，證明兩套字典吃的是同一批 200 段。

純 numpy，不含 torch/MuJoCo，秒級，直接前景跑（不需要 slurm）。
"""
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))                      # drift_analysis_g16k16/
WV_DIR = os.path.dirname(HERE)                                          # walk_verify/
TR_DIR = os.path.join(WV_DIR, "teacher_relay")                          # walk_verify/teacher_relay/
sys.path.insert(0, WV_DIR)
sys.path.insert(0, TR_DIR)

import wv_common as wv  # noqa: E402
import run_teacher_relay as trk32  # noqa: E402 - G16K16 側 build_tasks 來源（run_teacher_relay_g16k16.py 實際 import 的模組）
import run_teacher_relay_byleg as trb  # noqa: E402 - K32/drift_analysis 側 build_tasks 來源（run_drift_analysis.py 實際 import 的模組）

DD = os.environ.get("OGBENCH_DATA_DIR", "/home/cymaxwelllee/.ogbench/data")
DATASET = "antmaze-medium-stitch-v0"
RULER = os.path.join(WV_DIR, "results", "ruler_pack.json")
SEED = 20260908
N_TRAJ = 200
RHO = 1.875

print(f"=== check_task_comparability ===")
print(f"raw: {DD}/{DATASET}-val.npz")
print(f"ruler: {RULER}")
print(f"seed={SEED} n_traj={N_TRAJ} rho={RHO}")
print(f"trk32.build_tasks source: {trk32.__file__}")
print(f"trb.build_tasks   source: {trb.__file__}")
print(f"trk32.DELTA_SUB={trk32.DELTA_SUB}  trb.DELTA_SUB={trb.DELTA_SUB}")

with open(RULER) as f:
    ruler = json.load(f)
raw = wv.load_npz(DD, DATASET, "val")

tasks_g16k16, skipped_g16k16 = trk32.build_tasks(raw, N_TRAJ, SEED, ruler, RHO)
tasks_k32, skipped_k32 = trb.build_tasks(raw, N_TRAJ, SEED, ruler, RHO)

print(f"\ntasks_g16k16(trk32.build_tasks): n={len(tasks_g16k16)} skipped={len(skipped_g16k16)}")
print(f"tasks_k32   (trb.build_tasks)  : n={len(tasks_k32)} skipped={len(skipped_k32)}")

mismatches = []
if len(tasks_g16k16) != len(tasks_k32):
    mismatches.append(f"n_tasks 不同: {len(tasks_g16k16)} vs {len(tasks_k32)}")
else:
    for i, (ta, tb) in enumerate(zip(tasks_g16k16, tasks_k32)):
        for field in ("episode", "s0", "e0", "T", "n_chunks", "M"):
            if ta[field] != tb[field]:
                mismatches.append(f"task[{i}].{field}: g16k16-side={ta[field]} k32-side={tb[field]}")
        if not np.array_equal(ta["wp_xy"], tb["wp_xy"]):
            mismatches.append(f"task[{i}].wp_xy 不同")

# 段 id / 起點 hash：逐 task 的 (episode, s0, e0, M) 串起來雜湊，兩邊各自算一份
def task_hash(tasks):
    h = hashlib.sha256()
    for t in tasks:
        h.update(f"{t['episode']},{t['s0']},{t['e0']},{t['M']}|".encode())
    return h.hexdigest()

hash_g16k16 = task_hash(tasks_g16k16)
hash_k32 = task_hash(tasks_k32)
print(f"\ntask-set sha256 (g16k16-side build_tasks call): {hash_g16k16}")
print(f"task-set sha256 (k32-side    build_tasks call): {hash_k32}")
print(f"hash 一致: {hash_g16k16 == hash_k32}")

episodes_g16k16 = [t["episode"] for t in tasks_g16k16]
episodes_k32 = [t["episode"] for t in tasks_k32]
print(f"\n前 10 條 episode id（g16k16 側）: {episodes_g16k16[:10]}")
print(f"前 10 條 episode id（k32 側）   : {episodes_k32[:10]}")

n_chunks_all = set(t["n_chunks"] for t in tasks_g16k16)
print(f"\nn_chunks 集合（應該全部一致，episode 長度固定 201 步 -> 50）: {n_chunks_all}")

if mismatches or hash_g16k16 != hash_k32:
    print(f"\n⛔⛔⛔ 段集不可比：{len(mismatches)} 項欄位不一致，hash {'一致' if hash_g16k16==hash_k32 else '不一致'}")
    for m in mismatches[:20]:
        print("   -", m)
    result = dict(status="FAIL", n_mismatches=len(mismatches), mismatches=mismatches[:50],
                  hash_g16k16=hash_g16k16, hash_k32=hash_k32)
else:
    print(f"\n✅ 段集可比性通過：{len(tasks_g16k16)} 條 task 逐欄位一致（episode/s0/e0/T/n_chunks/M/wp_xy），"
          f"hash 相同 —— 兩套字典吃的是同一批 200 段。")
    result = dict(status="PASS", n_tasks=len(tasks_g16k16),
                  hash_g16k16=hash_g16k16, hash_k32=hash_k32,
                  episodes_first10=episodes_g16k16[:10])

out_path = os.path.join(HERE, "results", "task_comparability_20260911.json")
with open(out_path, "w") as f:
    json.dump(result, f, indent=2)
print(f"\nsaved: {out_path}")
