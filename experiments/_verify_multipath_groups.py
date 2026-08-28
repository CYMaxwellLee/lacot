"""獨立驗證 measure_stitch_multipath 的分組：印出實際的組，人眼看它是不是真的「同一題多解」。
⛔ 不共用 measure 那支的 group_key —— 故意用另一種寫法（tuple key + dict），
   兩邊算出來的組大小分布若對不上，就是其中一支壞了。"""
import os, numpy as np
from collections import defaultdict

OGB = os.environ.get("OGBENCH_DATA_DIR", "/archive/cymaxwelllee/data/ogbench")
ENV = os.environ.get("LACOT_ENV", "pointmaze-large-stitch-v0")
EPS = float(os.environ.get("LACOT_EPS", 1.0))
PAIRS = int(os.environ.get("LACOT_PAIRS", 300000))

d = np.load(f"{OGB}/{ENV}.npz")
OBS = np.asarray(d["observations"], np.float32); TERM = np.asarray(d["terminals"], bool)
N = OBS.shape[0]; ends = np.flatnonzero(TERM); starts = np.concatenate([[0], ends[:-1] + 1])
traj_id = np.empty(N, np.int64); traj_end = np.empty(N, np.int64)
for t, (s0, e0) in enumerate(zip(starts, ends)):
    traj_id[s0:e0+1] = t; traj_end[s0:e0+1] = e0
lo = OBS.min(0)

rng = np.random.default_rng(0)
i = rng.integers(0, N, size=PAIRS*2); te = traj_end[i]
off = rng.integers(1, 200, size=PAIRS*2); j = np.minimum(i + off, te)
keep = j > i; i, j = i[keep][:PAIRS], j[keep][:PAIRS]
S, G, L, T = OBS[i], OBS[j], (j-i).astype(np.int64), traj_id[i]
print(f"{ENV}  ε={EPS}  pairs={len(i)}")

# ---- 獨立分組：tuple key + dict（跟 measure 那支的 int64 編碼無關）----
groups = defaultdict(list)
sb = np.floor((S - lo)/EPS).astype(int); gb = np.floor((G - lo)/EPS).astype(int)
for n in range(len(i)):
    groups[(sb[n,0], sb[n,1], gb[n,0], gb[n,1])].append(n)

sizes = np.array([len(v) for v in groups.values()])
print(f"  組數 {len(groups)}   多解組(>=2) {int((sizes>=2).sum())}   最大組 {sizes.max()}")

# 去重後 std 的分布（跟 measure 那支同定義）
stds, ndists = [], []
for v in groups.values():
    if len(v) < 2: continue
    vt = T[v]; _, first = np.unique(vt, return_index=True)
    if len(first) < 2: continue
    lu = L[np.array(v)[first]]
    stds.append(lu.std()); ndists.append(len(first))
stds = np.array(stds); ndists = np.array(ndists)
print(f"  跨軌跡組 {len(stds)}   去重後 std: median {np.median(stds):.2f}  "
      f"mean {stds.mean():.2f}  p10 {np.percentile(stds,10):.2f}  p90 {np.percentile(stds,90):.2f}")
print(f"  每組不同軌跡數: median {np.median(ndists):.0f}  max {ndists.max()}")
print(f"  ⚠️ std==0 的組佔 {float((stds==0).mean())*100:.1f}%   std<5 佔 {float((stds<5).mean())*100:.1f}%")

# ---- 印實際的組給人眼看 ----
cand = [v for v in groups.values() if len(v) >= 2 and len(np.unique(T[v])) >= 3]
print(f"\n=== 隨機 4 個跨軌跡組（>=3 條軌跡）===")
for v in [cand[k] for k in rng.choice(len(cand), size=min(4, len(cand)), replace=False)]:
    print("  ---")
    for n in v[:8]:
        print(f"    traj {T[n]:>5}  s=({S[n,0]:6.2f},{S[n,1]:6.2f})  "
              f"g=({G[n,0]:6.2f},{G[n,1]:6.2f})  路長 {L[n]:>4}")
