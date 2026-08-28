"""訓練時 u 被要求表示多長的路，考試時要求多長 —— 兩把尺分開量。

🚨 起因：主人 2026-08-28 的直覺「我們目前都是要一次就把整條 path 一次規劃好」。
   `[實測]` 環境給的是【最終目標】，整集固定（scratch_lacot_rollout.py:352-358），
   而訓練時 goal 在【同一條軌跡內】均勻抽（:78-92，官方 GCBC 抽法）
   ⇒ 訓練時 u 從來沒有表示過跨軌跡的長路。

⚠️ 兩把尺量的是兩個不同的病，⛔ 不要換算成同一個單位：
   ① 格數尺  cond 的 OOD ⇒ flow / head 的病（它們吃的是 (s,g)）
   ② 步數尺  表示跨幅的 OOD ⇒ decoder 的病
      （T_CAP 重取樣讓長路【不是更多點】，是同樣 128 點【攤得更開】
       ⇒ decoder 外推的是「相鄰點間距」，⛔ 不是「序列長度」）

⛔ 抽樣邏輯必須跟主線同一份，⛔ 不准另寫 —— 量的必須是訓練實際看到的分布。
"""
import os, sys, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lacot import dev_eval as DE

OGB_DATA = os.environ.get("OGBENCH_DATA_DIR", "/archive/cymaxwelllee/data/ogbench")
ENV_NAME = os.environ.get("LACOT_ENV", "pointmaze-medium-stitch-v0")
N_SAMPLE = int(os.environ.get("LACOT_SPAN_N", 50000))
CHUNK = int(os.environ.get("LACOT_CHUNK", 4))

d = np.load(f"{OGB_DATA}/{ENV_NAME}.npz")
OBS = np.asarray(d["observations"], np.float32)
TERM = np.asarray(d["terminals"], bool)
N = OBS.shape[0]
ends = np.flatnonzero(TERM); starts = np.concatenate([[0], ends[:-1] + 1])
traj_end = np.empty(N, np.int64)
for s0, e0 in zip(starts, ends):
    traj_end[s0:e0 + 1] = e0
print(f"{ENV_NAME}: {N} row / {len(ends)} 條軌跡，最長 {int((ends-starts+1).max())} 步", flush=True)

# ── 用主線那一份抽法抽 (r, gr) ────────────────────────────────
rng = np.random.default_rng(0)
rows, goals = [], []
while len(rows) < N_SAMPLE:
    r = int(rng.integers(0, N)); te = int(traj_end[r])
    if te - r < CHUNK:
        continue
    _d = rng.random()
    gr = int(round(min(r + 1, te) * _d + te * (1 - _d)))
    gr = max(gr, min(r + CHUNK, te))
    rows.append(r); goals.append(gr)
rows, goals = np.array(rows), np.array(goals)

# ── 尺②：步數（表示跨幅）───────────────────────────────────
step_gap = goals - rows
print("\n=== 尺② 步數：訓練時 u 被要求跨多少步 ===")
for q in (50, 90, 99, 100):
    print(f"  p{q:<3} {np.percentile(step_gap, q):>7.1f} 步")

# ── 尺①：路徑實際長度（訓練 vs 考試，同單位才比得了）─────────
# ⭐ 訓練那條：沿【軌跡實際走過】的路累積 —— 那正是 u 被要求表示的東西
seg = np.linalg.norm(OBS[1:] - OBS[:-1], axis=1)
cum = np.concatenate([[0.0], np.cumsum(seg)])
train_len = cum[goals] - cum[rows]
print("\n=== 尺① 路徑長度：訓練時 u 被要求表示多長的路 ===")
for q in (50, 90, 99, 100):
    print(f"  p{q:<3} {np.percentile(train_len, q):>7.2f}")

out = dict(env=ENV_NAME, n=N_SAMPLE,
           step_gap={f"p{q}": float(np.percentile(step_gap, q)) for q in (50, 90, 99, 100)},
           train_path_len={f"p{q}": float(np.percentile(train_len, q)) for q in (50, 90, 99, 100)})

# ── 考試那條：dev 尺的題，用 BFS 最短路 × 格寬 ────────────────
import ogbench
env, _, _ = ogbench.make_env_and_datasets(ENV_NAME, dataset_dir=OGB_DATA)
tasks = DE.build_dev_tasks(env, n_per_tier=100, n_tiers=3, seed=0, min_dist=3)
# 🚨 2026-08-28 修：舊版取【頭兩個可通行格】的距離當格寬 —— cells 是 row-major 列舉，
#    只有在「同一列相鄰兩行都可通行」時才是一格寬。`[實測]` 把某列換成 [1,0,1,0,1,0,1]
#    ⇒ cell_w 變兩倍，⛔ 而且不會叫。
#    ⚠️ 而 `≈路長 = bfs_dist × cell_w` 正是「最難那層 100% 超出訓練 p99」與 DELTA_SUB=7.5
#      的來源 ⇒ 這個常數錯掉，那兩個結論一起錯。
#    ⭐ `[實測 2026-08-28]` 影響範圍：medium 與 large 的頭兩個可通行格【剛好】是 (1,1)(1,2)
#      ⇒ 舊版在這兩張圖上算出來是對的 4.00 ⇒ ⛔ 既有的 DELTA_SUB=7.5 與 tier 結論【沒有】被咬到。
#      🚨 但 giant 的頭兩格是 (1,1)(1,3) ⇒ 舊版算出 8.00（翻倍）⇒ 換到 giant 就會靜默地錯。
#    ⭐ 正確版本（全對距離取最小）本來就在 scratch_lacot_rollout.py 裡 —— 兩份不一致
#      ⇒ 抽進 lacot/dev_eval.cell_width 共用，⛔ 不留兩份。
cells = DE._passable_cells(env)
cell_w = DE.cell_width(env)   # 相鄰兩格的間距
print(f"\n=== 尺① 考試：dev 題的最短路長度（格寬 {cell_w:.2f}）===")
print(f"{'層':>4} {'題數':>5} {'BFS 格':>8} {'≈路長':>8}   {'超出訓練 p99 的比例':>18}")
p99 = float(np.percentile(train_len, 99))
out["train_len_p99"] = p99; out["cell_w"] = cell_w; out["dev"] = {}
for t in sorted({x["tier"] for x in tasks}):
    sub = [x for x in tasks if x["tier"] == t]
    b = np.array([x["bfs_dist"] for x in sub], float)
    L = b * cell_w
    frac = float((L > p99).mean())
    print(f"  t{t} {len(sub):>6} {np.median(b):>8.1f} {np.median(L):>8.2f}   {frac:>17.1%}")
    out["dev"][f"t{int(t)}"] = dict(n=len(sub), bfs_med=float(np.median(b)),
                                    len_med=float(np.median(L)), frac_beyond_train_p99=frac)

print(f"\n⇒ 訓練 p99 的路長是 {p99:.2f}；超過它的題，u 要表示的東西是訓練時沒見過的跨度。")

os.makedirs("results", exist_ok=True)
pth = f"results/spangap_{ENV_NAME.replace('pointmaze-','').replace('-v0','')}.json"
json.dump(out, open(pth, "w"), ensure_ascii=False, indent=1)
print(f"寫入 {pth}", flush=True)
