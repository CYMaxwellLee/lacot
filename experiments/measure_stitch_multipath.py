"""量 stitch 資料裡到底有沒有「同一題多解」——V(u) 的壞路要從哪裡來（主人 2026-08-25 裁示走 (b)）。

問題：給定 (s,g)，資料集裡若只有【一條】真實的路，V(u) 看不到「同一題的好答案 vs 壞答案」
      ⇒ 它可能只學會「這條路多長」＝距離，而不是「這條路好不好」。
      stitch 的資料是隨機亂走的片段 ⇒ 主人的假說是：同一個 (s,g) 附近會有多條品質不同的路。

量法：抽 M 個同軌跡內的 (i,j) 對 ⇒ s=OBS[i], g=OBS[j], 路長=j-i。
      把 (s,g) 量化成格子（掃描 bin size ε），統計每一組：
        ① 有幾條路、來自幾條【不同軌跡】（跨軌跡才算真正的多解）
        ② 組內路長的變異 —— 我們【想要】它大（那就是好路 vs 壞路的對照）
        ③ ⚠️ 健康檢查：組內 (s,g) 的實際散布 —— 必須【小】，
           否則所謂的「多解」只是 bin 太大把不同的題混在一起（＝主人講的「重複太高要過濾」）
      ⇒ ε 的曲線同時回答：(b) 的前提成不成立、門檻該抓在哪。

⛔ 這支只讀資料、不訓練、不用 GPU。
"""
import json, os, sys
import numpy as np

OGB_DATA = os.environ.get("OGBENCH_DATA_DIR", "/archive/cymaxwelllee/data/ogbench")
ENV_NAME = os.environ.get("LACOT_ENV", "pointmaze-large-stitch-v0")
PAIRS = int(os.environ.get("LACOT_PAIRS", 300000))
SEED = int(os.environ.get("LACOT_SEED", 0))
# ⚠️ s≈g 的「原地不動」題：ε 一大就大量聚成一組，路長全是 1~5 步 ⇒ 把變異的中位數壓垮。
#    這種題對「好路 vs 壞路」沒有對照價值 ⇒ 用最短路長過濾掉。
MINLEN = int(os.environ.get("LACOT_MINLEN", 0))
# ε 掃描。0.5 是 scratch_value.py 用的 GOAL_TOL，前後各鋪幾格。
BINS = [float(x) for x in os.environ.get("LACOT_BINS", "0.25,0.5,1.0,2.0").split(",")]
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

print(f"env={ENV_NAME} pairs={PAIRS} seed={SEED} minlen={MINLEN} bins={BINS}", flush=True)
d = np.load(f"{OGB_DATA}/{ENV_NAME}.npz")
OBS = np.asarray(d["observations"], np.float32)
TERM = np.asarray(d["terminals"], bool)
N = OBS.shape[0]
ends = np.flatnonzero(TERM)
starts = np.concatenate([[0], ends[:-1] + 1])
n_traj = len(ends)
traj_id = np.empty(N, np.int64)
traj_end = np.empty(N, np.int64)
for t, (s0, e0) in enumerate(zip(starts, ends)):
    traj_id[s0:e0 + 1] = t
    traj_end[s0:e0 + 1] = e0
lo, hi = OBS.min(0), OBS.max(0)
print(f"  N={N}  trajectories={n_traj}  平均長度={N/n_traj:.1f}", flush=True)
print(f"  座標範圍 x[{lo[0]:.2f},{hi[0]:.2f}]  y[{lo[1]:.2f},{hi[1]:.2f}]", flush=True)

# ---- 抽 (i,j)：同一條軌跡內、i<j ----
rng = np.random.default_rng(SEED)
i = rng.integers(0, N, size=PAIRS * 4)
te = traj_end[i]
off = rng.integers(1, 200, size=PAIRS * 4)          # goal 距離 1..199 步
j = np.minimum(i + off, te)
keep = (j - i) >= max(1, MINLEN)
i, j = i[keep][:PAIRS], j[keep][:PAIRS]
M = len(i)
S, G, L, T = OBS[i], OBS[j], (j - i).astype(np.int64), traj_id[i]
print(f"  抽到 {M} 個 (s,g) 對；路長 中位數 {np.median(L):.0f} 範圍 [{L.min()},{L.max()}]", flush=True)

def group_key(eps):
    """把 (s,g) 量化成單一 int64 key。"""
    span = np.ceil((hi - lo) / eps).astype(np.int64) + 1
    sb = ((S - lo) / eps).astype(np.int64)
    gb = ((G - lo) / eps).astype(np.int64)
    k = sb[:, 0]
    for arr, w in ((sb[:, 1], span[1]), (gb[:, 0], span[0]), (gb[:, 1], span[1])):
        k = k * w + arr
    return k

report = {"env": ENV_NAME, "pairs": M, "seed": SEED, "minlen": MINLEN, "n_traj": int(n_traj), "bins": {}}
print("\n=== 每個 bin size ε 的結果 ===", flush=True)
for eps in BINS:
    key = group_key(eps)
    order = np.argsort(key, kind="stable")
    ks, ls, ts = key[order], L[order], T[order]
    ss, gs = S[order], G[order]
    _, gstart, gcount = np.unique(ks, return_index=True, return_counts=True)
    multi = np.flatnonzero(gcount >= 2)
    if len(multi) == 0:
        print(f"  ε={eps:<5} ⛔ 沒有任何一組有 2 條以上的路", flush=True)
        report["bins"][str(eps)] = {"n_groups": int(len(gcount)), "n_multi": 0}
        continue
    n_distinct, spread_l, spread_sg, ratio_l, dup_frac, spread_l_raw = [], [], [], [], [], []
    for gi in multi:
        a, b = gstart[gi], gstart[gi] + gcount[gi]
        tg, lg = ts[a:b], ls[a:b]
        # 🚨 同一條軌跡會貢獻多個 (i,j) 對到同一組，它們的 (s,g) 與路長幾乎一樣
        #    ⇒ 會【稀釋】變異，讓「同一題多解」看起來比實際弱。
        #    每條軌跡只留一個代表，才是「同一題的【不同】解」。
        #    ＝主人 2026-08-25「如果 s,g 重複太高，可能還是要過濾」講的那件事。
        spread_l_raw.append(float(lg.std()))
        _, first = np.unique(tg, return_index=True)
        nd = len(first)
        n_distinct.append(nd)
        dup_frac.append(1.0 - nd / len(tg))
        if nd < 2:
            spread_l.append(np.nan); ratio_l.append(np.nan); spread_sg.append(np.nan)
            continue
        lu = lg[first]
        spread_l.append(float(lu.std()))
        ratio_l.append(float(lu.max() / max(lu.min(), 1)))
        su, gu = ss[a:b][first], gs[a:b][first]
        spread_sg.append(float(max(np.abs(su - su.mean(0)).max(), np.abs(gu - gu.mean(0)).max())))
    n_distinct = np.array(n_distinct)
    cross = n_distinct >= 2                            # 跨軌跡＝真正的多解
    covered = int(gcount[multi][cross].sum())
    med = lambda arr: float(np.nanmedian(np.array(arr)[cross])) if cross.any() else 0.0
    row = {
        "n_groups": int(len(gcount)),
        "n_multi": int(len(multi)),
        "n_cross_traj": int(cross.sum()),
        "pairs_in_cross_groups": covered,
        "coverage": round(covered / M, 4),
        "median_distinct_traj": float(np.median(n_distinct[cross])) if cross.any() else 0.0,
        "median_dup_frac": med(dup_frac),
        "median_len_std_dedup": med(spread_l),
        "median_len_std_raw": med(spread_l_raw),
        "median_len_maxmin_dedup": med(ratio_l),
        "median_sg_spread": med(spread_sg),
        "global_len_std": float(L.std()),
    }
    report["bins"][str(eps)] = row
    print(f"  ε={eps:<5} 組數 {row['n_groups']:>7}  多解組 {row['n_multi']:>6}  "
          f"跨軌跡 {row['n_cross_traj']:>6}  覆蓋 {row['coverage']*100:5.1f}%", flush=True)
    print(f"          └ 每組不同軌跡(中位) {row['median_distinct_traj']:.0f}   "
          f"同軌跡重複佔 {row['median_dup_frac']*100:4.1f}%", flush=True)
    print(f"          └ 去重後路長 std {row['median_len_std_dedup']:6.1f}  "
          f"(去重前 {row['median_len_std_raw']:.1f} / 全體 {row['global_len_std']:.1f})   "
          f"max/min {row['median_len_maxmin_dedup']:.2f}   ⚠️(s,g)散布 {row['median_sg_spread']:.3f}", flush=True)

os.makedirs(OUT_DIR, exist_ok=True)
out = os.path.join(OUT_DIR, f"stitch_multipath_{ENV_NAME}_p{M}_min{MINLEN}_s{SEED}.json")
with open(out, "w") as f:
    json.dump(report, f, indent=2)
print(f"\n=> 存到 {out}", flush=True)
print("""
判讀方式：
  ⭐ 想要看到的  跨軌跡多解組多、組內路長 std 大（有好路也有壞路）、⚠️(s,g)散布 << ε
  ⛔ 警訊        (s,g)散布 接近或超過 ε ⇒ bin 太大，不同的題被混成一組
                 ＝主人講的「s,g 重複太高要過濾」那個病
""", flush=True)
