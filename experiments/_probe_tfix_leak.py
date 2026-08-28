"""🚨 T_FIX 變大會不會讓 8/25 那個洩漏用另一種形式回來？

make_segments 用 np.linspace(r, g, T_FIX).round() 取索引。
⇒ 當 L+1 < T_FIX 時，索引會【重複】⇒ 不重複點的個數 ＝ min(L+1, T_FIX)
⇒ 那個數字直接就是長度。

這裡量的是：拿「不重複點的個數」當分數，同題內排序能拿多少。
⛔ 高 ⇒ 洩漏確實回來了，T_FIX 越大越嚴重。
"""
import os, numpy as np
D = os.environ.get("OGBENCH_DATA_DIR", "/archive/cymaxwelllee/data/ogbench")
ENV = os.environ.get("LACOT_ENV", "pointmaze-medium-stitch-v0")
EPS = 0.5

d = np.load(f"{D}/{ENV}-val.npz")
obs = np.asarray(d["observations"], np.float32); term = np.asarray(d["terminals"], bool)
n = obs.shape[0]; ends = np.flatnonzero(term)
te = np.empty(n, np.int64); tid = np.empty(n, np.int64)
st = np.concatenate([[0], ends[:-1] + 1])
for t, (s0, e0) in enumerate(zip(st, ends)):
    te[s0:e0+1] = e0; tid[s0:e0+1] = t

rng = np.random.default_rng(1234)           # ⚠️ 跟 exp_value_u 的題庫同 seed
got_r, got_g, have = [], [], 0
while have < 20000:
    r = rng.integers(0, n, size=20000); e = te[r]
    off = rng.integers(20, 200, size=20000)
    g = np.minimum(r + off, e); g = np.maximum(g, np.minimum(r + 20, e))
    ok = (g - r) >= 20
    got_r.append(r[ok]); got_g.append(g[ok]); have += int(ok.sum())
qr = np.concatenate(got_r)[:20000]; qg = np.concatenate(got_g)[:20000]
qL = (qg - qr).astype(np.int64); qT = tid[qr]
lo = obs.min(0); span = np.ceil((obs.max(0) - lo) / EPS).astype(np.int64) + 1
sb = ((obs[qr] - lo) / EPS).astype(np.int64); gb = ((obs[qg] - lo) / EPS).astype(np.int64)
key = sb[:, 0]
for arr, w in ((sb[:, 1], span[1]), (gb[:, 0], span[0]), (gb[:, 1], span[1])):
    key = key * w + arr
order = np.argsort(key, kind="stable")
ks, L_, T_ = key[order], qL[order], qT[order]
_, gs, gc = np.unique(ks, return_index=True, return_counts=True)
pairs = []
for gi in np.flatnonzero(gc >= 2):
    a, b = gs[gi], gs[gi] + gc[gi]
    _, first = np.unique(T_[a:b], return_index=True)
    if len(first) < 2: continue
    fi = a + first
    for x in range(len(fi)):
        for y in range(x+1, len(fi)):
            pairs.append((fi[x], fi[y]))
pairs = np.array(pairs)
La, Lb = L_[pairs[:, 0]], L_[pairs[:, 1]]
m = np.abs(La - Lb) >= 1
print(f"題庫配對 {len(pairs)}（有效 {int(m.sum())}）  L 中位 {np.median(L_):.0f}  "
      f"p10 {np.percentile(L_,10):.0f}  p90 {np.percentile(L_,90):.0f}")
print(f"\n{'T_FIX':>6} {'洩漏排序':>9} {'L+1<T 的路佔比':>16}")
for T in (32, 64, 128, 201):
    nu_a, nu_b = np.minimum(La + 1, T), np.minimum(Lb + 1, T)   # ＝不重複點的個數
    # 分數：點數少 ⇒ 路短 ⇒ 分高
    sa, sb_ = -nu_a.astype(float), -nu_b.astype(float)
    rnd = np.random.default_rng(0).random(len(sa)) - 0.5
    sa2, sb2 = sa + rnd * 1e-9, sb_ - rnd * 1e-9                # tie 隨機拆（⛔ 不要全算對）
    ok = (sa2[m] > sb2[m]) == (La[m] < Lb[m])
    frac = float((L_ + 1 < T).mean())
    print(f"{T:>6} {ok.mean():>9.3f} {frac*100:>15.1f}%")
