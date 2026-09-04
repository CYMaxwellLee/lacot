"""intent 底座 smoke：fake 佔據圖上驗 hindsight 提取／選路／重採樣（分鐘級、無 GPU）。"""
import numpy as np

from lacot.intent import (traj_to_cells, jitter_rate, route_cells,
                          cells_to_anchors, anchors_resample,
                          hindsight_intent, route_intent)

# fake 迷宮：7x7、外牆＋中央十字缺口（有兩條繞法 ⇒ 最短路唯一性不假設）
occ = np.zeros((7, 7), bool)
occ[1:6, 1:6] = True
occ[3, 2] = occ[3, 4] = False          # 中列留 (3,1)(3,3)(3,5) 三個洞

CELL = 1.0
xy_to_cell = lambda xy: (int(round(xy[1])), int(round(xy[0])))   # (x,y)→(i,j) 故意轉置，驗注入
cell_to_xy = lambda c: np.array([c[1], c[0]], np.float64)

ok = 0

# ① 選路：對角 (1,1)→(5,5)，路線存在、兩端正確、全在自由格
p = route_cells(occ, (1, 1), (5, 5))
assert p is not None and p[0] == (1, 1) and p[-1] == (5, 5), f"route 兩端錯：{p[:2]}…{p[-2:]}"
assert all(occ[c] for c in p), "route 穿牆"
assert all(abs(a[0]-b[0]) + abs(a[1]-b[1]) == 1 for a, b in zip(p, p[1:])), "route 不連續"
ok += 1; print(f"① route_cells OK：len={len(p)} 經過中列洞 {[c for c in p if c[0]==3]}")

# ② 不可達：把三個洞全堵死 ⇒ None
occ2 = occ.copy(); occ2[3, 1] = occ2[3, 3] = occ2[3, 5] = False
assert route_cells(occ2, (1, 1), (5, 5)) is None, "不可達應回 None"
ok += 1; print("② 不可達回 None OK")

# ③ hindsight：合成軌跡（沿 x 再沿 y、每格 4 個取樣點＋小噪聲）→ cell 序列應等於 L 型路
rng = np.random.default_rng(0)
seg1 = [(x, 1.0) for x in np.linspace(1, 5, 17)]
seg2 = [(5.0, y) for y in np.linspace(1, 5, 17)]
traj = np.asarray(seg1 + seg2, np.float64) + rng.normal(0, 0.05, (34, 2))
cells = traj_to_cells(traj, xy_to_cell)
want = [(1, x) for x in range(1, 6)] + [(y, 5) for y in range(2, 6)]
assert cells[0] == (1, 1) and cells[-1] == (5, 5), f"hindsight 兩端錯：{cells[0]}…{cells[-1]}"
jit = jitter_rate(cells)
assert jit <= 0.2, f"低噪聲軌跡抖動率異常高 {jit:.2f}（提取邏輯有問題）"
extra = len(cells) - len(want)
assert 0 <= extra <= 2, f"cell 數偏離 L 型路太多：{len(cells)} vs {len(want)}"
ok += 1; print(f"③ hindsight OK：{len(cells)} cells（L 型 {len(want)}+{extra} 抖動格）jitter {jit:.2f}")

# ④ 重採樣：端點保持、點都在折線上、弧長大致均勻、K=1 tile
#    ⚠️ cv 門檻 0.15 不是 0.05 —— 90° 轉角處「弧長均勻的兩點」歐氏步距天生縮 ~30%
#    （跨角直線 < 沿折線弧長），這是折線幾何不是插值 bug；真正嚴的判準是下面「在折線上」。
A = cells_to_anchors(cells, cell_to_xy)
R = anchors_resample(A, 16)
assert R.shape == (16, 2)
assert np.allclose(R[0], A[0]) and np.allclose(R[-1], A[-1]), "重採樣端點漂了"

def _dist_to_polyline(q, P):
    best = np.inf
    for a, b in zip(P[:-1], P[1:]):
        d = b - a; L2 = float(d @ d)
        t = 0.0 if L2 == 0 else float(np.clip((q - a) @ d / L2, 0, 1))
        best = min(best, float(np.linalg.norm(q - (a + t * d))))
    return best

offline = max(_dist_to_polyline(q, A) for q in R)
assert offline < 1e-9, f"重採樣點離開折線 {offline:.2e}（插值邏輯錯）"
step = np.linalg.norm(np.diff(R, axis=0), axis=1)
cv = step.std() / step.mean()
assert cv < 0.15, f"弧長不均勻超過轉角可解釋範圍：cv={cv:.3f}"
R1 = anchors_resample(A[:1], 16)
assert R1.shape == (16, 2) and np.allclose(R1, A[0]), "K=1 tile 錯"
ok += 1; print(f"④ resample OK：全點在折線上（max 偏 {offline:.1e}）、步距 cv {cv:.3%}、K=1 tile 正確")

# ⑤ 一站式：hindsight_intent 與 route_intent 各跑一次、輸出形狀＋有限值
H, n_c, j = hindsight_intent(traj, xy_to_cell, cell_to_xy, 16)
assert H.shape == (16, 2) and np.isfinite(H).all() and n_c == len(cells)
Rt = route_intent(occ, np.array([1.0, 1.0]), np.array([5.0, 5.0]), xy_to_cell, cell_to_xy, 16)
assert Rt is not None and Rt.shape == (16, 2) and np.isfinite(Rt).all()
assert np.allclose(Rt[0], [1, 1], atol=1e-9) and np.allclose(Rt[-1], [5, 5], atol=1e-9)
ok += 1; print("⑤ 一站式 OK：hindsight/route 兩條路形狀、端點、有限值全過")

# ⑥ 弧長測試空轉的補丁（驗收攻擊實證）：非等距錨（段長 1/4/3，故意不等距）resample
#    後，相鄰重採樣點沿「折線弧長」的間隔必須均勻——不是沿索引均攤。用獨立於
#    anchors_resample 內部算法的「投影找弧長座標」（沿用④ _dist_to_polyline 的
#    投影公式）重新量測，這樣如果實作退化成索引均攤，這裡一定會 FAIL。
A6 = np.array([[0.0, 0.0], [1.0, 0.0], [5.0, 0.0], [5.0, 3.0]])  # 段長 1,4,3：不等距
T6 = 8
R6 = anchors_resample(A6, T6)


def _arclen_coord(q, P):
    """q 在折線 P 上的弧長座標（從 P[0] 起算）：對每段投影取距離最小的那段定弧長，
    獨立於 anchors_resample 的內部算法。"""
    seglens = [float(np.linalg.norm(b - a)) for a, b in zip(P[:-1], P[1:])]
    cumlens = np.concatenate([[0.0], np.cumsum(seglens)])
    best_dist, best_arc = np.inf, 0.0
    for i, (a, b) in enumerate(zip(P[:-1], P[1:])):
        d = b - a; L2 = float(d @ d)
        t = 0.0 if L2 == 0 else float(np.clip((q - a) @ d / L2, 0, 1))
        dist = float(np.linalg.norm(q - (a + t * d)))
        if dist < best_dist:
            best_dist, best_arc = dist, cumlens[i] + t * seglens[i]
    return best_arc


arcs6 = np.array([_arclen_coord(q, A6) for q in R6])
d_arcs6 = np.diff(arcs6)
total6 = float(np.linalg.norm(np.diff(A6, axis=0), axis=1).sum())
expect_step = total6 / (T6 - 1)
assert np.allclose(d_arcs6, expect_step, atol=1e-9), (
    f"弧長間隔不均勻（非等距錨下應仍均勻）：steps={d_arcs6}, expect={expect_step}"
)
ok += 1; print(f"⑥ 非等距錨弧長均勻 OK：step={expect_step:.6f}，max|diff|={np.abs(d_arcs6-expect_step).max():.2e}")

print(f"ALL PASS ({ok}/6)")
