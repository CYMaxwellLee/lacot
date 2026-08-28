"""smoke：GeoValue ＋ 主人的更新式。⭐ 每格都寫明【為什麼期望是這個答案】。"""
import sys
sys.path.insert(0, "/home/cymaxwelllee/Projects/lacot")
import numpy as np, torch
from lacot.refine_grad import GeoValue, grad_refine, _clip
from lacot.traj_decoder import TrajDecoder

fails = []
rng = np.random.default_rng(0); torch.manual_seed(0)

# ── 造一個 L 形走廊：只有走廊裡有資料，轉角外側是「牆」──────────────
pts = []
for _ in range(400):
    t = rng.random(60)
    x = np.where(t < 0.5, t * 8, 4.0) + rng.normal(0, .05, 60)
    y = np.where(t < 0.5, 0.0, (t - 0.5) * 8) + rng.normal(0, .05, 60)
    pts.append(np.stack([x, y], 1))
OBS = np.concatenate(pts).astype(np.float32)
mu, sd = OBS.mean(0), OBS.std(0) + 1e-6
geo = GeoValue(OBS, mu, sd, res=8)
print(f"佔據圖 {tuple(geo.shape)}，資料覆蓋 {geo.coverage:.1%} 的格")

Z = torch.tensor((OBS - mu) / sd, dtype=torch.float32)

# ── case 1：真軌跡的穿牆懲罰必須 ≈ 0 ────────────────────────────
#    ⭐ 這是 SDF 有沒有蓋歪的照妖鏡 —— 資料裡的路本來就不穿牆
real = Z[:64 * 32].reshape(64, 32, 2)
wd = geo.wall_depth(real)
print(f"case1  真軌跡穿牆深度  中位 {wd.median():.4f}  p90 {wd.flatten().quantile(.9):.4f}")
if wd.median() > 0.15:
    fails.append(f"case1 真軌跡穿牆中位 {wd.median():.3f} 太大 ⇒ SDF 蓋歪了")

# ── case 2：走廊外面的點必須被判成穿牆 ─────────────────────────
#    ⭐ 一把不會叫的尺跟壞掉的尺長一樣 —— L 形的凹角外側是明確的牆
outside = torch.tensor([[[6.0, 6.0]] * 32], dtype=torch.float32)
outside = (outside - torch.tensor(mu)) / torch.tensor(sd)
wo = geo.wall_depth(outside)
print(f"case2  走廊外的點穿牆深度 {wo.mean():.4f}")
if wo.mean() < 0.5:
    fails.append(f"case2 走廊外只判 {wo.mean():.3f} ⇒ 這把尺分不出牆")

# ── case 3：V 對座標可微，而且四項都有貢獻 ────────────────────
p = real.clone().requires_grad_(True)
s_, g_ = real[:, 0], real[:, -1]
v, terms = geo(p, s_, g_, per_term=True)
v.sum().backward()
print(f"case3  V 中位 {v.median():.3f}   四項 " +
      " ".join(f"{k}={float(t.mean()):.3f}" for k, t in terms.items()) +
      f"   ‖∇V‖={p.grad.norm():.3f}")
if p.grad.norm() < 1e-6:
    fails.append("case3 V 對座標的梯度是 0 ⇒ 爬坡無處可去")
if float(terms["start"].mean()) > 1e-5 or float(terms["goal"].mean()) > 1e-5:
    fails.append("case3 s/g 直接取自軌跡端點 ⇒ 起終點兩項應該 ≈0")

# ── case 4：爬坡要真的把壞的 u 修好 ────────────────────────────
#    ⭐ 這是「機器活著沒」的直接證據，⛔ 不是替身指標
dec = TrajDecoder(64, 32)
class MockFlow:                       # log p = -||u||²/2 ⇒ 結界把 u 拉回原點
    def log_prob(self, u, cond): return -0.5 * u.pow(2).flatten(1).sum(1)
u_bad = torch.randn(16, 4, 64) * 3.0  # 故意壞的 u（高溫抽樣的替身）
sg = real[:16, 0], real[:16, -1]
v0 = geo(dec(u_bad), *sg)
u_fix, hist = grad_refine(u_bad, None, dec, MockFlow(), geo, *sg,
                          steps=60, eta=0.1, lam=0.3, trace=True)
v1 = geo(dec(u_fix), *sg)
print(f"case4  爬坡前 V {v0.mean():.3f} → 爬坡後 {v1.mean():.3f}   ({v1.mean()-v0.mean():+.3f})")
print(f"       穿牆 {hist[0]['wall']:.3f}→{hist[-1]['wall']:.3f}"
      f"   離終點 {hist[0]['goal']:.3f}→{hist[-1]['goal']:.3f}"
      f"   log p {hist[0]['logp']:.1f}→{hist[-1]['logp']:.1f}")
if v1.mean() <= v0.mean():
    fails.append(f"case4 爬坡沒有讓 V 上升（{v0.mean():.3f}→{v1.mean():.3f}）⇒ 更新式方向錯了")

# ── case 5：η=0 的 null control 必須【不動】────────────────────
#    ⭐ 沒有這格的話，case4 的上升可能只是「動了就會變好」
u_null = grad_refine(u_bad, None, dec, MockFlow(), geo, *sg, steps=60, eta=0.0)
if not torch.allclose(u_null, u_bad):
    fails.append("case5 η=0 卻動了 ⇒ 更新式裡有不受 η 控制的東西")
print(f"case5  η=0 的對照組原地不動 {'✓' if torch.allclose(u_null, u_bad) else '🚨'}")

# ── case 6：兩項梯度各自限長 ───────────────────────────────────
big, small = torch.randn(4, 4, 64) * 100, torch.randn(4, 4, 64) * 0.001
raw_s = float(small.norm(dim=(1, 2)).mean())      # 小梯度的原始長度（≈0.016）
for mode in ("normalize", "clamp"):
    nb = float(_clip(big, mode).norm(dim=(1, 2)).mean())
    ns = float(_clip(small, mode).norm(dim=(1, 2)).mean())
    if mode == "normalize":
        ok = abs(nb - 1) < .01 and abs(ns - 1) < .01          # 兩項都拉成單位長 ⇒ 等權
        why = "兩項等權"
    else:
        ok = abs(nb - 1) < .01 and abs(ns - raw_s) < 1e-4     # 只縮不放 ⇒ 小的保持原樣
        why = "只限步長、相對量級不變"
    print(f"case6  {mode:<10} 大 {nb:.3f}  小 {ns:.5f}（原始 {raw_s:.5f}）  {why}  {'✓' if ok else '🚨'}")
    if not ok:
        fails.append(f"case6 {mode} 的行為不符：大 {nb:.3f} 小 {ns:.5f} 原始 {raw_s:.5f}")

print()
if fails:
    print("🚨 FAIL"); [print("  -", f) for f in fails]; sys.exit(1)
print("✅ 6/6 PASS")
