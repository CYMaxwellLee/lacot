"""intent 接法 (ii) anchor smoke：屬性／形狀＋梯度／局部性／恆等／防呆五項驗證（CPU、秒級、無資料）。"""
import torch

from lacot.intent_anchor import IntentAdapter

ok = 0

# ① 建構＋介面屬性
m = IntentAdapter(32, 8, 8, 128)
assert m.cond_extra_dim == 0, f"cond_extra_dim 應為 0，拿到 {m.cond_extra_dim}"
assert m.pertoken_dim == 16, f"pertoken_dim 應為 16，拿到 {m.pertoken_dim}"
ok += 1; print("① 建構＋屬性 OK：cond_extra_dim=0, pertoken_dim=16")

# ② 形狀＋梯度：cond_pertoken 輸出 [4,8,16]，backward 後共用 Linear 收到 grad
anchors = torch.randn(4, 32, 2)
out = m.cond_pertoken(anchors)
assert out.shape == (4, 8, 16), f"cond_pertoken 形狀錯：{tuple(out.shape)}"
out.sum().backward()
lin = m.proj[0]
assert lin.weight.grad is not None and torch.isfinite(lin.weight.grad).all(), "Linear 沒收到 grad"
ok += 1; print(f"② 形狀＋梯度 OK：cond_pertoken -> {tuple(out.shape)}，Linear.weight.grad 有限")

# ③ 局部性（這個接法的靈魂）：只重新取樣第一段（→ token 0），其它 token 的輸出不該動
base = torch.randn(4, 32, 2)
with torch.no_grad():
    out_base = m.cond_pertoken(base)
    changed = base.clone()
    changed[:, 0:4] = torch.randn(4, 4, 2)   # 只動第一段（對應 token 0）
    out_changed = m.cond_pertoken(changed)
moved = (out_changed - out_base).abs().amax(dim=-1)   # [4,8]：每個 token 的最大變化量
assert (moved[:, 0] > 1e-4).all(), f"token 0 該變但沒變，moved={moved[:, 0]}"
assert torch.allclose(out_changed[:, 1:], out_base[:, 1:], atol=1e-6), "token 1..7 不該變卻變了——硬錨漏水"
ok += 1; print(f"③ 局部性 OK：token 0 最大變化 {moved[:, 0].max():.3f}，token 1..7 allclose 不變")

# ④ cond_global 恆回 None；target_fwd／target_inv 恆等
assert m.cond_global(base) is None, "cond_global 應恆回 None"
traj = torch.randn(4, 8, 8)
out2 = torch.randn(4, 8, 8)
assert torch.allclose(m.target_fwd(traj, base), traj), "target_fwd 不是恆等"
assert torch.allclose(m.target_inv(out2, base), out2), "target_inv 不是恆等"
ok += 1; print("④ cond_global=None、target_fwd/target_inv 恆等 OK")

# ⑤ 防呆：T_A 不是 K 的倍數應 raise，且錯誤訊息要清楚
raised, msg = False, ""
try:
    IntentAdapter(30, 8, 8, 128)
except AssertionError as e:
    raised, msg = True, str(e)
assert raised, "t_anchor=30, k_tokens=8（不整除）應該要 raise 卻沒有"
assert "t_anchor=30" in msg and "k_tokens=8" in msg, f"錯誤訊息不夠清楚：{msg}"
ok += 1; print(f"⑤ 防呆 OK：{msg}")

print(f"ALL PASS ({ok}/5)")
