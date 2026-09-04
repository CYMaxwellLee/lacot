"""intent_embed 自測：CPU、無資料、秒級 —— 驗 (i) embed 接法的介面契約。"""
import torch

from lacot.intent_embed import IntentAdapter

# ① 建構、維度屬性
m = IntentAdapter(32, 8, 8, 128)
assert m.cond_extra_dim == 64, f"cond_extra_dim 應為 64，實得 {m.cond_extra_dim}"
assert m.pertoken_dim == 0, f"pertoken_dim 應為 0，實得 {m.pertoken_dim}"
print("① 建構 OK：cond_extra_dim=64、pertoken_dim=0")

# ② cond_global 形狀 + grad 傳播到 MLP 權重
anchors = torch.randn(4, 32, 2)
out = m.cond_global(anchors)
assert out.shape == (4, 64), f"cond_global 形狀錯：{tuple(out.shape)}"
out.sum().backward()
grads = [p.grad for p in m.mlp.parameters()]
assert all(g is not None for g in grads), "MLP 權重沒收到 grad"
assert any(float(g.abs().sum()) > 0 for g in grads), "grad 全零，backward 沒真的傳到"
print(f"② cond_global OK：shape={tuple(out.shape)}、MLP {len(grads)} 組權重皆有 grad")

# ③ cond_pertoken 恆 None；target_fwd/target_inv 恆等（T 故意跟 T_A 不同，驗沒偷用 32）
assert m.cond_pertoken(anchors) is None, "cond_pertoken 應恆回 None"
traj = torch.randn(4, 16, 2)
fwd = m.target_fwd(traj, anchors)
assert torch.allclose(fwd, traj), "target_fwd 不是恆等"
inv = m.target_inv(fwd, anchors)
assert torch.allclose(inv, traj), "target_inv 不是恆等"
print("③ cond_pertoken=None、target_fwd/target_inv 恆等 OK")

print("ALL PASS")
