"""nf_head per-token cond 擴展 smoke：golden 回歸（2D 行為 bit 級不變）＋3D 正確性。"""
import os
import numpy as np
import torch

from lacot.nf_head import Flow

GOLD = os.environ.get("NF_GOLDEN_DIR",
    "/tmp/claude-2007/-home-cymaxwelllee-Projects-elsa-agent-workspaces-luna/"
    "e0a02f0a-3a36-47eb-b681-eb5f124604b9/scratchpad")
ok = 0

# ① golden 回歸：改動前存的權重＋輸入 ⇒ forward z/logdet 與 sample 全 allclose
torch.manual_seed(7)
f = Flow(token_dim=8, seq_len=8, n_blocks=4, cond_dim=128)   # 同 seed 下建構＝同 golden 權重
g0 = np.load(f"{GOLD}/nf_golden.npz")
f.load_state_dict(torch.load(f"{GOLD}/nf_golden_sd.pt", weights_only=True))
u = torch.randn(4, 8, 8); cond2 = torch.randn(4, 128)        # 建構後的 RNG 流與 golden 腳本一致
z, ld = f(u, cond2)
s = f.sample(4, cond2, generator=torch.Generator().manual_seed(11))
assert np.allclose(z.detach().numpy(), g0["z"], atol=1e-6), "2D forward z 變了（回歸破壞）"
assert np.allclose(ld.detach().numpy(), g0["ld"], atol=1e-6), "2D logdet 變了"
assert np.allclose(s.numpy(), g0["s"], atol=1e-6), "2D sample 變了"
ok += 1; print("① golden 回歸 OK：2D cond 的 forward/sample 與改動前 bit 級一致")

# ⚠️ 以下測試【必須】打破 zero-init：to_params 零權重時 block 是恆等映射，cond 進不到
#    輸出 ⇒ round-trip／logdet／影響力全部空測（③ 會 0=0「假過」）。ルナ 9/4 親踩。
def _wake(fl):
    with torch.no_grad():
        for b in fl.blocks:
            b.to_params.weight.normal_(0, 0.05); b.to_params.bias.normal_(0, 0.02)
    return fl
_wake(f)

# ② 3D round-trip：forward 後照 sample 的逆序手寫還原 ⇒ allclose 原 u
cond3 = torch.randn(4, 8, 128)
z3, ld3 = f(u, cond3)
r = z3.clone()
for i in reversed(range(len(f.blocks))):
    if i < len(f.blocks) - 1:
        r = f.perm.inverse(r)
    r = f.blocks[i].inverse(r, f._cond_for_block(cond3, i))
assert torch.allclose(r, u, atol=1e-4), f"3D round-trip 爆了 max err {(r-u).abs().max():.2e}"
ok += 1; print(f"② 3D round-trip OK：max err {(r-u).abs().max():.2e}")

# ③ logdet 精確性（小維、autograd jacobian）：K=3、D=2、B=1、單 block、⚠️ 非零權重
torch.manual_seed(3)
fs = _wake(Flow(token_dim=2, seq_len=3, n_blocks=1, cond_dim=16)).double()
u0 = torch.randn(1, 3, 2, dtype=torch.float64)
c3 = torch.randn(1, 3, 16, dtype=torch.float64)
J = torch.autograd.functional.jacobian(
    lambda x: fs(x, c3)[0].reshape(-1), u0, vectorize=True).reshape(6, 6)
ld_true = torch.slogdet(J)[1]
ld_got = fs(u0, c3)[1][0]
assert abs(float(ld_true)) > 1e-4, "logdet 仍為 0 ⇒ 恆等沒被打破，測試無效"
assert torch.allclose(ld_true, ld_got, atol=1e-8), f"logdet 錯：真 {ld_true:.6f} vs 報 {ld_got:.6f}"
ok += 1; print(f"③ 3D logdet 精確 OK（非退化）：autograd {ld_true:.6f} = 報告 {ld_got:.6f}")

# ④ causal（⚠️ 只在【單 block】驗 —— 多 block 疊了 Permutation 後最終 z 本來就依賴全部 u，
#    那是設計不是 bug；ルナ第一版在 4-block 上驗 causal＝測試寫錯層級）＋cond 影響力
u1 = torch.randn(1, 3, 2, dtype=torch.float64)
z1, _ = fs(u1, c3)
u1p = u1.clone(); u1p[:, 2] += 1.0
z1p, _ = fs(u1p, c3)
assert torch.allclose(z1p[:, :2], z1[:, :2], atol=1e-10), "單 block causal 被 prefix 擴展弄壞"
assert not torch.allclose(z1p[:, 2], z1[:, 2]), "token 2 沒反應（切片錯位）"
c_pert = cond3.clone(); c_pert[:, 5] += 1.0
z_c, _ = f(u, c_pert)
assert not torch.allclose(z_c, z3, atol=1e-6), "prefix 段對輸出沒有影響（接了等於沒接）"
ok += 1; print("④ causal（單 block）／cond 影響力 OK")

# ⑤ 奇數 block 的 cond flip 生效：_cond_for_block 在 i=1 應是 flip、i=0/2 原樣
cf0 = Flow._cond_for_block(cond3, 0); cf1 = Flow._cond_for_block(cond3, 1)
assert cf0 is cond3 and torch.allclose(cf1, torch.flip(cond3, dims=(1,))), "flip 節奏錯"
assert Flow._cond_for_block(cond2, 1) is cond2, "2D cond 不准被 flip"
ok += 1; print("⑤ Permutation 同步 OK：奇數 block flip 3D cond、2D 原樣")

print(f"ALL PASS ({ok}/5)")
