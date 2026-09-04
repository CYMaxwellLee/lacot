"""intent 接法 (iii) residual smoke：CPU、無資料、秒級 —— round-trip 是這個
接法的靈魂（target_fwd/target_inv 必須精確互逆），另外驗殘差意義、退化、
cond_global 可訓、T 不寫死。"""
import torch

from lacot.intent_residual import NAME, TAG, IntentAdapter, _upsample_arclength

torch.manual_seed(0)
adapter = IntentAdapter(32, 8, 8, 128)
ok = 0

# ① 建構：介面規格的兩個固定值 + NAME/TAG
assert adapter.cond_extra_dim == 64, f"cond_extra_dim 錯：{adapter.cond_extra_dim}"
assert adapter.pertoken_dim == 0, f"pertoken_dim 錯：{adapter.pertoken_dim}"
assert NAME == "residual" and TAG == "itr", f"NAME/TAG 錯：{NAME}/{TAG}"
ok += 1; print(f"① 建構 OK：cond_extra_dim={adapter.cond_extra_dim} pertoken_dim={adapter.pertoken_dim} NAME={NAME} TAG={TAG}")

# ② round-trip（靈魂）：target_inv(target_fwd(traj,anchors),anchors) == traj
traj = torch.randn(4, 128, 2)
anchors = torch.randn(4, 32, 2)
resid = adapter.target_fwd(traj, anchors)
back = adapter.target_inv(resid, anchors)
assert resid.shape == traj.shape
err = (back - traj).abs().max().item()
assert torch.allclose(back, traj, atol=1e-5), f"round-trip 沒對齊，max err={err:.2e}"
ok += 1; print(f"② round-trip OK：max|back-traj|={err:.2e}（atol=1e-5）")

# ③ 殘差意義：輪廓完美（anchors、traj 是同一條直線）⇒ 殘差範數 ≈ 0
#   用 float64 排除 float32 累積誤差干擾，讓「輪廓完美時殘差為零」這個數學
#   性質本身被乾淨驗到，不跟浮點雜訊混在一起。
p0 = torch.tensor([0.7, -1.3], dtype=torch.float64)
p1 = torch.tensor([4.2, 3.6], dtype=torch.float64)
t_a = torch.linspace(0.0, 1.0, 32, dtype=torch.float64).unsqueeze(-1)
t_t = torch.linspace(0.0, 1.0, 128, dtype=torch.float64).unsqueeze(-1)
anchors_line = (p0 + t_a * (p1 - p0)).unsqueeze(0)          # [1,32,2] 直線路線
traj_line = (p0 + t_t * (p1 - p0)).unsqueeze(0)              # [1,128,2] 同一條直線上採樣
resid_line = adapter.target_fwd(traj_line, anchors_line)
resid_norm = resid_line.norm().item()
assert resid_norm < 1e-6, f"完美輪廓的殘差範數應 ≈0，實得 {resid_norm:.2e}"
ok += 1; print(f"③ 殘差意義 OK：直線輪廓完美時 ‖residual‖={resid_norm:.2e}（<1e-6）")

# ④ 退化：anchors 全部同一點（總弧長 0）⇒ target_fwd 不 NaN、upsample 等於 tile
anchors_deg = torch.zeros(2, 32, 2)
anchors_deg[0] += torch.tensor([3.0, -2.0])
anchors_deg[1] += torch.tensor([-1.0, 5.0])
T_deg = 20
up_deg = _upsample_arclength(anchors_deg, T_deg)
assert torch.isfinite(up_deg).all(), "退化 upsample 出現非有限值"
tile_expect = anchors_deg[:, :1, :].expand(-1, T_deg, -1)
assert torch.allclose(up_deg, tile_expect), "退化 upsample 不等於 tile 第一錨"
traj_dummy = torch.randn(2, T_deg, 2)
resid_deg = adapter.target_fwd(traj_dummy, anchors_deg)
assert torch.isfinite(resid_deg).all(), "退化 target_fwd 出現非有限值（NaN/Inf）"
#   T_A=1（只有一個錨，起訖同格）：不能崩（舊實作 clamp(1, T_A-1)=clamp(1,0) 會
#   min>max 反轉、idx0=-1），且 upsample 精確等於 tile 該錨；round-trip 照樣要對齊。
anchors_t1 = torch.randn(3, 1, 2)
T_t1 = 12
up_t1 = _upsample_arclength(anchors_t1, T_t1)
assert up_t1.shape == (3, T_t1, 2), f"T_A=1 upsample 形狀錯：{tuple(up_t1.shape)}"
assert torch.allclose(up_t1, anchors_t1.expand(-1, T_t1, -1)), "T_A=1 upsample 應等於 tile 該錨"
traj_t1 = torch.randn(3, T_t1, 2)
resid_t1 = adapter.target_fwd(traj_t1, anchors_t1)
back_t1 = adapter.target_inv(resid_t1, anchors_t1)
assert torch.allclose(back_t1, traj_t1, atol=1e-5), "T_A=1 round-trip 沒對齊"
ok += 1; print(f"④ 退化 OK：全同一點 ⇒ 不 NaN、upsample 精確等於 tile(第一錨)；T_A=1 不崩、tile 精確、round-trip 對齊")

# ⑤ cond_global 形狀 + 可訓（backward 有 grad）；cond_pertoken 恆回 None
anchors5 = torch.randn(4, 32, 2)
cond = adapter.cond_global(anchors5)
assert cond.shape == (4, 64), f"cond_global 形狀錯：{tuple(cond.shape)}"
adapter.zero_grad()
cond.sum().backward()
grads = [p.grad for p in adapter.parameters()]
assert all(g is not None for g in grads), "有參數沒收到梯度"
assert any(torch.any(g != 0) for g in grads), "所有梯度都是 0，backward 沒真的動"
assert adapter.cond_pertoken(anchors5) is None, "cond_pertoken 沒回 None"
ok += 1; print(f"⑤ cond_global OK：shape={tuple(cond.shape)}，{len(grads)} 組參數皆收到非零梯度；cond_pertoken=None")

# ⑥ T 不寫死：traj T=96（非 128）一樣要 round-trip
traj6 = torch.randn(2, 96, 2)
anchors6 = torch.randn(2, 32, 2)
resid6 = adapter.target_fwd(traj6, anchors6)
assert resid6.shape == (2, 96, 2), f"T=96 殘差形狀錯：{tuple(resid6.shape)}"
back6 = adapter.target_inv(resid6, anchors6)
err6 = (back6 - traj6).abs().max().item()
assert torch.allclose(back6, traj6, atol=1e-5), f"T=96 round-trip 沒對齊，max err={err6:.2e}"
ok += 1; print(f"⑥ T 不寫死 OK：T=96 round-trip max|back-traj|={err6:.2e}")

# ⑦ 弧長測試空轉的補丁：非等距錨（段長 1/4/3）upsample 後，中段某點要落在「弧長
#    比例」對應的位置，不是「索引比例」位置——兩者在非等距下差異明顯。用同一批
#    t_dst 手算一個獨立的「索引均攤」參照位置（t_src 用 index-uniform 而非弧長
#    累積），跟真實輸出比對距離；若 _upsample_arclength 退化成索引均攤，距離會
#    趨近 0，這裡就會 FAIL。
A7 = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [5.0, 0.0], [5.0, 3.0]]])  # [1,4,2] 段長 1,4,3
T7 = 8
out7 = _upsample_arclength(A7, T7)
mid = T7 // 2  # 中段索引（非端點，才有鑑別力）
t_dst_mid = mid / (T7 - 1)
t_src_idx = torch.linspace(0.0, 1.0, A7.shape[1])  # 錯誤（索引均攤）版本的參數化
j1 = int(torch.searchsorted(t_src_idx, torch.tensor(t_dst_mid), right=True).clamp(1, A7.shape[1] - 1))
j0 = j1 - 1
w_idx = (t_dst_mid - t_src_idx[j0]) / (t_src_idx[j1] - t_src_idx[j0])
idx_based_pos = A7[0, j0] + w_idx * (A7[0, j1] - A7[0, j0])
dist = float(torch.linalg.norm(out7[0, mid] - idx_based_pos))
assert dist > 0.5, f"弧長位置跟索引位置差異太小（懷疑退化成索引均攤）：dist={dist:.3f}"
ok += 1; print(f"⑦ 非等距錨弧長比例 OK：中段點={out7[0, mid].tolist()}，索引均攤參照={idx_based_pos.tolist()}，距離差={dist:.3f}（>0.5）")

print(f"ALL PASS ({ok}/7)")
