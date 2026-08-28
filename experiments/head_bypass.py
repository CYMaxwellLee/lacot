"""第 0 步（主人 2026-08-24 批准）：量 action head 到底把多少權重放在 u 上、多少放在 cond 上。

背景：三篇 test-time-scaling 論文（Huginn 2502.05171 / STARS 2605.26733 / RD-VLA 2602.07845）
的共同結構是「latent 是輸出的唯一通路」，而我們的 ahead 同時吃 cond 和 u ⇒ u 可以是垃圾。
這支直接量那條旁路有多寬。

⭐ 為什麼看第一層權重：ActionMLP 的輸入是 cat([cond, u.flatten()])，
   ⇒ net.0.weight 的 column 可以【乾淨地】切成 cond 那塊與 u 那塊。過了第一層就混在一起，切不開了。
⚠️ 侷限：這只量「第一層給了多少通道」，⛔ 不等於端到端敏感度（後面的層可能放大某一路）。
   端到端那格要用 Jacobian，需要真 (s,g)，另外跑。

🚨 控制組（沒有它這支就是廢的）：一顆【隨機初始化】的同結構 head。
   nn.Linear 的初始化對所有 column 是同一個分布 ⇒ 未訓練時 cond 與 u 的每維 RMS 必須幾乎相等。
   ⇒ 如果訓練後仍然相等，本支的診斷（旁路假說）就是【錯的】。
"""
import os
import sys

import torch
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def sota_mlp(i, h, o, n=2):
    L, p = [], i
    for _ in range(n):
        L += [nn.Linear(p, h), nn.GELU(), nn.LayerNorm(h)]; p = h
    return nn.Sequential(*L, nn.Linear(p, o))


def split_stats(W, n_cond):
    """W: (out, n_cond + n_u)。回傳兩塊的 Frobenius norm 與【每維 RMS】。"""
    Wc, Wu = W[:, :n_cond], W[:, n_cond:]
    f = lambda X: (X.pow(2).sum().sqrt().item(), (X.pow(2).mean().sqrt()).item())
    return f(Wc), f(Wu)


rows = []
for fn in sorted(os.listdir("results")):
    if not (fn.startswith("ckpt_") and fn.endswith(".pt")):
        continue
    ck = torch.load(os.path.join("results", fn), map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    COND, DIM = cfg["COND"], cfg["K"] * cfg["D_MODEL"]
    W = ck["ahead"]["net.0.weight"]
    assert W.shape[1] == COND + DIM, f"{fn}: {W.shape} vs {COND}+{DIM}"
    (fc, rc), (fu, ru) = split_stats(W, COND)
    # 純 BC head（只吃 cond）當第二個參考點：ahead 對 cond 的用力程度跟它比。
    rbc = ck["bc_head"]["net.0.weight"].pow(2).mean().sqrt().item()
    rows.append((fn[5:-3], COND, DIM, fc, rc, fu, ru, rc / max(ru, 1e-12), rbc))

# 🚨 控制組：未訓練的同結構 head。
torch.manual_seed(0)
COND0, DIM0 = 256, 1024
ctrl = sota_mlp(COND0 + DIM0, 512, 4 * 2, n=3)
(fc0, rc0), (fu0, ru0) = split_stats(ctrl[0].weight.detach(), COND0)

print()
print("每維 RMS = ‖該塊‖_F / sqrt(該塊的元素數)  ⇒ 已消掉 cond(256) 與 u(1024) 的維度差")
print(f"{'checkpoint':<44} {'cond每維':>9} {'u每維':>9} {'cond/u':>8} {'bcHead每維':>11}")
print("-" * 88)
print(f"{'[控制組] 隨機初始化、未訓練':<44} {rc0:>9.5f} {ru0:>9.5f} {rc0/ru0:>8.3f} {'—':>11}")
print("-" * 88)
for name, C, D, fc, rc, fu, ru, ratio, rbc in rows:
    print(f"{name:<44} {rc:>9.5f} {ru:>9.5f} {ratio:>8.3f} {rbc:>11.5f}")
print("-" * 88)
if rows:
    import statistics
    print(f"訓練後 cond/u 比值：中位數 {statistics.median(r[7] for r in rows):.3f}"
          f"　（控制組 {rc0/ru0:.3f}）")
