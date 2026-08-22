"""head 到底有多依賴 `u`？—— 直接量，不用推的（2026-08-23 深夜）

背景：三個 seed 量到 `null-u floor`（head(cond,0)）≈ LaCoT R=3，
      由此【推論】head 學會忽略 u。但那是從結果反推的。
      這支直接量 head 對兩個輸入的敏感度，把推論變成量測。

量三件事（都在訓練好的 head 上）：
  ① 擾動敏感度：對 u 加噪聲 vs 對 cond 加噪聲，動作各變多少
  ② 梯度範數：||∂a/∂u|| vs ||∂a/∂cond||（各自除以輸入維度，才可比）
  ③ 歸零測試：u 歸零 vs cond 歸零，動作各偏離多少

⛔ 對照組：同時訓一個【沒有 cond】的 head（只吃 u）。
   若「有 cond」那組對 u 的敏感度顯著低於「沒 cond」那組，就證實了
   ——【是 cond 的存在讓 head 放棄了 u】，而不是 u 本身沒有資訊。
"""
import os, sys, numpy as np, torch
from torch import nn
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lacot.e_target import PerceiverPooler
from lacot.heads import ContinuousActionHead

OGB_DATA = os.environ.get("OGBENCH_DATA_DIR", "/archive/cymaxwelllee/data/ogbench")
device = "cuda" if torch.cuda.is_available() else "cpu"
ENV = "pointmaze-medium-navigate-v0"
K, D_MODEL, T_CAP, COND, CHUNK, ADIM, B = 4, 256, 16, 256, 4, 2, 64
GEOM_P, TEMP, WANDER_MAX = 0.02, 0.1, 3.0
DIM = K * D_MODEL
STEPS1, STEPS2 = 1500, 3000

d = np.load(f"{OGB_DATA}/{ENV}.npz")
OBS = np.asarray(d["observations"], np.float32)
ACT = np.asarray(d["actions"], np.float32)
TERM = np.asarray(d["terminals"], bool)
N = OBS.shape[0]
ends = np.flatnonzero(TERM); starts = np.concatenate([[0], ends[:-1] + 1])
traj_end = np.empty(N, np.int64)
for s0, e0 in zip(starts, ends):
    traj_end[s0:e0 + 1] = e0
mu, sd = OBS.mean(0), OBS.std(0) + 1e-6
print(f"device {device} | K={K} dim={DIM}", flush=True)


def make_batch(rng, b=B):
    rows, goals = [], []
    while len(rows) < b:
        r = int(rng.integers(0, N)); te = int(traj_end[r])
        if te - r < CHUNK:
            continue
        gr = min(r + int(rng.geometric(GEOM_P)), te)
        if gr - r < CHUNK:
            continue
        path = OBS[r:gr + 1]
        if np.linalg.norm(np.diff(path, axis=0), axis=1).sum() / (np.linalg.norm(path[-1] - path[0]) + 1e-6) > WANDER_MAX:
            continue
        rows.append(r); goals.append(gr)
    rows, goals = np.array(rows), np.array(goals)
    idxs = [np.unique(np.linspace(rows[i], goals[i], min(T_CAP, goals[i] - rows[i] + 1)).round().astype(int)) for i in range(b)]
    Tmax = max(len(ix) for ix in idxs)
    traj = np.zeros((b, Tmax, 2), np.float32); mask = np.ones((b, Tmax), bool)
    for i, ix in enumerate(idxs):
        traj[i, :len(ix)] = (OBS[ix] - mu) / sd; mask[i, :len(ix)] = False
    s = (OBS[rows] - mu) / sd; g = (OBS[goals] - mu) / sd
    act = np.stack([ACT[r:r + CHUNK] for r in rows]).astype(np.float32)
    T_ = lambda x: torch.from_numpy(x.astype(np.float32)).to(device)
    return T_(traj), torch.from_numpy(mask).to(device), T_(s), T_(g), T_(act)


def mlp(i, h, o, n=2):
    L, p = [], i
    for _ in range(n):
        lin = nn.Linear(p, h); nn.init.xavier_uniform_(lin.weight); nn.init.zeros_(lin.bias)
        L += [lin, nn.GELU(), nn.LayerNorm(h)]; p = h
    lin = nn.Linear(p, o); nn.init.xavier_uniform_(lin.weight); nn.init.zeros_(lin.bias)
    return nn.Sequential(*L, lin)


torch.manual_seed(0); rng = np.random.default_rng(0)
traj_enc = mlp(2, 512, 512).to(device)
e_pooler = PerceiverPooler(512, D_MODEL, K, 2, 4).to(device)
sg_c = mlp(2, 512, 512).to(device)
q_pooler = PerceiverPooler(512, D_MODEL, K, 2, 4).to(device)
opt1 = torch.optim.Adam([p for m in (traj_enc, e_pooler, sg_c, q_pooler) for p in m.parameters()], lr=1e-3)
lab = torch.arange(B, device=device)


def etarget(traj, mask):
    b, t, _ = traj.shape
    return e_pooler(traj_enc(traj.reshape(b * t, 2)).reshape(b, t, 512), key_padding_mask=mask)


print("stage 1: contrastive e_target ...", flush=True)
for stp in range(STEPS1):
    traj, mask, s, g, _ = make_batch(rng)
    et = etarget(traj, mask); q = q_pooler(torch.stack([sg_c(s), sg_c(g)], 1))
    logits = (F.normalize(q.reshape(B, -1), dim=1) @ F.normalize(et.reshape(B, -1), dim=1).t()) / TEMP
    loss = 0.5 * (F.cross_entropy(logits, lab) + F.cross_entropy(logits.t(), lab))
    opt1.zero_grad(set_to_none=True); loss.backward(); opt1.step()
for m in (traj_enc, e_pooler):
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
print(f"  match-acc {(logits.argmax(1)==lab).float().mean().item():.3f}", flush=True)

# 兩組 head：吃 [cond,u] vs 只吃 u
cond_enc = mlp(2, 512, 512).to(device); cond_head = mlp(1024, 512, COND).to(device)
head_cond = ContinuousActionHead(COND + DIM, ADIM, CHUNK).to(device)   # 有 cond
head_only = ContinuousActionHead(DIM, ADIM, CHUNK).to(device)          # ⛔ 沒 cond（對照）
opt2 = torch.optim.Adam(
    [p for m in (cond_enc, cond_head, head_cond, head_only) for p in m.parameters()], lr=5e-4)


def condvec(s, g):
    return cond_head(torch.cat([cond_enc(s), cond_enc(g)], 1))


print(f"stage 2: 兩個 head 並排訓 {STEPS2} 步（同一個 u、同一批資料）...", flush=True)
for stp in range(STEPS2):
    traj, mask, s, g, act = make_batch(rng)
    with torch.no_grad():
        et = etarget(traj, mask)
    c = condvec(s, g)
    u_flat = et.reshape(B, -1)
    l1 = (head_cond(torch.cat([c, u_flat], -1)) - act).pow(2).mean()
    l2 = (head_only(u_flat) - act).pow(2).mean()
    opt2.zero_grad(set_to_none=True); (l1 + l2).backward(); opt2.step()
    if (stp + 1) % 1000 == 0:
        print(f"  step {stp+1}  有cond {l1.item():.4f}  只吃u {l2.item():.4f}", flush=True)
for m in (cond_enc, cond_head, head_cond, head_only):
    m.eval()

# ---- 量測 ----
eval_rng = np.random.default_rng(777)
E_traj, E_mask, E_s, E_g, E_act = make_batch(eval_rng, b=512)
with torch.no_grad():
    E_et = etarget(E_traj, E_mask)
    E_c = condvec(E_s, E_g)
E_u = E_et.reshape(E_u_b := E_et.shape[0], -1)


def base_action(with_cond=True):
    with torch.no_grad():
        return head_cond(torch.cat([E_c, E_u], -1)) if with_cond else head_only(E_u)


print("\n" + "=" * 70)
print("=== ① 擾動敏感度（加同樣相對強度的噪聲，看動作變多少）===", flush=True)
a_cond, a_only = base_action(True), base_action(False)
for scale in (0.1, 0.5):
    with torch.no_grad():
        du = torch.randn_like(E_u) * E_u.std() * scale
        dc = torch.randn_like(E_c) * E_c.std() * scale
        pu = head_cond(torch.cat([E_c, E_u + du], -1))
        pc = head_cond(torch.cat([E_c + dc, E_u], -1))
        po = head_only(E_u + du)
    print(f"  噪聲 {scale:>3}×std | 有cond: 擾動u→Δa {(pu-a_cond).abs().mean():.5f}   "
          f"擾動cond→Δa {(pc-a_cond).abs().mean():.5f}   "
          f"| 只吃u: 擾動u→Δa {(po-a_only).abs().mean():.5f}", flush=True)

print("\n=== ② 梯度範數（除以輸入維度才可比）===", flush=True)
u_ = E_u.clone().requires_grad_(True); c_ = E_c.clone().requires_grad_(True)
out = head_cond(torch.cat([c_, u_], -1)).sum()
gu, gc = torch.autograd.grad(out, [u_, c_])
u2 = E_u.clone().requires_grad_(True)
go = torch.autograd.grad(head_only(u2).sum(), [u2])[0]
print(f"  有cond  ||∂a/∂u||/dim {gu.norm()/DIM:.3e}   ||∂a/∂cond||/dim {gc.norm()/COND:.3e}"
      f"   比值 cond/u = {(gc.norm()/COND)/(gu.norm()/DIM):.1f}×", flush=True)
print(f"  只吃u   ||∂a/∂u||/dim {go.norm()/DIM:.3e}"
      f"   <- 對照：head 被迫用 u 時的敏感度", flush=True)
print(f"  ⇒ 有cond 的 u 敏感度是「只吃u」的 {(gu.norm()/go.norm()).item():.3f} 倍", flush=True)

print("\n=== ③ 歸零測試 ===", flush=True)
with torch.no_grad():
    z_u = head_cond(torch.cat([E_c, torch.zeros_like(E_u)], -1))
    z_c = head_cond(torch.cat([torch.zeros_like(E_c), E_u], -1))
print(f"  u 歸零   → 動作偏離 {(z_u-a_cond).abs().mean():.5f}   MSE對真值 {(z_u-E_act).pow(2).mean():.5f}")
print(f"  cond 歸零 → 動作偏離 {(z_c-a_cond).abs().mean():.5f}   MSE對真值 {(z_c-E_act).pow(2).mean():.5f}")
print(f"  都不動   →                        MSE對真值 {(a_cond-E_act).pow(2).mean():.5f}")
print(f"  只吃u的head →                      MSE對真值 {(a_only-E_act).pow(2).mean():.5f}")
print("=" * 70)
print("讀法：若『有cond』組對 u 的敏感度/梯度顯著低於『只吃u』組，")
print("      就證實【是 cond 的存在讓 head 放棄了 u】，而非 u 本身沒有資訊。")
