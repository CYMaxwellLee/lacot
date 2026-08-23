"""e_target 到底編了什麼？—— 是路徑，還是只有 (s, g)？（主人 2026-08-23 同意跑）

懷疑：stage 1 的 contrastive 目標是「讓 e_target 能被 (s,g) 認出來」，
      於是它學到的正是【我是哪兩點之間的路】—— 而那正好是 cond 已經有的資訊。
      如果成立，就解釋了為什麼 head 拿到 cond 之後就放掉 u。

三個檢查（每個都帶對照，沒有對照的數字不算數）：

  A. 從 e_target 還原 (s, g)     —— 對照：猜資料集平均
     很準 => 它把容量花在編起終點

  B. 從 e_target 還原【中間路徑點】—— 對照①猜平均　對照②【用 (s,g) 線性內插】
     ⛔ 對照② 才是關鍵：如果 e_target 預測中點的能力沒有超過「拿 (s,g) 內插」，
        就代表它對路徑形狀【沒有提供任何額外資訊】。

  C. 同 (s,g) 但走不同路的樣本對，e_target 分不分得開 —— 對照：走同一條路的樣本對
     pointmaze-medium 中間有一個口字型的環（主人指出），左右兩點之間可以繞上面
     或繞下面。若兩種走法的 e_target 幾乎一樣，就實錘：它把不同的路壓成同一個東西。

讀法：A 準 + B 沒贏過內插 + C 分不開 => 懷疑成立，該換 stage 1 的訓練目標。
"""
import os, sys, numpy as np, torch
from torch import nn
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lacot.e_target import PerceiverPooler

OGB_DATA = os.environ.get("OGBENCH_DATA_DIR", "/archive/cymaxwelllee/data/ogbench")
device = "cuda" if torch.cuda.is_available() else "cpu"
ENV = "pointmaze-medium-navigate-v0"
K, D_MODEL, T_CAP, B = 4, 256, 16, 64
GEOM_P, TEMP, WANDER_MAX = 0.02, 0.1, 3.0
STEPS1, PROBE_STEPS = 1500, 2000

d = np.load(f"{OGB_DATA}/{ENV}.npz")
OBS = np.asarray(d["observations"], np.float32)
TERM = np.asarray(d["terminals"], bool)
N = OBS.shape[0]
ends = np.flatnonzero(TERM); starts = np.concatenate([[0], ends[:-1] + 1])
traj_end = np.empty(N, np.int64)
for s0, e0 in zip(starts, ends):
    traj_end[s0:e0 + 1] = e0
mu, sd = OBS.mean(0), OBS.std(0) + 1e-6
print(f"device {device} | K={K}", flush=True)


def sample_rows(rng, b):
    rows, goals = [], []
    while len(rows) < b:
        r = int(rng.integers(0, N)); te = int(traj_end[r])
        if te - r < 8:
            continue
        gr = min(r + int(rng.geometric(GEOM_P)), te)
        if gr - r < 8:
            continue
        path = OBS[r:gr + 1]
        if np.linalg.norm(np.diff(path, axis=0), axis=1).sum() / (np.linalg.norm(path[-1] - path[0]) + 1e-6) > WANDER_MAX:
            continue
        rows.append(r); goals.append(gr)
    return np.array(rows), np.array(goals)


def build(rows, goals):
    b = len(rows)
    idxs = [np.unique(np.linspace(rows[i], goals[i], min(T_CAP, goals[i] - rows[i] + 1)).round().astype(int)) for i in range(b)]
    Tmax = max(len(ix) for ix in idxs)
    traj = np.zeros((b, Tmax, 2), np.float32); mask = np.ones((b, Tmax), bool)
    mids = np.zeros((b, 2), np.float32)
    for i, ix in enumerate(idxs):
        traj[i, :len(ix)] = (OBS[ix] - mu) / sd; mask[i, :len(ix)] = False
        mids[i] = (OBS[ix[len(ix) // 2]] - mu) / sd           # 路徑中點（正規化後）
    s = (OBS[rows] - mu) / sd; g = (OBS[goals] - mu) / sd
    T_ = lambda x: torch.from_numpy(x.astype(np.float32)).to(device)
    return T_(traj), torch.from_numpy(mask).to(device), T_(s), T_(g), T_(mids)


def mlp(i, h, o, n=2):
    L, p = [], i
    for _ in range(n):
        lin = nn.Linear(p, h); nn.init.xavier_uniform_(lin.weight); nn.init.zeros_(lin.bias)
        L += [lin, nn.GELU(), nn.LayerNorm(h)]; p = h
    lin = nn.Linear(p, o); nn.init.xavier_uniform_(lin.weight); nn.init.zeros_(lin.bias)
    return nn.Sequential(*L, lin)


# ---- stage 1：跟正式訓練完全一樣的 contrastive e_target ----
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
    r, gl = sample_rows(rng, B)
    traj, mask, s, g, _ = build(r, gl)
    et = etarget(traj, mask); q = q_pooler(torch.stack([sg_c(s), sg_c(g)], 1))
    logits = (F.normalize(q.reshape(B, -1), dim=1) @ F.normalize(et.reshape(B, -1), dim=1).t()) / TEMP
    loss = 0.5 * (F.cross_entropy(logits, lab) + F.cross_entropy(logits.t(), lab))
    opt1.zero_grad(set_to_none=True); loss.backward(); opt1.step()
for m in (traj_enc, e_pooler):
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
print(f"  match-acc {(logits.argmax(1)==lab).float().mean().item():.3f}", flush=True)

DIM = K * D_MODEL
probe_sg = mlp(DIM, 512, 4, n=3).to(device)       # e_target -> (s,g)
probe_mid = mlp(DIM, 512, 2, n=3).to(device)      # e_target -> 中間點
probe_interp = mlp(4, 512, 2, n=3).to(device)     # ⛔ 對照：只用 (s,g) -> 中間點
opt_p = torch.optim.Adam(
    [p for m in (probe_sg, probe_mid, probe_interp) for p in m.parameters()], lr=1e-3)

print(f"probe: 三個探針各訓 {PROBE_STEPS} 步 ...", flush=True)
for stp in range(PROBE_STEPS):
    r, gl = sample_rows(rng, B)
    traj, mask, s, g, mid = build(r, gl)
    with torch.no_grad():
        et = etarget(traj, mask).reshape(B, -1)
    sg = torch.cat([s, g], 1)
    l = ((probe_sg(et) - sg).pow(2).mean()
         + (probe_mid(et) - mid).pow(2).mean()
         + (probe_interp(sg) - mid).pow(2).mean())
    opt_p.zero_grad(set_to_none=True); l.backward(); opt_p.step()
for m in (probe_sg, probe_mid, probe_interp):
    m.eval()

# ---- 評估 ----
er = np.random.default_rng(4242)
r, gl = sample_rows(er, 1024)
traj, mask, s, g, mid = build(r, gl)
with torch.no_grad():
    ET = etarget(traj, mask)
    et = ET.reshape(ET.shape[0], -1)
    sg = torch.cat([s, g], 1)
    mse = lambda a, b: (a - b).pow(2).mean().item()
    base_sg = mse(sg.mean(0, keepdim=True).expand_as(sg), sg)
    base_mid = mse(mid.mean(0, keepdim=True).expand_as(mid), mid)
    p_sg, p_mid, p_int = mse(probe_sg(et), sg), mse(probe_mid(et), mid), mse(probe_interp(sg), mid)

print("\n" + "=" * 64)
print("=== A. 從 e_target 還原 (s, g) ===")
print(f"  猜平均（基準）      {base_sg:.4f}")
print(f"  從 e_target 還原    {p_sg:.4f}   解釋掉 {100*(1-p_sg/base_sg):.1f}% 的變異")
print("\n=== B. 從 e_target 還原【中間路徑點】===")
print(f"  猜平均（基準）      {base_mid:.4f}")
print(f"  只用 (s,g) 內插     {p_int:.4f}   <- 關鍵對照")
print(f"  從 e_target 還原    {p_mid:.4f}")
if p_mid < p_int * 0.9:
    print(f"  => e_target 比內插好 {100*(1-p_mid/p_int):.1f}% ：它確實帶了額外的路徑資訊")
else:
    print(f"  => e_target 沒有贏過內插（{p_mid:.4f} vs {p_int:.4f}）：對路徑形狀沒有提供額外資訊")

# ---- C. 同 (s,g) 不同路 ----
print("\n=== C. 同 (s,g) 但走不同路的樣本對 ===")
with torch.no_grad():
    D = torch.cdist(sg, sg); D.fill_diagonal_(1e9)
    close = D < 0.35                                  # (s,g) 幾乎相同
    ii, jj = torch.nonzero(close, as_tuple=True)
    keep = ii < jj
    ii, jj = ii[keep], jj[keep]
    if len(ii) < 10:
        print(f"  只找到 {len(ii)} 對 (s,g) 相近的樣本，太少，跳過")
    else:
        mid_gap = (mid[ii] - mid[jj]).norm(dim=1)     # 中間點差多遠 = 走的路差多遠
        same = mid_gap < mid_gap.median()
        diff = mid_gap >= mid_gap.median()
        cos = lambda a, b: F.cosine_similarity(a.reshape(a.shape[0], -1), b.reshape(b.shape[0], -1), dim=1)
        c_same = cos(ET[ii[same]], ET[jj[same]]).mean().item()
        c_diff = cos(ET[ii[diff]], ET[jj[diff]]).mean().item()
        print(f"  找到 {len(ii)} 對 (s,g) 相近的")
        print(f"  走【同一條路】的（中點差 {mid_gap[same].mean():.2f}）  e_target cosine {c_same:.3f}")
        print(f"  走【不同路】的  （中點差 {mid_gap[diff].mean():.2f}）  e_target cosine {c_diff:.3f}")
        print(f"  差距 {c_same - c_diff:+.3f}")
        if c_same - c_diff < 0.05:
            print("  => 分不開：走不同路的 e_target 跟走同一條路的一樣像 —— 它沒編路徑")
        else:
            print("  => 分得開：e_target 確實區分了不同走法")
print("=" * 64)
