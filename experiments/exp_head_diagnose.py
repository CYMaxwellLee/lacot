"""action head 學不起來 — 是分類 head 的問題，還是別的？（2026-08-23）

現象: 修好 l_nf 量級之後，LaCoT 的 l_act_anchor（吃【真】e_target 預測 action）
      在 3000 步時仍是 5.08，而 action 的邊際熵（什麼都不學的基準）是 4.7736
      ⇒ head 比「不學」還差。

這支只做一件事: 拿【凍結的真 e_target】餵給三種 head，其他完全一樣，看誰學得動。
  A. DiscretizedActionHead (256 bins, cross-entropy)  <- LaCoT 現在用的
  B. DiscretizedActionHead (32 bins)                  <- 同一個 head，粗一點的 bin
  C. MLP + MSE                                        <- 08-22 那個替身

⛔ 三者用【同一個共同指標】比較才公平: 把預測 decode 回連續 action 算 MSE。
   分類的 nats 跟回歸的平方誤差不能直接比。
基準:
  - 邊際 MSE  = 用整個資料集 action 的平均去猜（什麼都不學）
  - 量化下限  = 把【真 action】丟進 bin 再 decode 回來的 MSE（256/32 bins 各自的天花板）
"""
import os, sys, numpy as np, torch
from torch import nn
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lacot.e_target import PerceiverPooler
from lacot.heads import DiscretizedActionHead

OGB_DATA = os.environ.get("OGBENCH_DATA_DIR", "/archive/cymaxwelllee/data/ogbench")
device = "cuda" if torch.cuda.is_available() else "cpu"
ENV = "pointmaze-medium-navigate-v0"
K = int(os.environ.get("LACOT_K", 4))
D_MODEL, T_CAP, CHUNK, ADIM, B = 256, 16, 4, 2, 64
GEOM_P, TEMP = 0.02, 0.1
STEPS1, STEPS2 = 1500, 5000
WANDER_MAX = 3.0
DIM = K * D_MODEL

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

# ---- 基準 ----
MARGINAL_MSE = float(((ACT - ACT.mean(0)) ** 2).mean())
print(f"\n=== 基準 ===")
print(f"  邊際 MSE（用整體平均猜）        {MARGINAL_MSE:.5f}")
for nb in (256, 32):
    bw = 2.0 / nb
    q = np.clip(np.floor((ACT + 1.0) / bw), 0, nb - 1)
    rec = -1.0 + (q + 0.5) * bw
    print(f"  量化下限 {nb:>3} bins（真 action 進出 bin）  {float(((ACT - rec) ** 2).mean()):.6f}")
print(f"  action 分布: mean {ACT.mean():.3f} std {ACT.std():.3f} "
      f"|a|>0.99 佔比 {float((np.abs(ACT) > 0.99).mean()):.3f}", flush=True)


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
    act = np.stack([ACT[r:r + CHUNK] for r in rows]).astype(np.float32)
    T_ = lambda x: torch.from_numpy(x.astype(np.float32)).to(device)
    return T_(traj), torch.from_numpy(mask).to(device), T_(act)


def sota_mlp(i, h, o, n=2):
    L, p = [], i
    for _ in range(n):
        lin = nn.Linear(p, h); nn.init.xavier_uniform_(lin.weight); nn.init.zeros_(lin.bias)
        L += [lin, nn.GELU(), nn.LayerNorm(h)]; p = h
    lin = nn.Linear(p, o); nn.init.xavier_uniform_(lin.weight); nn.init.zeros_(lin.bias)
    return nn.Sequential(*L, lin)


# ---- stage 1: 跟其他實驗一致的 contrastive e_target，之後凍結 ----
torch.manual_seed(0); rng = np.random.default_rng(0)
traj_enc = sota_mlp(2, 512, 512).to(device)
e_pooler = PerceiverPooler(512, D_MODEL, K, 2, 4).to(device)
sg_c = sota_mlp(2, 512, 512).to(device)
q_pooler = PerceiverPooler(512, D_MODEL, K, 2, 4).to(device)
opt1 = torch.optim.Adam([p for m in (traj_enc, e_pooler, sg_c, q_pooler) for p in m.parameters()], lr=1e-3)
lab = torch.arange(B, device=device)


def etarget(traj, mask):
    b, t, _ = traj.shape
    return e_pooler(traj_enc(traj.reshape(b * t, 2)).reshape(b, t, 512), key_padding_mask=mask)


print("\nstage 1: contrastive e_target ...", flush=True)
for stp in range(STEPS1):
    traj, mask, _ = make_batch(rng)
    b = traj.shape[0]
    rows_s = traj[:, 0]; rows_g = traj[:, -1]
    et = etarget(traj, mask)
    q = q_pooler(torch.stack([sg_c(rows_s), sg_c(rows_g)], 1))
    logits = (F.normalize(q.reshape(b, -1), dim=1) @ F.normalize(et.reshape(b, -1), dim=1).t()) / TEMP
    loss = 0.5 * (F.cross_entropy(logits, lab) + F.cross_entropy(logits.t(), lab))
    opt1.zero_grad(set_to_none=True); loss.backward(); opt1.step()
for m in (traj_enc, e_pooler):
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
print(f"  match-acc {(logits.argmax(1)==lab).float().mean().item():.3f}", flush=True)


class MSEHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = sota_mlp(DIM, 512, CHUNK * ADIM, n=3)

    def forward(self, u):
        return self.net(u.reshape(u.shape[0], -1)).reshape(-1, CHUNK, ADIM)


class MLPDisc(nn.Module):
    """分類 head，但前面補上跟 MSEHead 一樣的 3 層 MLP —— 控制【容量】這個變因。

    ⚠️ 加這個是因為第一版比較不公平: DiscretizedActionHead 只有 self.proj 一層
    nn.Linear（它預期輸入已經被 backbone 處理過），而 MSEHead 有 3 層 MLP。
    不控制容量就沒辦法分辨「分類不行」還是「頭太小」。
    """

    def __init__(self, bins):
        super().__init__()
        self.trunk = sota_mlp(DIM, 512, 512, n=3)
        self.head = DiscretizedActionHead(512, ADIM, CHUNK, bins)

    def forward(self, u):
        return self.head(self.trunk(u.reshape(u.shape[0], -1)))

    def nll(self, logits, act):
        return self.head.nll(logits, act)

    def decode_bins(self, idx):
        return self.head.decode_bins(idx)


heads = {
    "A. Disc 256, 單層 (現用)": DiscretizedActionHead(DIM, ADIM, CHUNK, 256).to(device),
    "B. Disc  32, 單層": DiscretizedActionHead(DIM, ADIM, CHUNK, 32).to(device),
    "C. MSE + 3層MLP (替身)": MSEHead().to(device),
    "D. Disc  32, +3層MLP": MLPDisc(32).to(device),
    "E. Disc 256, +3層MLP": MLPDisc(256).to(device),
}
opts = {k: torch.optim.Adam(v.parameters(), lr=5e-4) for k, v in heads.items()}

# 固定評估批
eval_rng = np.random.default_rng(999)
E_traj, E_mask, E_act = make_batch(eval_rng, b=512)
with torch.no_grad():
    E_et = etarget(E_traj, E_mask)


@torch.no_grad()
def eval_mse(name, head):
    """共同指標：decode 回連續 action 後的 MSE。"""
    head.eval()
    if isinstance(head, MSEHead):
        pred = head(E_et).clamp(-1, 1)
    elif isinstance(head, MLPDisc):
        pred = head.decode_bins(head(E_et).argmax(-1))
    else:
        pred = head.decode_bins(head(E_et.reshape(E_et.shape[0], -1)).argmax(-1))
    head.train()
    return float((pred - E_act).pow(2).mean())


print(f"\nstage 2: 三種 head 吃【同一個真 e_target】，各訓 {STEPS2} 步", flush=True)
print(f"{'step':>6} | " + " | ".join(f"{k:>24}" for k in heads))
print("-" * 132)
CK = [500, 1000, 2000, 3000, 4000, 5000]
for stp in range(1, STEPS2 + 1):
    traj, mask, act = make_batch(rng)
    with torch.no_grad():
        et = etarget(traj, mask)
    for name, head in heads.items():
        if isinstance(head, MSEHead):
            loss = (head(et) - act).pow(2).mean()
        elif isinstance(head, MLPDisc):
            loss = head.nll(head(et), act).mean()
        else:
            loss = head.nll(head(et.reshape(et.shape[0], -1)), act).mean()
        opts[name].zero_grad(set_to_none=True); loss.backward(); opts[name].step()
    if stp in CK:
        vals = [f"{eval_mse(n, h):>24.5f}" for n, h in heads.items()]
        print(f"{stp:>6} | " + " | ".join(vals), flush=True)

print("-" * 132)
print(f"{'基準':>6} | " + " | ".join(f"{MARGINAL_MSE:>24.5f}" for _ in heads) + "   <- 什麼都不學")
print("\n讀法：低於基準才算學到東西。三者差很多 => head 型態是主因；"
      "都爛 => 是 e_target 或訓練設定的問題。")
