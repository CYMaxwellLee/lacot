"""為什麼 u 練不好 — 檢驗三個假設（主人 2026-08-23 問「u 沒練好可以檢查一下為什麼」）。

假設 H1（高維典型集）: u 是 K*D = 64*256 = 16384 維，base 是標準高斯。
    在 16384 維，randn 的樣本幾乎全部落在半徑 sqrt(16384)=128 的薄殼上，
    ⇒ 採樣【永遠採不到 mode】。若真 e_target 靠近 mode，sample 必然偏離。
    預測: cosine(sample_T, true_et) 隨 temperature T 下降而單調上升，
          T=0（z=0，正好是 flow 的 mode）時最高。

假設 H2（分布其實是 deterministic 的）: 給定 (s,g)，maze 裡的路徑幾乎唯一
    ⇒ p(e_target|cond) 接近 delta ⇒ 用「採樣」這個工具本身就是錯的，
      直接 regression 應該贏過 flow sample。
    預測: cosine(MLP_regress(cond), true_et) >> cosine(flow.sample(cond), true_et)
    控制: 同時量「真實資料裡相似 (s,g) 的 e_target 有多分散」——
          若那個分散度很小，就證實 p(e|cond) 真的窄。

假設 H3（尺度不匹配）: e_target 是 InfoNCE 學的，只約束方向（F.normalize）不約束長度
    ⇒ 它的 norm 可能離 flow base 的 128 很遠，flow 得學一個大 scaling。
    預測: ||true_et|| 跟 sqrt(16384)=128 差很多。

⛔ 每一項都印出對照組，沒有對照的數字不算數。
"""
import os, sys, math, numpy as np, torch
from torch import nn
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lacot.e_target import PerceiverPooler
from lacot.nf_head import Flow

OGB_DATA = os.environ.get("OGBENCH_DATA_DIR", "/archive/cymaxwelllee/data/ogbench")
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device, flush=True)

ENV = "pointmaze-medium-navigate-v0"
d = np.load(f"{OGB_DATA}/{ENV}.npz")
OBS = np.asarray(d["observations"], np.float32)
TERM = np.asarray(d["terminals"], bool)
N = OBS.shape[0]
ends = np.flatnonzero(TERM); starts = np.concatenate([[0], ends[:-1] + 1])
traj_end = np.empty(N, np.int64)
for s0, e0 in zip(starts, ends):
    traj_end[s0:e0 + 1] = e0
mu, sd = OBS.mean(0), OBS.std(0) + 1e-6

B, T_CAP, D_MODEL, K, GEOM_P, TEMP, COND = 64, 16, 256, 64, 0.02, 0.1, 256
DIM = K * D_MODEL
WANDER_MAX = 3.0


def make_batch(rng, b=B):
    rows, goals = [], []
    while len(rows) < b:
        r = int(rng.integers(0, N)); te = int(traj_end[r])
        if te - r < 4:
            continue
        gr = min(r + int(rng.geometric(GEOM_P)), te)
        if gr - r < 4:
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
    T_ = lambda x: torch.from_numpy(x.astype(np.float32)).to(device)
    return T_(traj), torch.from_numpy(mask).to(device), T_(s), T_(g), rows, goals


def sota_mlp(i, h, o, n=2):
    L, p = [], i
    for _ in range(n):
        lin = nn.Linear(p, h); nn.init.xavier_uniform_(lin.weight); nn.init.zeros_(lin.bias)
        L += [lin, nn.GELU(), nn.LayerNorm(h)]; p = h
    lin = nn.Linear(p, o); nn.init.xavier_uniform_(lin.weight); nn.init.zeros_(lin.bias)
    return nn.Sequential(*L, lin)


torch.manual_seed(0); rng = np.random.default_rng(0)
traj_enc = sota_mlp(2, 512, 512).to(device)
e_pooler = PerceiverPooler(512, D_MODEL, K, 2, 4).to(device)
sg_c = sota_mlp(2, 512, 512).to(device)
q_pooler = PerceiverPooler(512, D_MODEL, K, 2, 4).to(device)
opt1 = torch.optim.Adam([p for m in (traj_enc, e_pooler, sg_c, q_pooler) for p in m.parameters()], lr=1e-3)
lab = torch.arange(B, device=device)


def etarget(traj, mask):
    Bc, Tc, _ = traj.shape
    return e_pooler(traj_enc(traj.reshape(Bc * Tc, 2)).reshape(Bc, Tc, 512), key_padding_mask=mask)


print("stage 1: contrastive e_target ...", flush=True)
for stp in range(1500):
    traj, mask, s, g, _, _ = make_batch(rng)
    et = etarget(traj, mask); q = q_pooler(torch.stack([sg_c(s), sg_c(g)], 1))
    logits = (F.normalize(q.reshape(B, -1), dim=1) @ F.normalize(et.reshape(B, -1), dim=1).t()) / TEMP
    loss = 0.5 * (F.cross_entropy(logits, lab) + F.cross_entropy(logits.t(), lab))
    opt1.zero_grad(set_to_none=True); loss.backward(); opt1.step()
for m in (traj_enc, e_pooler):
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
print(f"  e_target match-acc {(logits.argmax(1)==lab).float().mean().item():.3f}", flush=True)

# ---- stage 2: flow  vs  MLP regressor（對照組），完全相同的 cond、相同步數 ----
cond_enc = sota_mlp(2, 512, 512).to(device)
cond_head = sota_mlp(1024, 512, COND).to(device)
flow = Flow(token_dim=D_MODEL, seq_len=K, n_blocks=4, cond_dim=COND).to(device)
reg = sota_mlp(COND, 1024, DIM, n=3).to(device)          # ← 對照組：直接 regress
opt_f = torch.optim.Adam([p for m in (cond_enc, cond_head, flow) for p in m.parameters()], lr=5e-4)
opt_r = torch.optim.Adam(reg.parameters(), lr=5e-4)


def condvec(s, g):
    return cond_head(torch.cat([cond_enc(s), cond_enc(g)], 1))


print("stage 2: flow (NLL)  ‖  regressor (MSE)  — 同 cond、同步數 ...", flush=True)
for stp in range(2000):
    traj, mask, s, g, _, _ = make_batch(rng)
    with torch.no_grad():
        et = etarget(traj, mask)
    cond = condvec(s, g)
    l_nf = flow.nll(et, cond) / DIM
    opt_f.zero_grad(set_to_none=True); l_nf.backward()
    torch.nn.utils.clip_grad_norm_([p for m in (cond_enc, cond_head, flow) for p in m.parameters()], 1.0)
    opt_f.step()
    l_reg = (reg(cond.detach()).reshape(-1, K, D_MODEL) - et).pow(2).mean()
    opt_r.zero_grad(set_to_none=True); l_reg.backward(); opt_r.step()
    if (stp + 1) % 500 == 0:
        print(f"  step {stp+1}  l_nf/dim {l_nf.item():.4f}  l_reg {l_reg.item():.4f}", flush=True)
for m in (cond_enc, cond_head, flow, reg):
    m.eval()


@torch.no_grad()
def sample_at_T(cond, T):
    """從 z = T * randn 反流回來。T=0 就是 flow 的 mode。"""
    n = cond.shape[0]
    z = torch.randn(n, K, D_MODEL, device=device) * T
    u = z
    for i in reversed(range(len(flow.blocks))):
        if i < len(flow.blocks) - 1:
            u = flow.perm.inverse(u)
        u = flow.blocks[i].inverse(u, cond)
    return u


def cos(a, b):
    return F.cosine_similarity(a.reshape(a.shape[0], -1), b.reshape(b.shape[0], -1), dim=1)


def ccos(a, b):
    """centered cosine — 扣掉各自 batch 平均，避免共同 offset 灌水。"""
    A = a.reshape(a.shape[0], -1); Bb = b.reshape(b.shape[0], -1)
    return F.cosine_similarity(A - A.mean(0, keepdim=True), Bb - Bb.mean(0, keepdim=True), dim=1)


print("\n" + "=" * 62, flush=True)
with torch.no_grad():
    traj, mask, s, g, rows, goals = make_batch(rng, b=256)
    et = etarget(traj, mask)
    cond = condvec(s, g)

    # ---------- H3: 尺度 ----------
    print("=== H3  尺度是否匹配 ===", flush=True)
    n_et = et.reshape(et.shape[0], -1).norm(dim=1)
    print(f"  ||true_et||      mean {n_et.mean():.1f}  std {n_et.std():.1f}")
    print(f"  flow base 期望值 sqrt(K*D) = {math.sqrt(DIM):.1f}   <- 標準高斯的典型半徑")
    print(f"  比值 ||et|| / 128 = {(n_et.mean()/math.sqrt(DIM)).item():.3f}")

    # ---------- H2 控制組: p(e|cond) 到底有多窄 ----------
    print("\n=== H2-控制  真實資料裡 p(e|cond) 有多窄 ===", flush=True)
    sg = torch.cat([s, g], 1)
    dist = torch.cdist(sg, sg); dist.fill_diagonal_(1e9)
    nn_d, nn_i = dist.min(1)
    close = nn_d < 0.35                      # (s,g) 幾乎重合的 pair
    if close.sum() >= 3:
        c_et = cos(et[close], et[nn_i[close]])
        print(f"  (s,g) 幾乎相同的 pair: {int(close.sum())} 組, 它們的 e_target cosine "
              f"mean {c_et.mean():.3f}  <- 接近 1 = 分布窄 = 幾乎 deterministic")
    else:
        print(f"  這批只有 {int(close.sum())} 組夠近的 pair，改看整體:")
    rand_pair = cos(et, et[torch.randperm(et.shape[0], device=device)])
    print(f"  隨機配對的 e_target cosine  mean {rand_pair.mean():.3f}  <- 基準線（不相關該接近 0）")

    # ---------- H1: temperature 掃描 ----------
    print("\n=== H1  採樣溫度掃描（cosine 對真 e_target，越高越好）===", flush=True)
    print(f"  {'T':>6} | {'cosine':>8} | {'centered':>9} | {'||u||':>7}")
    print("  " + "-" * 40)
    for T in (1.0, 0.7, 0.5, 0.3, 0.2, 0.1, 0.05, 0.0):
        u = sample_at_T(cond, T)
        print(f"  {T:>6.2f} | {cos(u, et).mean().item():>8.3f} | {ccos(u, et).mean().item():>9.3f} | "
              f"{u.reshape(u.shape[0],-1).norm(dim=1).mean().item():>7.1f}", flush=True)

    # ---------- H2: regressor 對照 ----------
    print("\n=== H2  對照組：直接 regression（同 cond、同步數）===", flush=True)
    ur = reg(cond).reshape(-1, K, D_MODEL)
    print(f"  regressor        cosine {cos(ur, et).mean().item():.3f}   centered {ccos(ur, et).mean().item():.3f}")
    u1 = sample_at_T(cond, 1.0); u0 = sample_at_T(cond, 0.0)
    print(f"  flow sample T=1  cosine {cos(u1, et).mean().item():.3f}   centered {ccos(u1, et).mean().item():.3f}")
    print(f"  flow mode  T=0   cosine {cos(u0, et).mean().item():.3f}   centered {ccos(u0, et).mean().item():.3f}")

    # ---------- 退化控制：conditioning 真的有用嗎 ----------
    print("\n=== 控制  conditioning 是否真的有用（打亂 cond 應該變差）===", flush=True)
    shuf = cond[torch.randperm(cond.shape[0], device=device)]
    print(f"  regressor(打亂 cond) cosine {cos(reg(shuf).reshape(-1,K,D_MODEL), et).mean().item():.3f}  <- 該掉下來")
    print(f"  flow mode(打亂 cond) cosine {cos(sample_at_T(shuf, 0.0), et).mean().item():.3f}  <- 該掉下來")

    # ---------- 同一 cond 重複採樣的自我分散 ----------
    print("\n=== 補充  同一個 cond 採樣兩次，樣本彼此有多像 ===", flush=True)
    a1 = sample_at_T(cond, 1.0); a2 = sample_at_T(cond, 1.0)
    print(f"  sample vs sample cosine {cos(a1, a2).mean().item():.3f}  "
          f"(若遠低於 sample-vs-true，代表噪聲主導而非 cond 主導)")
print("=" * 62, flush=True)
