"""refine 這個「思考過程」到底在做什麼？（主人 2026-08-23 指定）

問題來自主人一句話：「u 不見得是有害的，那三輪後的 u 設為 0，還可跑嗎」

⚠️ 先更正一個ルナ推錯的：R=0 只有 0.51（比 u 歸零還低）不代表 flow 生的 u 有害。
   看訓練 code：head 只在 `u_target`（真 e_target）與 `us[1..3]`（refine 後）上訓過，
   ⛔ 從沒看過 `us[0]` = flow 剛生、未 refine 的 u。所以 R=0 差是 OOD，不是 u 有毒。

四個 rollout 對照：
  R=3          — 正常推論路徑
  R=3 then 0   — refine 跑完，再把 u 換成 0 餵給 head  <- 主人問的：refine 完的 u 有沒有在出力
  R=0          — flow 直接生的 u（head 沒訓過這種輸入）
  null         — u 從頭就是 0

三件量測（都在同一個訓好的模型上）：
  ① 塌沒塌   — 不同 cond 的 u 彼此還分得開嗎（batch 內兩兩 cosine；越接近 1 = 越塌）
  ② 路徑資訊 — 用 probe 從 u 還原路徑中點，比 flow生的／refine後的／真 e_target
               ⛔ 對照：只用 (s,g) 內插。沒贏過內插 = 對路徑沒有提供額外資訊
  ③ 往哪去   — 每一輪的 ‖u‖、與真 e_target 的 cosine、與 batch 平均的距離
               （靠近 batch 平均 = 塌向一個與 cond 無關的常數）
"""
import os, sys, numpy as np, torch
from torch import nn
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lacot.model import LaCoTActorState
from lacot.e_target import PerceiverPooler
import ogbench

OGB_DATA = os.environ.get("OGBENCH_DATA_DIR", "/archive/cymaxwelllee/data/ogbench")
device = "cuda" if torch.cuda.is_available() else "cpu"
ENV_NAME = "pointmaze-medium-navigate-v0"
K, D_MODEL, T_CAP, COND, CHUNK, ADIM, B = 4, 256, 16, 256, 4, 2, 64
GEOM_P, TEMP, WANDER_MAX = 0.02, 0.1, 3.0
STEPS1 = int(os.environ.get("LACOT_STEPS1", 1500))
STEPS2 = int(os.environ.get("LACOT_STEPS2", 6000))
SEED = int(os.environ.get("LACOT_SEED", 0))
DIM = K * D_MODEL

d = np.load(f"{OGB_DATA}/{ENV_NAME}.npz")
OBS = np.asarray(d["observations"], np.float32)
ACT = np.asarray(d["actions"], np.float32)
TERM = np.asarray(d["terminals"], bool)
N = OBS.shape[0]
ends = np.flatnonzero(TERM); starts = np.concatenate([[0], ends[:-1] + 1])
traj_end = np.empty(N, np.int64)
for s0, e0 in zip(starts, ends):
    traj_end[s0:e0 + 1] = e0
mu, sd = OBS.mean(0), OBS.std(0) + 1e-6
MU = torch.tensor(mu, device=device); SD = torch.tensor(sd, device=device)
print(f"device {device} | K={K} seed={SEED}", flush=True)


def make_batch(rng, b=B):
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
    rows, goals = np.array(rows), np.array(goals)
    idxs = [np.unique(np.linspace(rows[i], goals[i], min(T_CAP, goals[i] - rows[i] + 1)).round().astype(int)) for i in range(b)]
    Tmax = max(len(ix) for ix in idxs)
    traj = np.zeros((b, Tmax, 2), np.float32); mask = np.ones((b, Tmax), bool)
    mids = np.zeros((b, 2), np.float32)
    for i, ix in enumerate(idxs):
        traj[i, :len(ix)] = (OBS[ix] - mu) / sd; mask[i, :len(ix)] = False
        mids[i] = (OBS[ix[len(ix) // 2]] - mu) / sd
    s = (OBS[rows] - mu) / sd; g = (OBS[goals] - mu) / sd
    act = np.stack([ACT[r:r + CHUNK] for r in rows]).astype(np.float32)
    T_ = lambda x: torch.from_numpy(x.astype(np.float32)).to(device)
    return T_(traj), torch.from_numpy(mask).to(device), T_(s), T_(g), T_(act), T_(mids)


def mlp(i, h, o, n=2):
    L, p = [], i
    for _ in range(n):
        lin = nn.Linear(p, h); nn.init.xavier_uniform_(lin.weight); nn.init.zeros_(lin.bias)
        L += [lin, nn.GELU(), nn.LayerNorm(h)]; p = h
    lin = nn.Linear(p, o); nn.init.xavier_uniform_(lin.weight); nn.init.zeros_(lin.bias)
    return nn.Sequential(*L, lin)


torch.manual_seed(SEED); rng = np.random.default_rng(SEED)
model = LaCoTActorState(state_dim=2, d_model=D_MODEL, k=K, action_dim=ADIM,
                        chunk_len=CHUNK, cond_dim=COND).to(device)
sg_c = mlp(2, 512, 512).to(device)
q_pooler = PerceiverPooler(512, D_MODEL, K, 2, 4).to(device)
opt1 = torch.optim.Adam(
    list(model.traj_enc.parameters()) + list(model.e_pooler.parameters())
    + list(sg_c.parameters()) + list(q_pooler.parameters()), lr=1e-3)
lab = torch.arange(B, device=device)

print("stage 1 ...", flush=True)
for stp in range(STEPS1):
    traj, mask, s, g, _, _ = make_batch(rng)
    et = model.e_target(traj, mask)
    q = q_pooler(torch.stack([sg_c(s), sg_c(g)], 1))
    logits = (F.normalize(q.reshape(B, -1), dim=1) @ F.normalize(et.reshape(B, -1), dim=1).t()) / TEMP
    loss = 0.5 * (F.cross_entropy(logits, lab) + F.cross_entropy(logits.t(), lab))
    opt1.zero_grad(set_to_none=True); loss.backward(); opt1.step()
model.freeze_front_end()
print(f"  match-acc {(logits.argmax(1)==lab).float().mean().item():.3f}", flush=True)

train_mods = [model.cond_enc, model.cond_head, model.flow, model.refine, model.action_head]
opt2 = torch.optim.Adam([p for m in train_mods for p in m.parameters()], lr=5e-4)
ZERO = torch.zeros(B, K, D_MODEL, device=device)
print(f"stage 2 ({STEPS2} 步) ...", flush=True)
for stp in range(STEPS2):
    traj, mask, s, g, act, _ = make_batch(rng)
    with torch.no_grad():
        et = model.e_target(traj, mask)
    cond = model.encode_cond(s, g)
    total, parts = model.losses_given(cond, et, act, rounds=3, lam_cons=0.5)
    l_null = model.action_head.nll(
        model.action_head(torch.cat([cond, ZERO.reshape(B, -1)], -1)), act).mean()
    (total + l_null).backward()
    torch.nn.utils.clip_grad_norm_([p for m in train_mods for p in m.parameters()], 1.0)
    opt2.step(); opt2.zero_grad(set_to_none=True)
    if (stp + 1) % 2000 == 0:
        print(f"  step {stp+1}  anchor {parts['l_act_anchor']:.4f} refine {parts['l_act_refine']:.4f}", flush=True)
model.eval()

# ---------- 量測 ----------
er = np.random.default_rng(4242)
E_traj, E_mask, E_s, E_g, E_act, E_mid = make_batch(er, b=512)
with torch.no_grad():
    E_et = model.e_target(E_traj, E_mask)
    E_c = model.encode_cond(E_s, E_g)
    u0 = model.sample_u(E_c)
    us = [u0]
    u = u0
    for _ in range(3):
        u = model.refine(E_c, u)
        us.append(u)

flat = lambda x: x.reshape(x.shape[0], -1)


def pair_cos(x):
    """batch 內兩兩 cosine 的平均 —— 越接近 1 代表大家長越像（塌）。"""
    z = F.normalize(flat(x), dim=1)
    m = z @ z.t()
    n = z.shape[0]
    return ((m.sum() - m.diag().sum()) / (n * (n - 1))).item()


print("\n" + "=" * 70)
print("=== ① 塌沒塌（batch 內兩兩 cosine，越接近 1 越塌）===", flush=True)
print(f"  真 e_target        {pair_cos(E_et):.4f}   <- 參考：真值本身的相似度")
for r, uu in enumerate(us):
    tag = "flow 生的" if r == 0 else f"refine 第 {r} 輪"
    print(f"  {tag:<16} {pair_cos(uu):.4f}")

print("\n=== ③ 往哪去（每輪的形狀變化）===", flush=True)
mean_et = E_et.mean(0, keepdim=True)
print(f"  {'':<16} {'‖u‖':>8} {'cos(真et)':>10} {'離batch平均':>12}")
for r, uu in enumerate(us):
    tag = "flow 生的" if r == 0 else f"refine 第 {r} 輪"
    c = F.cosine_similarity(flat(uu), flat(E_et), dim=1).mean().item()
    dist_mean = (flat(uu) - flat(uu).mean(0, keepdim=True)).norm(dim=1).mean().item()
    print(f"  {tag:<16} {flat(uu).norm(dim=1).mean():>8.1f} {c:>10.3f} {dist_mean:>12.1f}")
print(f"  {'真 e_target':<16} {flat(E_et).norm(dim=1).mean():>8.1f} {1.0:>10.3f} "
      f"{(flat(E_et)-flat(E_et).mean(0,keepdim=True)).norm(dim=1).mean():>12.1f}")

# ---------- ② 路徑資訊 ----------
print("\n=== ② 路徑資訊（從 u 還原路徑中點的 MSE，越低越好）===", flush=True)
probes = {}
for name in ("flow", "refined", "etarget", "interp"):
    probes[name] = (mlp(4, 512, 2, n=3) if name == "interp" else mlp(DIM, 512, 2, n=3)).to(device)
opt_p = torch.optim.Adam([p for m in probes.values() for p in m.parameters()], lr=1e-3)
for stp in range(1500):
    traj, mask, s, g, _, mid = make_batch(rng)
    with torch.no_grad():
        et = model.e_target(traj, mask)
        c = model.encode_cond(s, g)
        uf = model.sample_u(c)
        ur = uf
        for _ in range(3):
            ur = model.refine(c, ur)
    l = ((probes["flow"](flat(uf)) - mid).pow(2).mean()
         + (probes["refined"](flat(ur)) - mid).pow(2).mean()
         + (probes["etarget"](flat(et)) - mid).pow(2).mean()
         + (probes["interp"](torch.cat([s, g], 1)) - mid).pow(2).mean())
    opt_p.zero_grad(set_to_none=True); l.backward(); opt_p.step()
for m in probes.values():
    m.eval()
with torch.no_grad():
    mse = lambda a, b: (a - b).pow(2).mean().item()
    base = mse(E_mid.mean(0, keepdim=True).expand_as(E_mid), E_mid)
    r_int = mse(probes["interp"](torch.cat([E_s, E_g], 1)), E_mid)
    r_et = mse(probes["etarget"](flat(E_et)), E_mid)
    r_fl = mse(probes["flow"](flat(us[0])), E_mid)
    r_rf = mse(probes["refined"](flat(us[3])), E_mid)
print(f"  猜平均（基準）        {base:.4f}")
print(f"  只用 (s,g) 內插       {r_int:.4f}   <- ⛔ 關鍵對照")
print(f"  真 e_target           {r_et:.4f}")
print(f"  flow 生的 u           {r_fl:.4f}")
print(f"  refine 三輪後的 u     {r_rf:.4f}")

# ---------- rollout ----------
os.environ.setdefault("OGBENCH_DATA_DIR", OGB_DATA)
env, _, _ = ogbench.make_env_and_datasets(ENV_NAME)
MAXH = int(os.environ.get("LACOT_EVAL_MAXH", env.spec.max_episode_steps or 1000))
N_TASKS = len(env.unwrapped.task_infos)
EPISODES = int(os.environ.get("LACOT_EVAL_EPISODES", 20))
print(f"\n=== rollout（官方標準 MAXH={MAXH}, {N_TASKS}x{EPISODES} 集）===", flush=True)


def normstate(x):
    return ((torch.tensor(np.asarray(x, np.float32), device=device) - MU) / SD)[None]


@torch.no_grad()
def rollout(mode, tag):
    succ = ep = 0
    for task in range(1, N_TASKS + 1):
        for sd_ in range(EPISODES):
            obs, info = env.reset(seed=1000 * task + sd_, options={"task_id": task, "render_goal": False})
            goal = info["goal"]; success = False; steps = 0
            torch.manual_seed(7 * task + sd_)
            while steps < MAXH and not success:
                c = model.encode_cond(normstate(obs), normstate(goal))
                if mode == "null":
                    uu = torch.zeros(1, K, D_MODEL, device=device)
                else:
                    uu = model.sample_u(c)
                    if mode in ("r3", "r3_then0"):
                        for _ in range(3):
                            uu = model.refine(c, uu)
                    if mode == "r3_then0":
                        uu = torch.zeros_like(uu)      # ← 主人問的：refine 跑完再換成 0
                a_chunk = model.act(c, u=uu)[0].cpu().numpy()
                for a in np.clip(a_chunk, -1, 1).astype(np.float32):
                    obs, rew, term, trunc, info = env.step(a)
                    steps += 1
                    if info.get("success"):
                        success = True
                    if success or term or trunc or steps >= MAXH:
                        break
            succ += int(success); ep += 1
    print(f"  {tag:<34} {succ:>3}/{ep} = {succ/ep:.3f}", flush=True)
    return succ / ep


res = {}
res["r3"] = rollout("r3", "R=3（正常）")
res["r3_then0"] = rollout("r3_then0", "R=3 跑完再把 u 換成 0")
res["r0"] = rollout("r0", "R=0（flow 直接生，head 沒訓過）")
res["null"] = rollout("null", "null（u 從頭就是 0）")
print("=" * 70)
print(f"R=3 {res['r3']:.3f} vs R=3-then-0 {res['r3_then0']:.3f}"
      f"   差 {res['r3']-res['r3_then0']:+.3f}  <- refine 後的 u 到底有沒有在出力")
print("=" * 70)
