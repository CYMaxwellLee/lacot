"""ORACLE 重做 — 這次用【真的 LaCoT】＋【官方 eval 標準】＋【closed / open loop 兩種】。

為什麼要重做（主人 2026-08-23 要「Oracle 版本要確定是對的」）:
  ① 08-22 的 oracle 用的是腳本自己寫的 ActionMLP + MSE，⛔ 不是 LaCoT 的
     DiscretizedActionHead（分類）。那天的 100% 是【替身】的 100%。
  ② eval 自己把 horizon 砍成 500、每 task 只跑 6 集；官方是 1000 / 20 集。
  ③ oracle 每 4 步就重問一次 env 內建 BFS = closed-loop。那測的是反應式導航，
     不是「想一條路再照著走」。

這支的四個對照:
  ORACLE-closed : e_target 每個 chunk 用當下位置重算 BFS 路徑  <- 舊版的做法
  ORACLE-open   : e_target 只在【起點算一次】，全程沿用          <- 想一次、照著走
  LaCoT R=0/R=3 : flow 採樣的 u（closed，每 chunk 重採）         <- 真實推論路徑
  GCBC / null   : 地板

  ⚠️ open-loop 的定義: 【u 只算一次】，但 cond（我在哪）每個 chunk 仍然更新 ——
     因為執行時當然知道自己的位置，被凍住的是「想出來的那條路」。
  ⇒ ORACLE-closed 與 ORACLE-open 的【差距】本身就是指標:
     差距小 = 那條想出來的路禁得起放著不管 = reasoning 有品質。

K=4（dim 1024），因為 exp_u_dim_sweep 量到 K=64 的 16384 維會把 flow 壓死
（flow mode 0.523），K=4 時 flow 追到 0.934 而 e_target match-acc 不變。
"""
import os, sys, numpy as np, torch
from torch import nn
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lacot.model import LaCoTActorState
import ogbench

OGB_DATA = os.environ.get("OGBENCH_DATA_DIR", "/archive/cymaxwelllee/data/ogbench")
device = "cuda" if torch.cuda.is_available() else "cpu"
ENV_NAME = "pointmaze-medium-navigate-v0"

K = int(os.environ.get("LACOT_K", 4))
D_MODEL, T_CAP, COND, CHUNK, ADIM = 256, 16, 256, 4, 2
B, GEOM_P, TEMP = 64, 0.02, 0.1
STEPS1 = int(os.environ.get("LACOT_STEPS1", 1500))
STEPS2 = int(os.environ.get("LACOT_STEPS2", 4000))
WANDER_MAX = 3.0
print(f"device {device} | K={K} dim={K*D_MODEL} | stage1 {STEPS1} stage2 {STEPS2}", flush=True)

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


torch.manual_seed(0); rng = np.random.default_rng(0)
model = LaCoTActorState(state_dim=2, d_model=D_MODEL, k=K, action_dim=ADIM,
                        chunk_len=CHUNK, cond_dim=COND).to(device)

# ---- stage 1: contrastive e_target（跟前面實驗一致），之後凍結 ----
q_pooler_in = nn.Sequential().to(device)
sg_c = nn.Sequential(nn.Linear(2, 512), nn.GELU(), nn.LayerNorm(512),
                     nn.Linear(512, 512), nn.GELU(), nn.LayerNorm(512)).to(device)
from lacot.e_target import PerceiverPooler
q_pooler = PerceiverPooler(512, D_MODEL, K, 2, 4).to(device)
opt1 = torch.optim.Adam(
    list(model.traj_enc.parameters()) + list(model.e_pooler.parameters())
    + list(sg_c.parameters()) + list(q_pooler.parameters()), lr=1e-3)
lab = torch.arange(B, device=device)
print("stage 1: contrastive e_target ...", flush=True)
accs = []
for stp in range(STEPS1):
    traj, mask, s, g, _ = make_batch(rng)
    et = model.e_target(traj, mask)
    q = q_pooler(torch.stack([sg_c(s), sg_c(g)], 1))
    logits = (F.normalize(q.reshape(B, -1), dim=1) @ F.normalize(et.reshape(B, -1), dim=1).t()) / TEMP
    loss = 0.5 * (F.cross_entropy(logits, lab) + F.cross_entropy(logits.t(), lab))
    opt1.zero_grad(set_to_none=True); loss.backward(); opt1.step()
    if stp >= STEPS1 - 50:
        accs.append((logits.argmax(1) == lab).float().mean().item())
MATCH_ACC = float(np.mean(accs))
model.freeze_front_end()
print(f"  match-acc {MATCH_ACC:.3f}", flush=True)

# ---- stage 2: LaCoT 的三個 loss（真 head、分類 NLL）＋ GCBC 地板 ----
train_mods = [model.cond_enc, model.cond_head, model.flow, model.refine, model.action_head]
opt2 = torch.optim.Adam([p for m in train_mods for p in m.parameters()], lr=5e-4)
gcbc_enc = nn.Sequential(nn.Linear(2, 512), nn.GELU(), nn.LayerNorm(512),
                         nn.Linear(512, 512), nn.GELU(), nn.LayerNorm(512)).to(device)
gcbc_head = nn.Sequential(nn.Linear(1024, 512), nn.GELU(), nn.LayerNorm(512),
                          nn.Linear(512, CHUNK * ADIM)).to(device)
opt_g = torch.optim.Adam(list(gcbc_enc.parameters()) + list(gcbc_head.parameters()), lr=5e-4)
ZERO = torch.zeros(B, K, D_MODEL, device=device)
print(f"stage 2: LaCoT (head={model.head_kind}) ＋ GCBC ...", flush=True)
for stp in range(STEPS2):
    traj, mask, s, g, act = make_batch(rng)
    with torch.no_grad():
        et = model.e_target(traj, mask)
    cond = model.encode_cond(s, g)
    total, parts = model.losses_given(cond, et, act, rounds=3, lam_cons=0.5)
    # null-u 地板：頭在沒有 u 的時候能做多好（跟舊版一致，用真 head 的 nll）
    l_null = model.action_head.nll(model.action_head(ZERO.reshape(B, -1)), act).mean()
    (total + l_null).backward()
    torch.nn.utils.clip_grad_norm_([p for m in train_mods for p in m.parameters()], 1.0)
    opt2.step(); opt2.zero_grad(set_to_none=True)
    pred = gcbc_head(torch.cat([gcbc_enc(s), gcbc_enc(g)], 1)).reshape(-1, CHUNK, ADIM)
    l_gcbc = (pred - act).pow(2).mean()
    opt_g.zero_grad(set_to_none=True); l_gcbc.backward(); opt_g.step()
    if (stp + 1) % 1000 == 0:
        print(f"  step {stp+1}  nf {parts['l_nf']:.4f} anchor {parts['l_act_anchor']:.4f} "
              f"refine {parts['l_act_refine']:.4f} cons {parts['l_cons']:.4f} null {l_null.item():.4f} "
              f"gcbc {l_gcbc.item():.4f}", flush=True)
model.eval()

# ---- 訓練健康度：head 有沒有真的學到東西？ ----
# ⛔ 基準必須跟 head 的 loss 【同單位】，否則這個檢查是壞的：
#    連續 head 的 nll 是 MSE     -> 基準 = 用資料集平均去猜的 MSE
#    離散 head 的 nll 是 nats    -> 基準 = action 的邊際熵
#    （第一版拿邊際熵 4.77 去比 MSE 0.33 就說「✅ 比基準好」—— 那個比較沒有意義。）
if model.head_kind == "continuous":
    MARGINAL_H = float(((ACT - ACT.mean(0)) ** 2).mean())
    _unit = "MSE"
else:
    _idx = np.clip(np.floor((ACT + 1.0) / (2.0 / 256)), 0, 255).astype(int)
    _p = np.bincount(_idx.reshape(-1), minlength=256).astype(np.float64); _p /= _p.sum()
    MARGINAL_H = float(-(_p[_p > 0] * np.log(_p[_p > 0])).sum())
    _unit = "nats"
_anchor, _null = parts["l_act_anchor"], l_null.item()
print(f"\n=== 訓練健康度 ===", flush=True)
print(f"  基準（什麼都不學，單位={_unit}）      {MARGINAL_H:.4f}", flush=True)
print(f"  l_act_anchor（吃真 e_target）      {_anchor:.4f}  "
      f"{'✅ 比基準好' if _anchor < MARGINAL_H else '🚨 比【什麼都不學】還差 — rollout 會無意義'}", flush=True)
print(f"  l_null（沒有 u）                   {_null:.4f}", flush=True)
print(f"  anchor 相對 null 的增益            {_null - _anchor:+.4f}  <- e_target 到底帶來多少資訊", flush=True)
if _anchor >= MARGINAL_H:
    print("  ⛔ head 沒學起來，下面的 success rate 不用讀。", flush=True)

# ---- eval：官方標準 ----
os.environ.setdefault("OGBENCH_DATA_DIR", OGB_DATA)
env, _, _ = ogbench.make_env_and_datasets(ENV_NAME)
MAXH = int(os.environ.get("LACOT_EVAL_MAXH", env.spec.max_episode_steps or 1000))
N_TASKS = len(env.unwrapped.task_infos)
EPISODES = int(os.environ.get("LACOT_EVAL_EPISODES", 20))
GAIN = 5.0
print(f"\n==== eval: 官方標準  MAXH={MAXH}  {N_TASKS} tasks x {EPISODES} eps = "
      f"{N_TASKS*EPISODES} 集 ====", flush=True)


def expert_positions(obs, goal, horizon=300):
    xy = np.asarray(obs, np.float64); g = np.asarray(goal, np.float64)
    poss = [xy.copy()]
    for _ in range(horizon):
        subgoal, bfs = env.unwrapped.get_oracle_subgoal(xy, g)
        here = env.unwrapped.xy_to_ij(xy)
        target = g if bfs[here[0], here[1]] == 0 else np.asarray(subgoal)
        a = np.clip(GAIN * (target - xy), -1, 1)
        xy = xy + 0.2 * a
        poss.append(xy.copy())
        if np.linalg.norm(xy - g) < 0.5:
            break
    return np.array(poss)


def normstate(x):
    return ((torch.tensor(np.asarray(x, np.float32), device=device) - MU) / SD)[None]


@torch.no_grad()
def oracle_u(obs, goal):
    poss = expert_positions(obs, goal)
    idx = np.unique(np.linspace(0, len(poss) - 1, min(T_CAP, len(poss))).round().astype(int))
    traj = torch.tensor(((poss[idx] - mu) / sd).astype(np.float32), device=device)[None]
    return model.e_target(traj)


@torch.no_grad()
def gcbc_act(s, g):
    return gcbc_head(torch.cat([gcbc_enc(s), gcbc_enc(g)], 1)).reshape(CHUNK, ADIM).clamp(-1, 1)


def rollout(kind, R, tag, open_loop=False):
    succ = ep = 0
    for task in range(1, N_TASKS + 1):
        for sd_ in range(EPISODES):
            obs, info = env.reset(seed=1000 * task + sd_, options={"task_id": task, "render_goal": False})
            goal = info["goal"]; success = False; steps = 0
            torch.manual_seed(7 * task + sd_)
            frozen_u = None                      # open-loop：只算一次
            while steps < MAXH and not success:
                s = normstate(obs); g = normstate(goal)
                cond = model.encode_cond(s, g)   # cond 每個 chunk 都更新（知道自己在哪）
                if kind == "gcbc":
                    a_chunk = gcbc_act(s, g).cpu().numpy()
                elif kind == "null":
                    a_chunk = model.act(cond, u=torch.zeros(1, K, D_MODEL, device=device))[0].cpu().numpy()
                else:
                    if kind == "oracle":
                        if open_loop:
                            if frozen_u is None:
                                frozen_u = oracle_u(obs, goal)
                            u = frozen_u
                        else:
                            u = oracle_u(obs, goal)
                    else:                        # lacot：flow 採樣
                        if open_loop:
                            if frozen_u is None:
                                frozen_u = model.sample_u(cond)
                            u = frozen_u
                        else:
                            u = model.sample_u(cond)
                    a_chunk = model.act(cond, rounds=R, u=u)[0].cpu().numpy()
                for a in np.clip(a_chunk, -1, 1).astype(np.float32):
                    obs, rew, term, trunc, info = env.step(a)
                    steps += 1
                    if info.get("success"):
                        success = True
                    if success or term or trunc or steps >= MAXH:
                        break
            succ += int(success); ep += 1
    print(f"  {tag:<42} {succ:>3}/{ep} = {succ/ep:.3f}", flush=True)
    return succ / ep


res = {}
res["oracle_closed"] = rollout("oracle", 0, "ORACLE closed-loop (每 chunk 重算 BFS)")
res["oracle_open"] = rollout("oracle", 0, "ORACLE open-loop  (只在起點想一次)", open_loop=True)
res["lacot_r0_closed"] = rollout("lacot", 0, "LaCoT R=0 closed (flow 每 chunk 重採)")
res["lacot_r3_closed"] = rollout("lacot", 3, "LaCoT R=3 closed (＋refine 3 輪)")
res["lacot_r3_open"] = rollout("lacot", 3, "LaCoT R=3 open   (想一次、照著走)", open_loop=True)
res["gcbc"] = rollout("gcbc", 0, "GCBC floor (無 u)")
res["null"] = rollout("null", 0, "null-u floor (head(0))")

print("\n" + "=" * 66)
print(f"K={K} (dim {K*D_MODEL})  |  e_target match-acc {MATCH_ACC:.3f}  |  真 DiscretizedActionHead")
print("-" * 66)
gap = res["oracle_closed"] - res["oracle_open"]
print(f"ORACLE closed {res['oracle_closed']:.3f}  vs  open {res['oracle_open']:.3f}   差距 {gap:+.3f}")
print(f"  -> 差距小 = 想出來的那條路禁得起放著不管；差距大 = 成績主要來自每 chunk 重規劃")
print(f"LaCoT  R=3 closed {res['lacot_r3_closed']:.3f}  vs  open {res['lacot_r3_open']:.3f}")
print(f"天花板差距: ORACLE closed {res['oracle_closed']:.3f} - LaCoT R=3 closed {res['lacot_r3_closed']:.3f} "
      f"= {res['oracle_closed']-res['lacot_r3_closed']:+.3f}  <- 這才是 u 還欠的部分")
print("=" * 66)
