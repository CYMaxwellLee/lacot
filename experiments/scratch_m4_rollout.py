"""M4 SUCCESS-RATE eval (the real OGBench metric, not BC-MSE).
Train M4 (state: contrastive e_target frozen -> flow -> refine -> action MLP),
then ROLL OUT in the pointmaze env and measure success rate, comparing:
  * (s,g)-only floor  = ahead(cond, ZERO-u)      [GCBC / depth-0 floor]
  * M4 refine R = 0/1/3/5/8                        [test-time scaling]
Success = the env's own info['success']. Receding-horizon CHUNK execution.
"""
import os, sys, numpy as np, torch
from torch import nn
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # m4 repo root
from lacot.e_target import PerceiverPooler
from lacot.nf_head import Flow
from lacot.model import RefineOperator
import ogbench

# 資料位置：預設走官方 OGBENCH_DATA_DIR，沒設才用本機 archive
OGB_DATA = os.environ.get("OGBENCH_DATA_DIR", "/archive/cymaxwelllee/data/ogbench")

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device, flush=True)
# ⚠️ 環境改成變數（主人 2026-08-23：「跑個 large，跑個 stitch」）。
# ★ 為什麼要換環境：medium-navigate 上誠實 BC 地板已經 0.900、天花板 1.000
#   ⇒ 只剩 0.1 的空間，而 seed 噪聲就有 0.1 ⇒ ⛔ 這個任務量不出 u 的價值，
#   不管 u 好不好。stitch 是刻意設計成「資料裡沒有完整路徑、必須把片段接起來」的，
#   BC 在那上面本來就會爛 —— 而接片段正是 u 該做的事。
ENV_NAME = os.environ.get("LACOT_ENV", "pointmaze-medium-navigate-v0")
d = np.load(f"{OGB_DATA}/{ENV_NAME}.npz")
OBS = np.asarray(d["observations"], np.float32); ACT = np.asarray(d["actions"], np.float32); TERM = np.asarray(d["terminals"], bool)
N = OBS.shape[0]; ends = np.flatnonzero(TERM); starts = np.concatenate([[0], ends[:-1] + 1])
traj_end = np.empty(N, np.int64)
for s0, e0 in zip(starts, ends):
    traj_end[s0:e0 + 1] = e0
mu, sd = OBS.mean(0), OBS.std(0) + 1e-6
MU = torch.tensor(mu, device=device); SD = torch.tensor(sd, device=device)
# 🚨 K 與 COND 改成環境變數，⛔ 不要再靠手改檔案跑不同設定 ——
#    2026-08-23 就踩到：檔案裡寫 K=64，但 FINDINGS 記的三 seed 成功率標的是 K=4，
#    對不起來，等於那組 0.85 沒辦法照原樣重跑（改了沒記＝下次的自己找不回來）。
#    預設用【文件記錄的那組】K=4；⚠️ K=4 是 u_dim sweep 量到 flow 對齊最好的（0.934 vs 0.523）。
K = int(os.environ.get("LACOT_K", 4))
COND = int(os.environ.get("LACOT_COND", 256))
# CHUNK 也做成環境變數：官方 GCBC 是【每步】重新決策（等於 CHUNK=1），
# LaCoT 是一次輸出 4 步 —— 而 pointmaze 的 observation 只有 (x,y)、沒有速度，
# 球又有慣性 ⇒ 對 Markovian policy 是 POMDP，分塊等於做了時間平滑。
# ⇒ 這可能就是「官方 0.15 vs 我們 0.85」的原因。主人 2026-08-23 核可查。
CHUNK = int(os.environ.get("LACOT_CHUNK", 4))
B, T_CAP, D_MODEL, GEOM_P, TEMP, ADIM = 64, 16, 256, 0.02, 0.1, 2
DIM = K * D_MODEL

def make_batch(rng):
    rows, goals = [], []
    while len(rows) < B:
        r = int(rng.integers(0, N)); te = int(traj_end[r])
        if te - r < CHUNK:
            continue
        gr = min(r + int(rng.geometric(GEOM_P)), te)
        if gr - r < CHUNK:
            continue
        rows.append(r); goals.append(gr)
    rows, goals = np.array(rows), np.array(goals)
    idxs = [np.unique(np.linspace(rows[i], goals[i], min(T_CAP, goals[i] - rows[i] + 1)).round().astype(int)) for i in range(B)]
    Tmax = max(len(ix) for ix in idxs)
    traj = np.zeros((B, Tmax, 2), np.float32); mask = np.ones((B, Tmax), bool)
    for i, ix in enumerate(idxs):
        traj[i, :len(ix)] = (OBS[ix] - mu) / sd; mask[i, :len(ix)] = False
    s = (OBS[rows] - mu) / sd; g = (OBS[goals] - mu) / sd
    act = np.stack([ACT[r:r + CHUNK] for r in rows]).astype(np.float32)
    T = lambda x: torch.from_numpy(x.astype(np.float32)).to(device)
    return T(traj), torch.from_numpy(mask).to(device), T(s), T(g), T(act)

def sota_mlp(i, h, o, n=2):
    L, p = [], i
    for _ in range(n):
        lin = nn.Linear(p, h); nn.init.xavier_uniform_(lin.weight); nn.init.zeros_(lin.bias)
        L += [lin, nn.GELU(), nn.LayerNorm(h)]; p = h
    lin = nn.Linear(p, o); nn.init.xavier_uniform_(lin.weight); nn.init.zeros_(lin.bias)
    return nn.Sequential(*L, lin)

class ActionMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = sota_mlp(COND + DIM, 512, CHUNK * ADIM, n=3)
    def forward(self, cond, u):
        x = torch.cat([cond, u.reshape(u.shape[0], -1)], -1)
        return self.net(x).reshape(-1, CHUNK, ADIM)

SEED = int(os.environ.get("LACOT_SEED", 0))
# refine 的 consistency 目標：self = 跟下一刻的自己比（原版）；ema = 跟 EMA 副本比。
# `[實測 2026-08-23]` anti-collapse 掃了 11 個變體，只有 ema 系列三項全過
#   （塌度 0.056 / 路徑資訊贏內插 / cos 真et +0.833），byol 那組塌度 0.48~0.69 全塌。
# ⚠️ 但那次掃描 **每格只有 1 個 seed**，⇒ ema 內部 m99/m996/m999 誰最好還不算數，
#    只有「ema 系列 vs byol 系列」那個差距大到不可能是 seed 噪聲。
CONS = os.environ.get("LACOT_CONS", "self")
EMA_M = float(os.environ.get("LACOT_EMA_M", 0.996))
torch.manual_seed(SEED); rng = np.random.default_rng(SEED)
print(f"設定：seed={SEED} cons={CONS} ema_m={EMA_M} K={K} COND={COND}", flush=True)
traj_enc = sota_mlp(2, 512, 512).to(device); e_pooler = PerceiverPooler(512, D_MODEL, K, 2, 4).to(device)
sg_c = sota_mlp(2, 512, 512).to(device); q_pooler = PerceiverPooler(512, D_MODEL, K, 2, 4).to(device)
opt1 = torch.optim.Adam([p for m in (traj_enc, e_pooler, sg_c, q_pooler) for p in m.parameters()], lr=1e-3)
lab = torch.arange(B, device=device)
def etarget(traj, mask):
    Bc, Tc, _ = traj.shape
    return e_pooler(traj_enc(traj.reshape(Bc * Tc, 2)).reshape(Bc, Tc, 512), key_padding_mask=mask)
print("stage 1 contrastive e_target ...", flush=True)
for stp in range(int(os.environ.get("LACOT_STEPS1", 1500))):
    traj, mask, s, g, _ = make_batch(rng)
    et = etarget(traj, mask); q = q_pooler(torch.stack([sg_c(s), sg_c(g)], 1))
    logits = (F.normalize(q.reshape(B, -1), dim=1) @ F.normalize(et.reshape(B, -1), dim=1).t()) / TEMP
    loss = 0.5 * (F.cross_entropy(logits, lab) + F.cross_entropy(logits.t(), lab))
    opt1.zero_grad(set_to_none=True); loss.backward(); opt1.step()
for m in (traj_enc, e_pooler):
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
print(f"  e_target match-acc {(logits.argmax(1)==lab).float().mean().item():.3f}", flush=True)

cond_enc = sota_mlp(2, 512, 512).to(device); cond_head = sota_mlp(1024, 512, COND).to(device)
flow = Flow(token_dim=D_MODEL, seq_len=K, n_blocks=4, cond_dim=COND).to(device)
refine = RefineOperator(COND, K, D_MODEL, hidden=256).to(device)
import copy
refine_ema = copy.deepcopy(refine)
for _p in refine_ema.parameters():
    _p.requires_grad_(False)
ahead = ActionMLP().to(device)

# 🚨 誠實的 BC 地板（主人 2026-08-23 要求）。
# ⛔ 舊做法是 eval 時把 u 塞 0 當地板 —— 但訓練損失裡【沒有】u=0 的分支，
#    head 從沒看過零 ⇒ 那是分布外探針，不是 baseline，而且訓練愈久掉愈兇。
#    同一個壞探針先後撐起了「u 沒貢獻」與「u 貢獻 0.61」兩個相反的結論。
# ⇒ 這裡另外養一顆【從頭到尾只吃 cond、永遠不給 u】的 head，跟 ahead 同容量、同優化器。
class CondOnlyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = sota_mlp(COND, 512, CHUNK * ADIM, n=3)
    def forward(self, cond):
        return self.net(cond).reshape(-1, CHUNK, ADIM)

bc_head = CondOnlyMLP().to(device)
f_mods = [cond_enc, cond_head, flow, refine, ahead, bc_head]
opt2 = torch.optim.Adam([p for m in f_mods for p in m.parameters()], lr=5e-4)
def condvec(s, g):
    return cond_head(torch.cat([cond_enc(s), cond_enc(g)], 1))
mse = lambda p, a: (p - a).pow(2).mean()
print("stage 2 flow+refine+action ...", flush=True)
STEPS2 = int(os.environ.get("LACOT_STEPS2", 2000))
for stp in range(STEPS2):
    traj, mask, s, g, act = make_batch(rng)
    with torch.no_grad():
        et = etarget(traj, mask)
    cond = condvec(s, g)
    l_nf = flow.nll(et, cond) / DIM
    l_anchor = mse(ahead(cond, et), act)
    u = flow.sample(B, cond).detach(); us = [u]
    for _ in range(3):
        u = refine(cond, u); us.append(u)
    if CONS == "ema":
        with torch.no_grad():
            tgts = [refine_ema(cond, us[r]) for r in range(3)]
        l_cons = sum((us[r + 1] - tgts[r]).pow(2).mean() for r in range(3)) / 3
    else:
        l_cons = sum((us[r] - us[r + 1].detach()).pow(2).mean() for r in range(3)) / 3
    l_refine = sum(mse(ahead(cond, us[r + 1]), act) for r in range(3)) / 3
    # ⭐ 誠實地板：只吃 cond，跟 u 完全無關。
    # ⚠️ cond 要 detach —— 否則 l_bc 的梯度會流進 cond_enc/cond_head，
    #    把 cond 訓練得更會單獨預測動作 ⇒ ① 主模型被這個 baseline 改動了、
    #    ② 比較會系統性地偏向「u 沒必要」。detach 之後量的才是乾淨的問題：
    #    「在【同一個】cond 表徵上，u 有沒有加值」。
    l_bc = mse(bc_head(cond.detach()), act)
    total = l_nf + l_anchor + l_refine + 0.5 * l_cons + l_bc
    opt2.zero_grad(set_to_none=True); total.backward()
    torch.nn.utils.clip_grad_norm_([p for m in f_mods for p in m.parameters()], 1.0); opt2.step()
    if CONS == "ema":
        with torch.no_grad():
            for pe, pr in zip(refine_ema.parameters(), refine.parameters()):
                pe.mul_(EMA_M).add_(pr, alpha=1 - EMA_M)
    if (stp + 1) % 1000 == 0:
        print(f"  step {stp+1}  l_nf/dim {l_nf.item():.3f} l_anchor {l_anchor.item():.4f} l_refine {l_refine.item():.4f}", flush=True)
for m in f_mods:
    m.eval()

# -------- SUCCESS-RATE ROLLOUT --------
def normstate(x):  # raw env position -> normalized torch [1,2]
    return ((torch.tensor(np.asarray(x, np.float32), device=device) - MU) / SD)[None]

# ⭐ 「別人的 u」探針（主人 2026-08-23 核可）。
# ⛔ 零向量那個探針會製造 OOD，量到的是「head 沒看過零」的懲罰，不是 u 的價值。
# ★ 這個換法只換【內容】不換【分布】：從資料集隨機抽另一組 (s,g)，用它的 cond 生成並
#   refine 出 u，再把那個 u 配上【本題】的 cond 餵給 head。
#   成績不掉 ⇒ u 的內容根本沒被讀，head 只需要「那個位置有東西」。
#   成績掉了 ⇒ 內容有被讀，只是沒比 cond 多帶東西。
_shuf_rng = np.random.default_rng(20260823)

@torch.no_grad()
def _foreign_u(R):
    """從資料集隨機抽一組 (s,g)，回傳它 refine R 輪之後的 u。"""
    while True:
        r = int(_shuf_rng.integers(0, N)); te = int(traj_end[r])
        if te - r >= CHUNK:
            break
    gr = min(r + int(_shuf_rng.geometric(GEOM_P)), te)
    s2 = torch.tensor((OBS[r] - mu) / sd, device=device)[None]
    g2 = torch.tensor((OBS[gr] - mu) / sd, device=device)[None]
    c2 = condvec(s2, g2)
    u = flow.sample(1, c2)
    for _ in range(R):
        u = refine(c2, u)
    return u


@torch.no_grad()
def policy_chunk(obs, goal, R, use_u):
    s = normstate(obs); g = normstate(goal); cond = condvec(s, g)
    if use_u == "bc":                              # ⭐ 誠實地板，走另一顆 head
        a = bc_head(cond)[0].cpu().numpy()
        return np.clip(a, -1.0, 1.0).astype(np.float32)
    if use_u == "shuf":                            # ⭐ 別人的 u，本題的 cond
        a = ahead(cond, _foreign_u(R))[0].cpu().numpy()
        return np.clip(a, -1.0, 1.0).astype(np.float32)
    if use_u:
        u = flow.sample(1, cond)
        for _ in range(R):
            u = refine(cond, u)
    else:
        u = torch.zeros(1, K, D_MODEL, device=device)  # (s,g)-only floor
    a = ahead(cond, u)[0].cpu().numpy()  # [CHUNK,2]
    return np.clip(a, -1.0, 1.0).astype(np.float32)

# ⚠️ ogbench 不看 OGBENCH_DATA_DIR，它只看 dataset_dir 參數（預設 ~/.ogbench/data）。
#    2026-08-23 實測：不給 dataset_dir 它會【重新下載到 home】，而 home 是 NFS。
#    ⇒ 一定要明確傳本機 /archive 的路徑。
os.environ.setdefault("OGBENCH_DATA_DIR", OGB_DATA)
env, _, _ = ogbench.make_env_and_datasets(ENV_NAME, dataset_dir=OGB_DATA)
MAXH = int(os.environ.get("LACOT_EVAL_MAXH", env.spec.max_episode_steps or 1000))  # 官方標準，不自訂難度
N_TASKS = len(env.unwrapped.task_infos); SEEDS = int(os.environ.get("LACOT_EVAL_EPISODES", 20))  # 官方 eval_episodes=20  # episodes = N_TASKS * SEEDS per variant

def rollout(R, use_u, tag):
    succ, ep = 0, 0
    for task in range(1, N_TASKS + 1):
        for sd_ in range(SEEDS):
            obs, info = env.reset(seed=1000 * task + sd_, options={"task_id": task, "render_goal": False})
            goal = info["goal"]; success = False; steps = 0
            torch.manual_seed(7 * task + sd_)  # action-sampler stream
            while steps < MAXH and not success:
                for a in policy_chunk(obs, goal, R, use_u):
                    obs, rew, term, trunc, info = env.step(a)
                    steps += 1
                    if info.get("success"):
                        success = True
                    if success or term or trunc or steps >= MAXH:
                        break
            succ += int(success); ep += 1
    print(f"  {tag}: success {succ}/{ep} = {succ/ep:.3f}", flush=True)
    return succ / ep

print(f"\n==== SUCCESS RATE (env={ENV_NAME}, {N_TASKS} tasks x {SEEDS} seeds, MAXH {MAXH}) ====", flush=True)
out = dict(env=ENV_NAME, seed=SEED, cons=CONS, ema_m=EMA_M, K=K, cond=COND, chunk=CHUNK, steps2=STEPS2,
           episodes=N_TASKS * SEEDS, maxh=MAXH, rates={})
# ⚠️ rollout 是整支腳本最貴的部分，而成本跟 CHUNK 成反比：CHUNK=1 每步都要重新決策，
#    比 CHUNK=4 多四倍的 policy 呼叫。⇒ 要跑 chunk 對照時用 LACOT_EVAL_RS 只留需要的輪數，
#    不然單格會超過叢集的時間上限（實測：CHUNK=1 一個變體就要 ~26 分）。
RS = [int(x) for x in os.environ.get("LACOT_EVAL_RS", "0,1,3,5,8").split(",") if x != ""]
out["rates"]["bc"] = rollout(0, "bc", "誠實 BC 地板（獨立 head，只吃 cond）")
out["rates"]["null_u"] = rollout(0, False, "u 歸零（⚠️ OOD 探針，不是地板）")
out["rates"]["shuf"] = rollout(3, "shuf", "別人的 u（分布對、內容錯）")
for R in RS:
    out["rates"][f"R{R}"] = rollout(R, True, f"M4 refine R={R}")
print("=> want: M4 > floor (reasoning helps on success), rate rises with R (test-time scaling).", flush=True)

# ⛔ 印到 stdout 不算存下來 —— log 會被覆蓋、也收不成表。
import json
# ⛔ 檔名要帶【所有會變的設定】—— 少一個就會跟別組互相覆蓋，
#    而覆蓋掉的舊檔在數值上完全合理、看不出來（2026-08-07 已經被這個咬過）。
tag = f"{ENV_NAME.replace('pointmaze-', '').replace('-v0', '')}_{CONS}_K{K}_c{COND}_ch{CHUNK}_st{STEPS2}_s{SEED}"
dst = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results",
                   f"rollout_{tag}.json")
os.makedirs(os.path.dirname(dst), exist_ok=True)
with open(dst, "w") as f:
    json.dump(out, f, indent=1)
print(f"寫入 {dst}", flush=True)

# ⭐ 存 checkpoint：以後要換探針就不必重訓一次（今天為了換探針重訓了三輪）。
ck = os.path.join(os.path.dirname(dst), f"ckpt_{tag}.pt")
torch.save({"cond_enc": cond_enc.state_dict(), "cond_head": cond_head.state_dict(),
            "flow": flow.state_dict(), "refine": refine.state_dict(),
            "ahead": ahead.state_dict(), "bc_head": bc_head.state_dict(),
            "traj_enc": traj_enc.state_dict(), "e_pooler": e_pooler.state_dict(),
            "cfg": dict(K=K, COND=COND, CHUNK=CHUNK, D_MODEL=D_MODEL, STEPS2=STEPS2,
                        CONS=CONS, EMA_M=EMA_M, SEED=SEED)}, ck)
print(f"存 checkpoint {ck}", flush=True)
