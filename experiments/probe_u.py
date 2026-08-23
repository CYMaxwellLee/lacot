"""不重訓，直接載入 checkpoint，換各種 u 餵進 head，量成功率。

主人 2026-08-23 問：「u 填 0 不行，那隨便填 random number 呢？」

⭐ 這支的價值在於它**不需要訓練** —— 之前每換一個探針就重訓一次（今天重訓了四輪）。
   checkpoint 是那四輪的副產品，現在換探針只要幾分鐘。

各種 u 的意義（由「離真 u 最遠」排到「就是真 u」）：
  zero    全零           ⛔ 訓練時沒看過 ⇒ 分布外，量到的是懲罰不是資訊
  gauss   標準常態       尺度不對、方向隨機
  matched 配對過的高斯   逐維對上真 u 的平均與標準差，但維度之間互相獨立
  shuffle 打亂 batch     ⭐ 真 u，但配到別人的題目 —— 分布完全正確、只有內容錯
  real    真的 u

判讀：
  real ≈ shuffle          ⇒ head 沒在讀內容
  real ≈ matched ≈ gauss  ⇒ head 連「像不像 u」都不在乎，只要有東西
  gauss 掉、shuffle 不掉  ⇒ head 要的是「落在 u 的流形上」，但不看是哪一點
"""
import os
import sys

import numpy as np
import torch
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lacot.nf_head import Flow
from lacot.model import RefineOperator
import ogbench

CKPT = os.environ["LACOT_CKPT"]
OGB_DATA = os.environ.get("OGBENCH_DATA_DIR", "/archive/cymaxwelllee/data/ogbench")
device = "cuda" if torch.cuda.is_available() else "cpu"
sd_ck = torch.load(CKPT, map_location=device, weights_only=False)
cfg = sd_ck["cfg"]
K, COND, CHUNK, D_MODEL = cfg["K"], cfg["COND"], cfg["CHUNK"], cfg["D_MODEL"]
ADIM = 2
ENV_NAME = os.environ.get("LACOT_ENV", "pointmaze-medium-navigate-v0")
R = int(os.environ.get("LACOT_R", 3))
EPISODES = int(os.environ.get("LACOT_EVAL_EPISODES", 20))
print(f"載入 {os.path.basename(CKPT)}  env={ENV_NAME}  cfg={cfg}", flush=True)

d = np.load(f"{OGB_DATA}/{ENV_NAME}.npz")
OBS = np.asarray(d["observations"], np.float32)
mu, sd = OBS.mean(0), OBS.std(0) + 1e-6
MU = torch.tensor(mu, device=device); SD = torch.tensor(sd, device=device)


def sota_mlp(i, h, o, n=2):
    L, p = [], i
    for _ in range(n):
        L += [nn.Linear(p, h), nn.GELU(), nn.LayerNorm(h)]; p = h
    return nn.Sequential(*L, nn.Linear(p, o))


class ActionMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = sota_mlp(COND + K * D_MODEL, 512, CHUNK * ADIM, n=3)
    def forward(self, cond, u):
        return self.net(torch.cat([cond, u.reshape(u.shape[0], -1)], -1)).reshape(-1, CHUNK, ADIM)


class CondOnlyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = sota_mlp(COND, 512, CHUNK * ADIM, n=3)
    def forward(self, cond):
        return self.net(cond).reshape(-1, CHUNK, ADIM)


cond_enc = sota_mlp(2, 512, 512).to(device); cond_head = sota_mlp(1024, 512, COND).to(device)
flow = Flow(token_dim=D_MODEL, seq_len=K, n_blocks=4, cond_dim=COND).to(device)
refine = RefineOperator(COND, K, D_MODEL, hidden=256).to(device)
ahead = ActionMLP().to(device); bc_head = CondOnlyMLP().to(device)
for m, key in [(cond_enc, "cond_enc"), (cond_head, "cond_head"), (flow, "flow"),
               (refine, "refine"), (ahead, "ahead"), (bc_head, "bc_head")]:
    m.load_state_dict(sd_ck[key]); m.eval()
condvec = lambda s, g: cond_head(torch.cat([cond_enc(s), cond_enc(g)], 1))

# 先抽一批真 u，量它的逐維統計（給 matched 用），順便當 shuffle 的池子
with torch.no_grad():
    idx = np.random.default_rng(7).integers(0, OBS.shape[0], 512)
    idx2 = np.random.default_rng(8).integers(0, OBS.shape[0], 512)
    s_p = torch.tensor((OBS[idx] - mu) / sd, device=device)
    g_p = torch.tensor((OBS[idx2] - mu) / sd, device=device)
    c_p = condvec(s_p, g_p)
    u_p = flow.sample(512, c_p)
    for _ in range(R):
        u_p = refine(c_p, u_p)
    U_MEAN, U_STD = u_p.mean(0, keepdim=True), u_p.std(0, keepdim=True)
    print(f"真 u 統計：平均絕對值 {u_p.abs().mean():.4f}  逐維標準差平均 {U_STD.mean():.4f}", flush=True)

POOL = u_p
_rng = np.random.default_rng(20260823)


@torch.no_grad()
def make_u(kind, cond):
    if kind == "real":
        u = flow.sample(1, cond)
        for _ in range(R):
            u = refine(cond, u)
        return u
    if kind == "zero":
        return torch.zeros(1, K, D_MODEL, device=device)
    if kind == "gauss":
        return torch.randn(1, K, D_MODEL, device=device)
    if kind == "matched":
        return U_MEAN + U_STD * torch.randn(1, K, D_MODEL, device=device)
    if kind == "shuffle":
        return POOL[int(_rng.integers(0, POOL.shape[0]))][None]
    raise ValueError(kind)


normstate = lambda x: ((torch.tensor(np.asarray(x, np.float32), device=device) - MU) / SD)[None]

env, _, _ = ogbench.make_env_and_datasets(ENV_NAME, dataset_dir=OGB_DATA)
MAXH = env.spec.max_episode_steps or 1000
N_TASKS = len(env.unwrapped.task_infos)


@torch.no_grad()
def rollout(kind):
    succ = ep = 0
    for task in range(1, N_TASKS + 1):
        for s_ in range(EPISODES):
            obs, info = env.reset(seed=1000 * task + s_, options={"task_id": task, "render_goal": False})
            goal = info["goal"]; success = False; steps = 0
            torch.manual_seed(7 * task + s_)
            while steps < MAXH and not success:
                s = normstate(obs); g = normstate(goal); cond = condvec(s, g)
                a = bc_head(cond) if kind == "bc" else ahead(cond, make_u(kind, cond))
                for act in np.clip(a[0].cpu().numpy(), -1.0, 1.0).astype(np.float32):
                    obs, _, term, trunc, info = env.step(act)
                    steps += 1
                    if info.get("success"):
                        success = True
                    if success or term or trunc or steps >= MAXH:
                        break
            succ += int(success); ep += 1
    print(f"  {kind:<9} {succ}/{ep} = {succ/ep:.3f}", flush=True)
    return succ / ep


print(f"\n==== {ENV_NAME}  R={R}  {N_TASKS}x{EPISODES} 集 ====", flush=True)
out = {k: rollout(k) for k in ["bc", "real", "shuffle", "matched", "gauss", "zero"]}
import json
dst = CKPT.replace("ckpt_", "probe_").replace(".pt", ".json")
with open(dst, "w") as f:
    json.dump(dict(ckpt=os.path.basename(CKPT), env=ENV_NAME, R=R, rates=out), f, indent=1)
print(f"寫入 {dst}", flush=True)
