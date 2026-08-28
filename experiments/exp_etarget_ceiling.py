"""「絕對對的 e_target 能出多好的動作」— 主人 2026-08-24 裁示：「重練一個好了，比較乾脆
   直接拿絕對對的 e target 來給產生動作」「做，但也量 success rate，妳昨晚量過」。

⭐ 為什麼重練而不是拿舊 checkpoint 餵 oracle：舊 head 訓練時只看過【資料集軌跡編的
   e_target】和【refine 出來的 u】，餵 oracle 路徑是第三種分布 ⇒ 昨天量到 0.120 而
   LaCoT R3 有 0.700，那個 0.120 是 OOD 懲罰、不是天花板（docs/ORACLE-REDESIGN.md）。
   重練一顆從頭只讀 e_target 的 head，就繞開了整個爭議。

三顆 head，同一批資料、同一個 optimizer，配對比較：
   A  只吃 e_target            ← latent 單獨的天花板（＝「head 不吃 cond」那個改動的上界）
   B  吃 cond + e_target       ← 現在架構的天花板
   C  只吃 cond                ← 誠實 BC 地板（跟 2026-08-23 的 bc_head 同一顆設計）
⚠️ A 的輸入是 1024 維、C 是 256 維 ⇒ 參數量不同。這是【天花板】實驗，容量多給 A 是
   讓它有最好的機會，⛔ 不能拿來當「A 比 C 好是因為架構好」的證據。

三段量測：
   eval-1  離線動作誤差，e_target 來自【資料集真實未來軌跡】  ← 幾秒，無爭議
   eval-2  離線動作誤差，e_target 來自【oracle BFS 路徑】     ← 🚨 這一格第一次量出
           「oracle 到底有多 OOD」。昨天那條 OOD 診斷是 [推論]，從沒直接驗過。
   eval-3  rollout success rate，A/B/C，oracle e_target      ← 主人要的可比數字

⚠️ eval-3 用 closed-loop（每個 chunk 重算 oracle）。⛔ 不能用 open-loop：A 沒有 cond，
   u 一凍住它就完全不知道自己在哪，每步輸出同一個動作 ⇒ 必然失敗，量到的是設計缺陷
   不是天花板。closed-loop 比較貴，所以 LACOT_EVAL_EPISODES 要留意（smoke 用小的）。
"""
import json
import os
import sys
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lacot.e_target import PerceiverPooler
import ogbench

OGB_DATA = os.environ.get("OGBENCH_DATA_DIR", "/archive/cymaxwelllee/data/ogbench")
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device, flush=True)

ENV_NAME = os.environ.get("LACOT_ENV", "pointmaze-medium-navigate-v0")
K = int(os.environ.get("LACOT_K", 4))
COND = int(os.environ.get("LACOT_COND", 256))
CHUNK = int(os.environ.get("LACOT_CHUNK", 4))
STEPS1 = int(os.environ.get("LACOT_STEPS1", 1500))
STEPS2 = int(os.environ.get("LACOT_STEPS2", 12000))
SEED = int(os.environ.get("LACOT_SEED", 0))
# 🚨 官方值，⛔ 不要再改回來。一手來源：OGBench repo `impls/hyperparameters.sh`
#    pointmaze 的【每一行】（navigate 與 stitch、六個 agent 全部）都是 `--eval_episodes=50`。
#    ⚠️ 20 是 `impls/main.py` 的 flag 預設值 —— 2026-08-24 之前這支腳本把它誤當成官方值。
EPISODES = int(os.environ.get("LACOT_EVAL_EPISODES", 50))
# ⭐ T_CAP = Perceiver 一次讀幾個軌跡點（⛔ 不是截斷，是整條路等距取樣成這麼多點）。
#    `[實測 2026-08-24]` eval 的 oracle 路徑：medium-navigate 平均 126 步、large-stitch 平均 255 步。
#    ⇒ 16 會讓 large-stitch 變成【每 15.9 步一個點】，而訓練時是每 2.4 步一個 ⇒ 密度差 6.6 倍，
#      encoder 從沒看過那麼稀的路 ⇒ 那才是 OOD 的真正來源（⛔ 不是「16 太小」這麼簡單）。
#    ⚠️ Perceiver 的 cross-attention 成本是 K×T（K=4 固定）⇒ 線性，放大 16 倍吃得下。
T_CAP_REQ = int(os.environ.get("LACOT_TCAP", 256))
# 🚨 F1（2026-08-24 subagent 稽核抓到，⭐ 這是ルナ自己改出來的坑）：
#    T_CAP 若【超過訓練時到得了的長度】，多出來那段 positional embedding 從沒收過梯度。
#    large-stitch 每條軌跡只有 201 步 ⇒ 訓練 T ≤ 201，而 eval 的 oracle 路徑 240~256 步
#    ⇒ pos_emb[201:256] 還是隨機初始化值，而那正是【最靠近目標】、資訊最重要的一段。
#    ⛔ T_CAP=16 反而沒這病（訓練 16 / eval 16 匹配）⇒ 放大 T_CAP 必須連同這個夾子一起。
#    ⇒ 實際 T_CAP 在資料載入後才定得下來，見下方 MAX_TRAIN_T。
B, D_MODEL, TEMP, ADIM = 64, 256, 0.1, 2
DIM = K * D_MODEL
GAIN = 5.0
# ⚠️ `[實測 2026-08-24]` horizon=300 時 large-stitch 有 20% 的 oracle 路徑撞到上限
#    ⇒ 那 20% 餵給 head 的「正確答案」是【半條路】。放大到 800，腳本會自報截斷率。
ORACLE_HORIZON = int(os.environ.get("LACOT_ORACLE_HORIZON", 800))

d = np.load(f"{OGB_DATA}/{ENV_NAME}.npz")
OBS = np.asarray(d["observations"], np.float32)
ACT = np.asarray(d["actions"], np.float32)
TERM = np.asarray(d["terminals"], bool)
N = OBS.shape[0]
ends = np.flatnonzero(TERM); starts = np.concatenate([[0], ends[:-1] + 1])
traj_end = np.empty(N, np.int64)
for s0, e0 in zip(starts, ends):
    traj_end[s0:e0 + 1] = e0
assert ends[-1] == N - 1, "資料集最後一筆不是 terminal ⇒ traj_end 尾巴是未初始化的記憶體"
MAX_TRAIN_T = int((ends - starts + 1).max())   # 訓練時 T 到得了的上限＝最長的一條軌跡
T_CAP = min(T_CAP_REQ, MAX_TRAIN_T)
if T_CAP != T_CAP_REQ:
    print(f"⚠️ T_CAP {T_CAP_REQ} → {T_CAP}（夾到本資料集最長軌跡 {MAX_TRAIN_T}，"
          f"⛔ 否則 eval 會用到沒訓練過的 pos_emb）", flush=True)
mu, sd = OBS.mean(0), OBS.std(0) + 1e-6
MU = torch.tensor(mu, device=device); SD = torch.tensor(sd, device=device)

torch.manual_seed(SEED); rng = np.random.default_rng(SEED)
print(f"設定：env={ENV_NAME} seed={SEED} K={K} COND={COND} chunk={CHUNK} "
      f"steps1={STEPS1} steps2={STEPS2} episodes={EPISODES}", flush=True)


def make_batch(r):
    rows, goals = [], []
    while len(rows) < B:
        i = int(r.integers(0, N)); te = int(traj_end[i])
        if te - i < CHUNK:
            continue
        # 🚨 官方抽法，⛔ 不要改回 geometric。一手來源：OGBench `impls/utils/datasets.py`
        #    的 GCDataset.sample_goals()，geom_sample=False 分支：
        #      distances = rand();  goal = round(min(i+1, final)*d + final*(1-d))
        #    ＝在【現在的下一步 → 這條軌跡的結尾】之間【均勻】抽。
        #    而 gcbc.py 的 get_config() 是 actor_p_trajgoal=1.0 / actor_geom_sample=False,
        #    且 hyperparameters.sh 裡 pointmaze 的 GCBC 沒有任何 override ⇒ 這就是官方值。
        #    `[實測 2026-08-24]` 換掉之前我們的 geometric(0.02) 平均只走 47.6 步，
        #    而 medium-navigate 的 eval 要走 126 步 ⇒ ⛔ 訓練根本沒涵蓋 eval 的難度。
        dist = r.random()
        gi = int(round(min(i + 1, te) * dist + te * (1 - dist)))
        # 🚨 F6：這裡原本是 `if gi - i < CHUNK: continue`，而 continue 會【連起點一起重抽】
        #    ⇒ 越靠近軌跡結尾的起點越容易被丟掉（實測 P(起點在結尾 5 步內) 被壓低 30 倍）。
        #    官方不重抽任何東西 ⇒ 改成 clamp：只把 goal 推到至少 CHUNK 步遠，起點分布不動。
        gi = max(gi, min(i + CHUNK, te))
        rows.append(i); goals.append(gi)
    rows, goals = np.array(rows), np.array(goals)
    idxs = [np.unique(np.linspace(rows[i], goals[i], min(T_CAP, goals[i] - rows[i] + 1)).round().astype(int))
            for i in range(B)]
    Tmax = max(len(ix) for ix in idxs)
    traj = np.zeros((B, Tmax, 2), np.float32); mask = np.ones((B, Tmax), bool)
    for i, ix in enumerate(idxs):
        traj[i, :len(ix)] = (OBS[ix] - mu) / sd; mask[i, :len(ix)] = False
    s = (OBS[rows] - mu) / sd; g = (OBS[goals] - mu) / sd
    act = np.stack([ACT[i:i + CHUNK] for i in rows]).astype(np.float32)
    T = lambda x: torch.from_numpy(x.astype(np.float32)).to(device)
    return T(traj), torch.from_numpy(mask).to(device), T(s), T(g), T(act)


def sota_mlp(i, h, o, n=2):
    L, p = [], i
    for _ in range(n):
        lin = nn.Linear(p, h); nn.init.xavier_uniform_(lin.weight); nn.init.zeros_(lin.bias)
        L += [lin, nn.GELU(), nn.LayerNorm(h)]; p = h
    lin = nn.Linear(p, o); nn.init.xavier_uniform_(lin.weight); nn.init.zeros_(lin.bias)
    return nn.Sequential(*L, lin)


# ---------- stage 1: contrastive e_target encoder（訓完凍住，跟 2026-08-23 同一套） ----------
traj_enc = sota_mlp(2, 512, 512).to(device)
e_pooler = PerceiverPooler(512, D_MODEL, K, 2, 4, max_len=max(512, T_CAP)).to(device)
sg_c = sota_mlp(2, 512, 512).to(device)
q_pooler = PerceiverPooler(512, D_MODEL, K, 2, 4, max_len=max(512, T_CAP)).to(device)
opt1 = torch.optim.Adam([p for m in (traj_enc, e_pooler, sg_c, q_pooler) for p in m.parameters()], lr=1e-3)
lab = torch.arange(B, device=device)


def etarget(traj, mask):
    Bc, Tc, _ = traj.shape
    return e_pooler(traj_enc(traj.reshape(Bc * Tc, 2)).reshape(Bc, Tc, 512), key_padding_mask=mask)


print("stage 1 contrastive e_target ...", flush=True)
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
# ⚠️ 這是【最後一個訓練批】的配對準確率，⛔ 不是驗證集指標，只當「stage 1 有沒有學起來」的粗訊號。
match_acc = float("nan") if STEPS1 == 0 else (logits.argmax(1) == lab).float().mean().item()
print(f"  e_target match-acc(train batch) {match_acc:.3f}", flush=True)


# ---------- stage 2: 三顆 head ----------
class HeadA(nn.Module):
    """只吃 e_target。⇒ latent 是唯一通路（Huginn / RD-VLA 的形狀）。"""
    def __init__(self):
        super().__init__()
        self.net = sota_mlp(DIM, 512, CHUNK * ADIM, n=3)

    def forward(self, u):
        return self.net(u.reshape(u.shape[0], -1)).reshape(-1, CHUNK, ADIM)


class HeadB(nn.Module):
    """吃 cond + e_target ＝ 現在 LaCoT 的 ActionMLP。"""
    def __init__(self):
        super().__init__()
        self.net = sota_mlp(COND + DIM, 512, CHUNK * ADIM, n=3)

    def forward(self, cond, u):
        return self.net(torch.cat([cond, u.reshape(u.shape[0], -1)], -1)).reshape(-1, CHUNK, ADIM)


class HeadC(nn.Module):
    """只吃 cond ＝ 誠實 BC 地板。"""
    def __init__(self):
        super().__init__()
        self.net = sota_mlp(COND, 512, CHUNK * ADIM, n=3)

    def forward(self, cond):
        return self.net(cond).reshape(-1, CHUNK, ADIM)


cond_enc = sota_mlp(2, 512, 512).to(device)
cond_head = sota_mlp(1024, 512, COND).to(device)
head_A, head_B, head_C = HeadA().to(device), HeadB().to(device), HeadC().to(device)
mods = [cond_enc, cond_head, head_A, head_B, head_C]
opt2 = torch.optim.Adam([p for m in mods for p in m.parameters()], lr=5e-4)
mse = lambda p, a: (p - a).pow(2).mean()


def condvec(s, g):
    return cond_head(torch.cat([cond_enc(s), cond_enc(g)], 1))


print("stage 2 three heads ...", flush=True)
for stp in range(STEPS2):
    traj, mask, s, g, act = make_batch(rng)
    with torch.no_grad():
        et = etarget(traj, mask)
    cond = condvec(s, g)
    lA = mse(head_A(et), act)
    lB = mse(head_B(cond, et), act)
    # ⚠️ C 吃 detach 過的 cond：⛔ 否則地板的梯度會回頭把 cond 編碼器訓得更會單獨預測動作，
    #    主模型被 baseline 改動，而且比較會系統性偏向「latent 沒必要」（2026-08-23 的教訓）。
    lC = mse(head_C(cond.detach()), act)
    total = lA + lB + lC
    opt2.zero_grad(set_to_none=True); total.backward()
    torch.nn.utils.clip_grad_norm_([p for m in mods for p in m.parameters()], 1.0)
    opt2.step()
    if (stp + 1) % 2000 == 0:
        print(f"  step {stp+1}  A {lA.item():.4f}  B {lB.item():.4f}  C {lC.item():.4f}", flush=True)
for m in mods:
    m.eval()

# ---------- env（eval-2 / eval-3 都要） ----------
os.environ.setdefault("OGBENCH_DATA_DIR", OGB_DATA)
env, _, _ = ogbench.make_env_and_datasets(ENV_NAME, dataset_dir=OGB_DATA)
MAXH = int(os.environ.get("LACOT_EVAL_MAXH", env.spec.max_episode_steps or 1000))
N_TASKS = len(env.unwrapped.task_infos)


_ORACLE_CALLS = _ORACLE_TRUNC = 0
_ORACLE_PHASES = {}


def _oracle_phase(name):
    """把 (呼叫數, 截斷數) 在階段邊界上結算，⇒ 每個階段各自有分母。"""
    global _ORACLE_CALLS, _ORACLE_TRUNC
    if name is not None:
        _ORACLE_PHASES[name] = (_ORACLE_CALLS, _ORACLE_TRUNC)
    _ORACLE_CALLS = _ORACLE_TRUNC = 0



def expert_positions(obs, goal, horizon=ORACLE_HORIZON):
    """env 內建 BFS ＋ 簡化點動力學，跟 exp_oracle_true.py 同一套。"""
    global _ORACLE_CALLS, _ORACLE_TRUNC
    _ORACLE_CALLS += 1
    xy = np.asarray(obs, np.float64); gg = np.asarray(goal, np.float64)
    poss = [xy.copy()]
    truncated = True
    for _ in range(horizon):
        subgoal, bfs = env.unwrapped.get_oracle_subgoal(xy, gg)
        here = env.unwrapped.xy_to_ij(xy)
        target = gg if bfs[here[0], here[1]] == 0 else np.asarray(subgoal)
        a = np.clip(GAIN * (target - xy), -1, 1)
        xy = xy + 0.2 * a
        poss.append(xy.copy())
        if np.linalg.norm(xy - gg) < 0.5:
            truncated = False
            break
    _ORACLE_TRUNC += int(truncated)
    return np.array(poss)


@torch.no_grad()
def oracle_et(obs, goal):
    poss = expert_positions(obs, goal)
    idx = np.unique(np.linspace(0, len(poss) - 1, min(T_CAP, len(poss))).round().astype(int))
    traj = torch.tensor(((poss[idx] - mu) / sd).astype(np.float32), device=device)[None]
    return e_pooler(traj_enc(traj.reshape(-1, 2)).reshape(1, -1, 512))


def normstate(x):
    return ((torch.tensor(np.asarray(x, np.float32), device=device) - MU) / SD)[None]


# ---------- eval-1 / eval-2：離線動作誤差 ----------
print("\n==== eval-1 / eval-2  離線動作誤差（越小越好） ====", flush=True)
N_OFF = int(os.environ.get("LACOT_OFFLINE_BATCHES", 20))
off = {k: [] for k in ("A_data", "B_data", "C", "A_oracle", "B_oracle")}
r_off = np.random.default_rng(12345)
t0 = time.time()
for _ in range(N_OFF):
    traj, mask, s, g, act = make_batch(r_off)
    with torch.no_grad():
        et = etarget(traj, mask)
        cond = condvec(s, g)
        off["A_data"].append(mse(head_A(et), act).item())
        off["B_data"].append(mse(head_B(cond, et), act).item())
        off["C"].append(mse(head_C(cond), act).item())
        # 🚨 同一批 (s,g)，但 e_target 改用 oracle 路徑 ⇒ 差距＝OOD 的大小。
        #    ⚠️ 用【未正規化】的原始座標問 BFS，所以要還原。
        raw_s = s.cpu().numpy() * sd + mu; raw_g = g.cpu().numpy() * sd + mu
        ets = torch.cat([oracle_et(raw_s[i], raw_g[i]) for i in range(B)], 0)
        off["A_oracle"].append(mse(head_A(ets), act).item())
        off["B_oracle"].append(mse(head_B(cond, ets), act).item())
off = {k: float(np.mean(v)) for k, v in off.items()}
print(f"  ({N_OFF} 批 x {B}，耗時 {time.time()-t0:.1f}s)")
print(f"  {'':<26}{'資料集 e_target':>16}{'oracle e_target':>18}")
print(f"  {'A 只吃 e_target':<26}{off['A_data']:>16.4f}{off['A_oracle']:>18.4f}")
print(f"  {'B 吃 cond + e_target':<26}{off['B_data']:>16.4f}{off['B_oracle']:>18.4f}")
print(f"  {'C 只吃 cond（地板）':<26}{off['C']:>16.4f}{'—':>18}")
print(f"  ⇒ OOD 代價 A {off['A_oracle']-off['A_data']:+.4f}　B {off['B_oracle']-off['B_data']:+.4f}", flush=True)
_oracle_phase("offline eval-2")

# ---------- eval-3：rollout success rate ----------
@torch.no_grad()
def policy_chunk(obs, goal, kind):
    s = normstate(obs); g = normstate(goal)
    if kind == "C":
        a = head_C(condvec(s, g))[0]
    else:
        u = oracle_et(obs, goal)          # closed-loop：每個 chunk 用當下位置重算
        a = head_A(u)[0] if kind == "A" else head_B(condvec(s, g), u)[0]
    return np.clip(a.cpu().numpy(), -1.0, 1.0).astype(np.float32)


def rollout(kind, tag):
    succ = ep = 0
    t = time.time()
    for task in range(1, N_TASKS + 1):
        for sd_ in range(EPISODES):
            obs, info = env.reset(seed=1000 * task + sd_, options={"task_id": task, "render_goal": False})
            goal = info["goal"]; success = False; steps = 0
            torch.manual_seed(7 * task + sd_)
            while steps < MAXH and not success:
                for a in policy_chunk(obs, goal, kind):
                    obs, rew, term, trunc, info = env.step(a)
                    steps += 1
                    if info.get("success"):
                        success = True
                    if success or term or trunc or steps >= MAXH:
                        break
            succ += int(success); ep += 1
    print(f"  {tag}: success {succ}/{ep} = {succ/ep:.3f}　({time.time()-t:.0f}s)", flush=True)
    return succ / ep


print(f"\n==== eval-3  SUCCESS RATE（{N_TASKS} tasks x {EPISODES} seeds, MAXH {MAXH}, closed-loop oracle） ====", flush=True)
rates = {}
rates["C_bc_floor"] = rollout("C", "C 只吃 cond（誠實 BC 地板）")
rates["A_etarget_only"] = rollout("A", "A 只吃 oracle e_target")
rates["B_cond_etarget"] = rollout("B", "B 吃 cond + oracle e_target")
_oracle_phase("rollout eval-3")
print(f"⇒ A 減地板 {rates['A_etarget_only']-rates['C_bc_floor']:+.3f}　"
      f"B 減地板 {rates['B_cond_etarget']-rates['C_bc_floor']:+.3f}", flush=True)

# 🚨 沒有這一行，「horizon 放到 800 夠不夠」就沒人知道 —— 而半條路的答案在數值上完全合理。
# 🚨 F3：⛔ 不能只報全域比率。offline 那 1280 次是【長距離 (s,g)】、才會截斷；
#    rollout 那十幾萬次是「當下位置→目標」、每個 chunk 都在變短，幾乎不可能截斷
#    ⇒ 全域分母會把 offline 的 30% 稀釋成 0.3%，然後印一個綠色的 ✅（假綠燈）。
_rates = {}
for _ph, (_c, _t) in _ORACLE_PHASES.items():
    _r = _t / max(_c, 1)
    _rates[_ph] = _r
    print(f"⚠️ oracle 截斷率［{_ph}］：{_t}/{_c} = {_r:.2%}"
          f"{'  ⛔ 有半條路的答案，horizon 要再放大' if _r > 0.01 else '  ✅'}", flush=True)
_tr = max(_rates.values()) if _rates else 0.0
print(f"（horizon={ORACLE_HORIZON}；⭐ 取【各階段最大值】{_tr:.2%} 當總判準，⛔ 不是全域平均）", flush=True)

tag = f"{ENV_NAME.replace('pointmaze-','').replace('-v0','')}_K{K}_c{COND}_ch{CHUNK}_st{STEPS2}_ep{EPISODES}_T{T_CAP}_h{ORACLE_HORIZON}_s1{STEPS1}_s{SEED}"
dst = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results",
                   f"etarget_ceiling_{tag}.json")
os.makedirs(os.path.dirname(dst), exist_ok=True)
with open(dst, "w") as f:
    json.dump(dict(env=ENV_NAME, seed=SEED, K=K, cond=COND, chunk=CHUNK, steps1=STEPS1, steps2=STEPS2,
                   episodes=N_TASKS * EPISODES, maxh=MAXH, match_acc=match_acc,
                   offline=off, rates=rates, tcap=T_CAP, oracle_horizon=ORACLE_HORIZON,
                   oracle_trunc_rate=_tr, oracle_trunc_by_phase=_rates, tcap_requested=T_CAP_REQ, max_train_T=MAX_TRAIN_T, goal_sampling="uniform-official"), f, indent=1)
print(f"寫入 {dst}", flush=True)
