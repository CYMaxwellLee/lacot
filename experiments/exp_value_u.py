"""Step 1 of value-guided refine：V(u) 學得起來嗎？（主人 2026-08-25 核可）

問的是一件事：**把一條軌跡壓成 u 之後，還讀得出「這條路走了多遠」嗎？**
⛔ 完全不碰 flow / refine / action head —— 只要 encoder ＋ 一顆新的 V。

  encoder（traj_enc + e_pooler）  contrastive 訓 1500 步 ⇒ 凍結（跟主線同一套）
  V                              sota_mlp(K*D_MODEL → 512×3 → 1)
  標籤 y = −L                    L ＝ 這條路實際走了幾步（return，每步 −1）

⭐ 主指標：**同一題內的排序準確率**（題庫來自 val 集的跨軌跡多解組）
   ⇒ 🚨 為什麼不是全體排序：全體排序只要學會「s 到 g 的直線距離」就能拿高分
        （主人 8/25 問「不是距離嗎」逼出來的）⇒ 那是假綠燈。
        在【同一題內】直線距離是常數，只有真的讀懂「這條 u 繞了多少」才過得了。

兩個會叫的對照（⛔ 少一個都不算數）：
  ① 直線距離 baseline —— 同題內它是常數 ⇒ 準確率必須 ≈ 50%
  ② 打亂標籤重訓一顆 V —— 也必須 ≈ 50%
     ⇒ 任一個顯著高於 50%，代表評估管線壞了，不是 V 厲害。
"""
import json, os, sys
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lacot.e_target import PerceiverPooler

OGB_DATA = os.environ.get("OGBENCH_DATA_DIR", "/archive/cymaxwelllee/data/ogbench")
ENV_NAME = os.environ.get("LACOT_ENV", "pointmaze-medium-stitch-v0")
SEED = int(os.environ.get("LACOT_SEED", 0))
STEPS1 = int(os.environ.get("LACOT_STEPS1", 1500))     # contrastive encoder
STEPS_V = int(os.environ.get("LACOT_STEPS_V", 5000))   # value head
T_FIX = int(os.environ.get("LACOT_TFIX", 64))          # ⭐ 每條路一律重採樣成這麼多點（見 make_segments）
EPS = float(os.environ.get("LACOT_EPS", 0.5))          # 題庫分組格子（§14.6 定的門檻）
EVAL_PAIRS = int(os.environ.get("LACOT_EVAL_PAIRS", 200000))
SHUFFLE_CTRL = int(os.environ.get("LACOT_SHUFFLE_CTRL", 1))
# ⭐ 2026-08-26 主人核可：把【訓練目標】接回【主指標】。
#    舊版只有 MSE(V(u), -L) ⇒ 訓練在拚「把絕對步數猜準」，評估卻只問「同題內排序」。
#    W_MSE=0 ⇒ 純 ranking（V 的絕對尺度會漂 ⇒ held-out MSE 那欄就不可比，判讀要看 w_mse）
W_MSE = float(os.environ.get("LACOT_W_MSE", 1.0))
W_RANK = float(os.environ.get("LACOT_W_RANK", 1.0))
TRAIN_PAIRS = int(os.environ.get("LACOT_TRAIN_PAIRS", 200000))   # 建【train】題庫用（ranking 要同題兩條）
ORACLE = int(os.environ.get("LACOT_ORACLE", 1))                  # 那把尺：V 直接吃原始座標、不過 encoder
# ⭐ 取樣方式。interp ＝ 插值（預設，沒有重複點）；round ＝ 8/25 的舊版（會從重複點洩漏長度）
#    ⛔ 留著 round 不是為了用它，是為了讓「洩漏版」可以隨時重現當對照，⛔ 不要讓它變成消失的歷史。
SAMPLE = os.environ.get("LACOT_SAMPLE", "interp")
# ⭐ 2026-08-26 主人的三層設計：
#    ① 同一題的幾條路 ⇒ 彼此分長短（誰比較好）
#    ② 別題的路      ⇒ 當負樣本（這根本不是答案）
#    🚨 ② 逼出一個改動：「這條路對不對」是相對於 s→g 的 ⇒ V 必須看得到 s 跟 g。
#       ⇒ 推翻了「V 只吃 u」那個決定（原理由是 u 已經是 s→g 那條路的壓縮，
#         但 8/26 實測 u 連長度都讀不好 ⇒ 它大概也沒好好編碼 s/g）。
#    ⭐ 而給 V 吃 s,g 是安全的：主指標是【同題內】比 ⇒ 同題裡 s,g 固定
#       ⇒ 直線距離是常數 ⇒ 拿它作弊拿不到分。
USE_SG = int(os.environ.get("LACOT_USE_SG", 1))       # V 要不要吃 (s,g)
W_NEG = float(os.environ.get("LACOT_W_NEG", 1.0))     # 「別題的路」負樣本的權重
# ⭐ 2026-08-26 主人：「我比較想的是 contrastive 把負樣本推遠 embedding」
#    ⇒ 動的是【u 的空間本身】，⛔ 不是在 V 外面掛一個判斷。
#    🚨 為什麼這個更根本：refine 是【在 u 的空間裡爬坡】⇒ 空間的幾何本身就該是
#       「好的在這邊、壞的在那邊」，爬坡才自然有方向。
#    ⚠️ 風險：拉太用力 u 會塌成一條「好壞分數」⇒ flow 就沒東西可生 ⇒ 所以權重小、且每輪量塌沒塌。
W_CTR2 = float(os.environ.get("LACOT_W_CTR2", 0.3))   # 新 contrastive 的權重（⛔ 別設大）
assert SAMPLE in ("interp", "round"), f"⛔ LACOT_SAMPLE 只能是 interp / round，收到 {SAMPLE}"
if W_MSE <= 0 and W_RANK <= 0 and W_NEG <= 0:
    raise SystemExit("⛔ W_MSE / W_RANK / W_NEG 不能全是 0 ⇒ loss 沒有任何項")
B, D_MODEL, TEMP = 64, 256, 0.1
K = int(os.environ.get("LACOT_K", 4))     # ⭐ Perceiver 的 query token 數 ⇒ u 的容量
# ⭐ 2026-08-26 主人：「不把 s,g 輸入，當 query 呢」
#    ⇒ query 不再是 K 個【對每條軌跡都一樣】的固定向量，而是由這一題的 (s,g) 產生。
#    ⇒ encoder 因此可以【直接】讀「相對於這一題，這條路繞了多少」，
#      而 u 也不必再花容量去重新編碼 s,g。
#    ⭐ 附帶好處：flow 要生的本來就是 p(u|s,g) ⇒ 兩邊的條件結構對齊了。
SG_QUERY = int(os.environ.get("LACOT_SG_QUERY", 1))
DIM = K * D_MODEL
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device={device} env={ENV_NAME} seed={SEED} steps1={STEPS1} stepsV={STEPS_V} "
      f"T_FIX={T_FIX} eps={EPS}", flush=True)

def load(split=""):
    d = np.load(f"{OGB_DATA}/{ENV_NAME}{split}.npz")
    obs = np.asarray(d["observations"], np.float32)
    term = np.asarray(d["terminals"], bool)
    n = obs.shape[0]
    ends = np.flatnonzero(term)
    assert ends[-1] == n - 1, f"{ENV_NAME}{split}: 最後一筆不是 terminal ⇒ traj_end 尾巴會是未初始化記憶體"
    te = np.empty(n, np.int64); tid = np.empty(n, np.int64)
    st = np.concatenate([[0], ends[:-1] + 1])
    for t, (s0, e0) in enumerate(zip(st, ends)):
        te[s0:e0 + 1] = e0; tid[s0:e0 + 1] = t
    return obs, te, tid, n, len(ends)

OBS, TRAJ_END, TRAJ_ID, N, NTRAJ = load()
VOBS, VTRAJ_END, VTRAJ_ID, VN, VNTRAJ = load("-val")
mu, sd = OBS.mean(0), OBS.std(0) + 1e-6                # ⚠️ 只用 train 的統計量，val 跟著它走
print(f"  train {N} pts / {NTRAJ} traj    val {VN} pts / {VNTRAJ} traj", flush=True)

def sota_mlp(i, h, o, n=2):
    L, p = [], i
    for _ in range(n):
        lin = nn.Linear(p, h); nn.init.xavier_uniform_(lin.weight); nn.init.zeros_(lin.bias)
        L += [lin, nn.GELU(), nn.LayerNorm(h)]; p = h
    lin = nn.Linear(p, o); nn.init.xavier_uniform_(lin.weight); nn.init.zeros_(lin.bias)
    return nn.Sequential(*L, lin)

def make_segments(rows, goals, obs):
    """把 (起點,終點) 重採樣成【固定 T_FIX 個點】的軌跡張量。

    🚨 這裡是 2026-08-25 稽核抓到的致命洞的修法。舊版用
       `np.unique(np.linspace(r, g, min(T_CAP, g-r+1)))`：因為 pointmaze 軌跡就 201 步、
       goal 偏移 <200，min() 永遠取後者、步長恰好 1 ⇒ **真實點數 ≡ L+1**
       ⇒ key_padding_mask 上面直接寫著答案。
       `[稽核 實測]` 光是「數沒被 mask 的 token」當分數 ⇒ 同題排序 1.000；
       隨機初始化、零梯度的 encoder ＋ 線性探針 ⇒ 也是 1.000。
       ⇒ ⛔ 那樣不管跑出什麼數字都跟「u 讀不讀得懂軌跡」無關。

    改成固定點數之後：每條路都是 T_FIX 個 token、mask 全真 ⇒ 長度資訊零洩漏。
    長度只能從【相鄰點的間距】讀 —— 走 60 步的路取 64 點是黏在一起的，
    走 200 步的路取 64 點是跳開的 ⇒ 逼 V 從幾何形狀推「這條路多費事」。
    """
    # 🚨 2026-08-26：舊版是 linspace(...).round() ⇒ 當 L+1 < T_FIX 時索引會【重複】
    #    ⇒ 不重複的點有幾個 ＝ min(L+1, T_FIX) ＝ 長度本身。
    #    `[實測]` 光數不重複的點：T=64 拿 0.898、T=128 拿 0.993、T=201 拿 1.000
    #    ⇒ ⛔ 8/25 的「固定點數」修法只是把洩漏從 mask 搬到重複點上，沒有真的堵掉。
    #    改成【線性插值】：每個點落在兩個真點之間 ⇒ 座標全都不一樣 ⇒ 數不出東西。
    #    ⭐ 長度資訊還在，它藏在【相鄰點的間距】裡 —— 那本來就是我們要它讀的。
    f = np.linspace(rows[:, None].astype(np.float64), goals[:, None].astype(np.float64),
                    T_FIX, axis=1).reshape(len(rows), T_FIX)
    if SAMPLE == "round":                            # 8/25 的舊版：留著當【洩漏對照】，⛔ 不是預設
        pts = obs[f.round().astype(np.int64)]
    else:
        lo_i = np.floor(f).astype(np.int64)
        hi_i = np.minimum(lo_i + 1, goals[:, None])  # ⚠️ 夾在終點內，⛔ 不可以跨到下一條軌跡
        w = (f - lo_i)[..., None].astype(np.float32)
        pts = obs[lo_i] * (1.0 - w) + obs[hi_i] * w
    traj = ((pts - mu) / sd).astype(np.float32)
    mask = np.zeros((len(rows), T_FIX), bool)          # 全 False ＝ 全是真點
    return traj, mask

def sample_pairs(rng, n, obs, te, minlen=1):
    """抽同一條軌跡內的 (i,j)。

    ⚠️ 訓練【不】濾短路 —— 要讓 V 看過完整的長度範圍，否則爬坡把 u 推短時它只能外插。
       短路只在【題庫】那邊濾（配對要有區辨力）。
    🚨 舊版 `while len(out_r) < n` 數的是 list 裡【陣列的個數】不是樣本數
       ⇒ n=200000 時要跑 20 萬圈、約 576 GB RAM（稽核實測）⇒ 這支從沒用預設值跑過。
    🚨 舊版還用 reject（`ok = (g-r) >= minlen` 丟掉整筆）⇒ 每條軌跡最後 minlen 步永遠當不了起點。
       主線 `scratch_lacot_rollout.py` L79-81 把同一個病記成 F6 並改成 clamp；這裡跟上。
    """
    got_r, got_g, have = [], [], 0
    while have < n:
        r = rng.integers(0, len(obs), size=n)
        e = te[r]
        off = rng.integers(minlen, 200, size=n)
        g = np.minimum(r + off, e)
        g = np.maximum(g, np.minimum(r + minlen, e))    # clamp，⛔ 不 reject ⇒ 起點分布不動
        ok = (g - r) >= minlen                          # 只有「整條軌跡尾巴不足 minlen」才會落空
        got_r.append(r[ok]); got_g.append(g[ok]); have += int(ok.sum())
    return np.concatenate(got_r)[:n], np.concatenate(got_g)[:n]

def sg_of(rows, goals, obs):
    """(s,g) 正規化之後接成一條 4 維向量。⚠️ 用 train 的 mu/sd，val 跟著它走。"""
    return np.concatenate([(obs[rows] - mu) / sd, (obs[goals] - mu) / sd], axis=1).astype(np.float32)


def train_batch(rng):
    r, g = sample_pairs(rng, B, OBS, TRAJ_END)
    traj, mask = make_segments(r, g, OBS)
    s = (OBS[r] - mu) / sd; gg = (OBS[g] - mu) / sd
    T = lambda x: torch.from_numpy(np.asarray(x, np.float32)).to(device)
    return T(traj), torch.from_numpy(mask).to(device), T(s), T(gg), T((g - r).astype(np.float32))

# ---------- encoder（可以有好幾顆：訓練過的、隨機沒訓的） ----------
def new_encoder(seed):
    torch.manual_seed(seed)
    te = sota_mlp(2, 512, 512).to(device)
    ep = PerceiverPooler(512, D_MODEL, K, 2, 4, max_len=max(512, T_FIX)).to(device)
    # ⭐ SG_QUERY：(s,g) 這 4 個數字 → K 個 query token
    qg = sota_mlp(4, 512, K * D_MODEL).to(device) if SG_QUERY else None
    return te, ep, qg

def etarget_with(enc, traj, mask, sg=None):
    te, ep, qg = enc
    Bc, Tc, _ = traj.shape
    ctx = te(traj.reshape(Bc * Tc, 2)).reshape(Bc, Tc, 512)
    q = None
    if SG_QUERY:
        # ⛔ 別靜默退回 learnable query —— 那會讓「有沒有用 s,g 當 query」變成看不見的差異
        assert sg is not None, "⛔ SG_QUERY=1 但沒有把 (s,g) 傳進 etarget_with"
        q = qg(sg).reshape(Bc, K, D_MODEL)
    return ep(ctx, key_padding_mask=mask, queries=q)

def features(enc, traj, mask, mode, sg=None):
    """V 的輸入。⭐ 這是「那把尺」的關鍵：兩種模式除了【看到什麼】以外，其他完全一樣。

      mode="enc"  ⇒ u ＝ e_target(軌跡)          （1024 維，encoder 壓出來的）
      mode="raw"  ⇒ 攤平的原始座標              （T_FIX×2＝128 維，資訊沒有經過任何壓縮）

    ⇒ 兩邊同 loss、同題庫、同評估、同 V 容量 ⇒ 差距只能歸因於 encoder 有沒有把長度留下來。
    """
    if mode == "raw":
        x = traj.reshape(traj.shape[0], -1)
    else:
        with torch.no_grad():
            x = etarget_with(enc, traj, mask, sg).reshape(traj.shape[0], -1)
    if USE_SG:
        assert sg is not None, "⛔ USE_SG=1 但沒有把 (s,g) 傳進來"
        x = torch.cat([x, sg], dim=1)
    return x

def freeze(enc):
    for m in enc:
        if m is None:
            continue
        m.eval()
        for p in m.parameters():
            p.requires_grad_(False)

rng = np.random.default_rng(SEED)
trained_enc = new_encoder(SEED)
torch.manual_seed(SEED)
sg_c = sota_mlp(2, 512, 512).to(device)
q_pooler = PerceiverPooler(512, D_MODEL, K, 2, 4, max_len=max(512, T_FIX)).to(device)
opt1 = torch.optim.Adam([p for m in (trained_enc[0], trained_enc[1], trained_enc[2],
                                        sg_c, q_pooler) if m is not None
                         for p in m.parameters()], lr=1e-3)
lab = torch.arange(B, device=device)

# ---------- stage 2：訓 V(u) ----------
def train_value(enc, shuffle_labels, tag, mode="enc", steps=None):
    """⭐ 2026-08-26：loss ＝ W_MSE·MSE(V,-L) ＋ W_RANK·softplus(V_長 − V_短)

    為什麼要加 ranking：舊版只有 MSE ⇒ 訓練在拚「把絕對步數猜準」，
    但主指標問的是「同一題內短的分數有沒有比較高」⇒ 兩件事，力氣花錯地方。
    ⚠️ ranking 的配對【一定】要同題（見 rank_batch），否則 V 學會直線距離就能壓低 loss。
    ⚠️ MSE 那一路照舊用【不濾短路】的隨機採樣 ⇒ 保住完整長度範圍（8/25 ⑥-2 的決策）。
    """
    torch.manual_seed(SEED + 5)          # ⚠️ matched control：兩組 V 的初始化一樣，只差標籤
    r2 = np.random.default_rng(SEED + 5)
    rr = np.random.default_rng(SEED + 6)
    rn = np.random.default_rng(SEED + 7)
    in_dim = (T_FIX * 2 if mode == "raw" else DIM) + (4 if USE_SG else 0)
    V = sota_mlp(in_dim, 512, 1, n=3).to(device)
    opt = torch.optim.Adam(V.parameters(), lr=3e-4)
    n_steps = STEPS_V if steps is None else steps
    # ⭐ 2026-08-26 對照④：steps=0 ⇒ V 完全不訓練，只有隨機初始化。
    #    🚨 它才是主指標真正要贏的底線 —— 「u 的幾何結構本身白送多少分」。
    #    ⛔ 拿 0.5 當底線是錯的：0.5 只在「u 裡沒有任何跟長度相關的結構」時才成立。
    print(f"stage 2  訓 V ({tag})  mode={mode} steps={n_steps} "
          f"w_mse={W_MSE} w_rank={W_RANK} ...", flush=True)
    for stp in range(n_steps):
        loss = torch.zeros((), device=device)
        m_item = r_item = n_item = float("nan")
        if W_MSE > 0:
            traj, mask, s, g, L = train_batch(r2)
            y = -L
            if shuffle_labels:                              # 對照組：標籤打亂 ⇒ 應該學不到東西
                y = y[torch.randperm(len(y), device=device)]
            lm = F.mse_loss(V(features(enc, traj, mask, mode,
                                      torch.cat([s, g], 1))).squeeze(-1), y)
            loss = loss + W_MSE * lm; m_item = lm.item()
        if W_RANK > 0:                                      # ① 同一題內分長短
            ta, ma_, sga, tb, mb_, sgb = rank_batch(rr, TG, OBS, max(B // 2, 1))
            va = V(features(enc, ta, ma_, mode, sga)).squeeze(-1)
            vb = V(features(enc, tb, mb_, mode, sgb)).squeeze(-1)
            d = vb - va                                     # a 比較短 ⇒ 要 va > vb ⇒ d 要往負的走
            if shuffle_labels:                              # 對照：隨機翻轉誰是「短的」⇒ 沒有方向可學
                fl = torch.rand(len(d), device=device) < 0.5
                d = torch.where(fl, -d, d)
            lr_ = F.softplus(d).mean()
            loss = loss + W_RANK * lr_; r_item = lr_.item()
        if W_NEG > 0:                                       # ② 別題的路 ⇒ 這根本不是答案
            tp, mp, tn, mn, sgn = neg_batch(rn, TG, OBS, max(B // 2, 1))
            vp = V(features(enc, tp, mp, mode, sgn)).squeeze(-1)
            vn = V(features(enc, tn, mn, mode, sgn)).squeeze(-1)
            dn = vn - vp                                    # 對的路要贏錯的路
            if shuffle_labels:
                fl = torch.rand(len(dn), device=device) < 0.5
                dn = torch.where(fl, -dn, dn)
            ln_ = F.softplus(dn).mean()
            loss = loss + W_NEG * ln_; n_item = ln_.item()
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if (stp + 1) % 1000 == 0:
            print(f"  step {stp+1}  mse {m_item:.2f}  rank {r_item:.4f}  neg {n_item:.4f}",
                  flush=True)
    V.eval()
    # ⚠️ held-out MSE：⛔ 沒有它就分不出「沒訓起來」跟「u 裡根本沒有 L」
    # 🚨 但 W_MSE=0 時 V 的絕對尺度會漂 ⇒ 這個數字【不可比】⇒ 判讀要先看 w_mse
    r3 = np.random.default_rng(9999)
    r4 = np.random.default_rng(31337)
    r5 = np.random.default_rng(24680)
    with torch.no_grad():
        mses = []
        for _ in range(20):
            traj, mask, s, g, L = train_batch(r3)
            mses.append(F.mse_loss(V(features(enc, traj, mask, mode,
                                             torch.cat([s, g], 1))).squeeze(-1), -L).item())
        oks = tot = 0                                       # ⭐ 尺度無關的健檢：train 集同題排序
        for _ in range(20):
            ta, ma_, sga, tb, mb_, sgb = rank_batch(r4, TG, OBS, 256)
            va = V(features(enc, ta, ma_, mode, sga)).squeeze(-1)
            vb = V(features(enc, tb, mb_, mode, sgb)).squeeze(-1)
            oks += int((va > vb).sum().item()); tot += len(va)
        oks_n = tot_n = 0            # ⭐ V 分不分得出「這條路是不是這一題的答案」
        for _ in range(20):
            tp, mp, tn, mn, sgn = neg_batch(r5, TG, OBS, 256)
            vp = V(features(enc, tp, mp, mode, sgn)).squeeze(-1)
            vn = V(features(enc, tn, mn, mode, sgn)).squeeze(-1)
            oks_n += int((vp > vn).sum().item()); tot_n += len(vp)
    # ⚠️ 這個是【train 集】的（V 訓練時看過這些組）⇒ 只當「有沒有訓起來」的診斷，⛔ 不是主指標
    return V, float(np.mean(mses)), float(oks / max(tot, 1)), float(oks_n / max(tot_n, 1))

# ---------- 題庫：val 集的「同一題多解」配對 ----------
print("\n建題庫（val 集，跨軌跡多解組）...", flush=True)
rq = np.random.default_rng(1234)
def build_groups(obs, te, tid, n_pairs, seed, minlen=20, who=""):
    """把 (s,g) 抽樣分成【同一題、不同軌跡】的多解組。

    ⭐ 2026-08-26 抽成函數：val（題庫）跟 train（ranking loss 的配對來源）必須用
       完全一樣的門檻，否則兩邊的「同一題」不是同一個定義。
       ⛔ 這也是 handoff 一直記的那個病（同一個修法只落在一支腳本）的預防。
    ε 是【格子邊長】不是距離：起點落同一格、終點也落同一格 ⇒ 同一題。
    ⇒ 偏誤方向是安全的：可能把很近的兩點拆開（漏），不可能把差很遠的兩題併在一起。
    """
    rq_ = np.random.default_rng(seed)
    qr, qg = sample_pairs(rq_, n_pairs, obs, te, minlen=minlen)   # ⚠️ 題庫才濾短路，訓練不濾
    qS, qG, qL, qT = obs[qr], obs[qg], (qg - qr).astype(np.int64), tid[qr]
    lo = obs.min(0)
    span = np.ceil((obs.max(0) - lo) / EPS).astype(np.int64) + 1
    sb = ((qS - lo) / EPS).astype(np.int64); gb = ((qG - lo) / EPS).astype(np.int64)
    key = sb[:, 0]
    for arr, w in ((sb[:, 1], span[1]), (gb[:, 0], span[0]), (gb[:, 1], span[1])):
        key = key * w + arr
    order = np.argsort(key, kind="stable")
    ks, L_, T_, R_, G_ = key[order], qL[order], qT[order], qr[order], qg[order]
    _, gs_, gc_ = np.unique(ks, return_index=True, return_counts=True)
    prs, pgr = [], []                         # (idxA, idxB) 同一題、不同軌跡；以及它來自哪一組
    for gi in np.flatnonzero(gc_ >= 2):
        a, b = gs_[gi], gs_[gi] + gc_[gi]
        _, first = np.unique(T_[a:b], return_index=True)
        if len(first) < 2:
            continue
        fi = a + first                        # ⚠️ 每條軌跡在每組只留一個代表
        for x in range(len(fi)):
            for y_ in range(x + 1, len(fi)):
                prs.append((fi[x], fi[y_])); pgr.append(gi)
    if len(prs) < 100:                        # ⚠️ 空/太小的題庫要在這裡停，⛔ 不要變成難懂的 IndexError
        raise SystemExit(f"⛔ {who}題庫只有 {len(prs)} 個配對 —— 調大 pairs 數或 LACOT_EPS")
    prs = np.array(prs); pgr = np.array(pgr)
    # ⚠️ 健康檢查（measure_stitch_multipath.py 明說必做）：組內 (s,g) 散布必須 << EPS
    sp_ = []
    for gi in np.unique(pgr):
        a, b = gs_[gi], gs_[gi] + gc_[gi]
        ss_, gg_ = obs[R_[a:b]], obs[G_[a:b]]
        sp_.append(max(np.abs(ss_ - ss_.mean(0)).max(), np.abs(gg_ - gg_.mean(0)).max()))
    return dict(pairs=prs, pair_group=pgr, lsort=L_, tsort=T_, rsort=R_, gsort=G_,
                gstart=gs_, gcount=gc_, spread=float(np.median(sp_)))

_VG = build_groups(VOBS, VTRAJ_END, VTRAJ_ID, EVAL_PAIRS, 1234, 20, "val ")
pairs, pair_group = _VG["pairs"], _VG["pair_group"]
lsort, tsort, rsort, gsort = _VG["lsort"], _VG["tsort"], _VG["rsort"], _VG["gsort"]
gstart, gcount = _VG["gstart"], _VG["gcount"]
uniq = np.unique(pairs.reshape(-1))
print(f"  題庫配對數 {len(pairs)}（來自 val 集，⛔ 訓練沒看過）", flush=True)
print(f"  ⚠️ 但獨立單位少很多：{len(uniq)} 條不同的路 / "
      f"{len(np.unique(pair_group))} 個題目 / {len(np.unique(tsort[uniq]))} 條軌跡", flush=True)
dL = np.abs(lsort[pairs[:, 0]] - lsort[pairs[:, 1]])
print(f"  兩條路的長度差 |ΔL|：中位 {np.median(dL):.0f}  p90 {np.percentile(dL,90):.0f}  "
      f"⚠️ ΔL=0 的佔 {float((dL==0).mean())*100:.2f}%（這些配對會被排除）", flush=True)
# ⚠️ 健康檢查（measure_stitch_multipath.py 明說必做）：組內 (s,g) 散布必須 << EPS，
#    否則「同一題」只是把不同的題硬圈在一起。
SG_SPREAD = _VG["spread"]
print(f"  ⚠️ 組內 (s,g) 散布中位 {SG_SPREAD:.3f}（必須 << ε={EPS}）"
      f"{'  ✓' if SG_SPREAD < EPS * 0.6 else '  🚨 太大，題目被混在一起了'}", flush=True)

# ---------- 【train】題庫：ranking loss 的配對來源 ----------
# 🚨 ⛔ 不能拿上面那份（val）來訓練 —— 那是評估用的題庫，用它等於直接洩漏。
if W_RANK > 0 or W_CTR2 > 0:
    print("\n建 train 題庫（ranking / contrastive 用）...", flush=True)
    TG = build_groups(OBS, TRAJ_END, TRAJ_ID, TRAIN_PAIRS, 777, 20, "train ")
    print(f"  配對數 {len(TG['pairs'])} / {len(np.unique(TG['pair_group']))} 個題目"
          f" / 組內散布中位 {TG['spread']:.3f}"
          f"{'  ✓' if TG['spread'] < EPS * 0.6 else '  🚨 太大'}", flush=True)
    _tl = TG["lsort"]; _dl = np.abs(_tl[TG["pairs"][:, 0]] - _tl[TG["pairs"][:, 1]])
    print(f"  |ΔL| 中位 {np.median(_dl):.0f}   ΔL=0 佔 {float((_dl==0).mean())*100:.2f}%"
          f"（訓練時排除）", flush=True)
    # ⚠️ 不需要查 val/train 軌跡重疊：它們是兩個獨立的 npz（{env}.npz 與 {env}-val.npz），
    #    軌跡 id 各自從 0 編號 ⇒ 交集比對沒有意義。⛔ 別寫一個永遠會過的 assert 來假裝有擋。
else:
    TG = None

def rank_batch(rng, G, obs, n):
    """從同題多解組抽 n 對，回 (a=比較短那條, b=比較長那條)。

    ⚠️ 一定要【同一題】：跨題的話 V 只要學會「這兩點離很遠」就能把 loss 壓下去
       ⇒ 那是我們親手餵的捷徑，跟主指標防的是同一件事。
    ⚠️ ΔL=0 的配對沒有方向 ⇒ 排除（跟評估的 correctness 一致）。
    🚨 這裡的 while 數的是【樣本數】不是 list 長度 —— 8/25 稽核抓到的那個 bug 的同款形狀。
    """
    P_, Ls = G["pairs"], G["lsort"]
    got_a, got_b, have = [], [], 0
    while have < n:
        sel = rng.integers(0, len(P_), size=max(n, 64))
        ia_, ib_ = P_[sel, 0], P_[sel, 1]
        keep = Ls[ia_] != Ls[ib_]
        got_a.append(ia_[keep]); got_b.append(ib_[keep])
        have += int(keep.sum())                       # ⭐ 樣本數，⛔ 不是 len(list)
    ia_ = np.concatenate(got_a)[:n]; ib_ = np.concatenate(got_b)[:n]
    swap = Ls[ia_] > Ls[ib_]                          # 讓 a 一律是【比較短】的那條
    A_ = np.where(swap, ib_, ia_); B_ = np.where(swap, ia_, ib_)
    ta, ma = make_segments(G["rsort"][A_], G["gsort"][A_], obs)
    tb, mb = make_segments(G["rsort"][B_], G["gsort"][B_], obs)
    to = lambda x: torch.from_numpy(np.asarray(x, np.float32)).to(device)
    sga = to(sg_of(G["rsort"][A_], G["gsort"][A_], obs))
    sgb = to(sg_of(G["rsort"][B_], G["gsort"][B_], obs))
    return (to(ta), torch.from_numpy(ma).to(device), sga,
            to(tb), torch.from_numpy(mb).to(device), sgb)


def neg_batch(rng, G, obs, n):
    """⭐ 主人 8/26 的②：給【同一個 s→g】，配一條【別題】的路當負樣本。

    ⇒ 教 V 的是「這條路根本不是答案」，⛔ 不只是「這條路比較長」。
    🚨 這堵的洞是：flow 生出一個 u，它對應的路到不了 g 但看起來很短
       ⇒ 現在的 V 會給它高分 ⇒ 爬坡就往那裡走。
    ⚠️ 負樣本用的是【正樣本的 (s,g)】＋【別題的軌跡】—— 這個配對本身就是「錯」的定義。
    """
    P_, Ls = G["pairs"], G["lsort"]
    R_, Gg_ = G["rsort"], G["gsort"]
    got_p, got_n, have = [], [], 0
    while have < n:
        pos = P_[rng.integers(0, len(P_), size=n), 0]
        neg = rng.integers(0, len(Ls), size=n)
        # ⚠️ 防呆：隨機抽到剛好同一題的要丟掉，⛔ 否則我們把正樣本當負樣本在教
        same = ((np.abs(obs[R_[neg]] - obs[R_[pos]]).max(1) < EPS) &
                (np.abs(obs[Gg_[neg]] - obs[Gg_[pos]]).max(1) < EPS))
        got_p.append(pos[~same]); got_n.append(neg[~same]); have += int((~same).sum())
    pos = np.concatenate(got_p)[:n]; neg = np.concatenate(got_n)[:n]
    tp, mp = make_segments(R_[pos], Gg_[pos], obs)
    tn, mn = make_segments(R_[neg], Gg_[neg], obs)
    sg = sg_of(R_[pos], Gg_[pos], obs)          # ⭐ 兩邊都用【正樣本的 s,g】
    to = lambda x: torch.from_numpy(np.asarray(x, np.float32)).to(device)
    return (to(tp), torch.from_numpy(mp).to(device),
            to(tn), torch.from_numpy(mn).to(device), to(sg))


# ---------- stage 1：訓 encoder（⚠️ 位置在 TG 之後，因為新的 contrastive 要用同題配對） ----------
def eff_dim(x):
    """u 的有效維度（participation ratio，正規化到 0~1）。

    🚨 這是【防塌】的守門員：contrastive 拉太用力時 u 會塌成一條「好壞分數」
       ⇒ 所有好路擠一邊、壞路擠另一邊 ⇒ u 不再是「這條路長什麼樣」⇒ flow 沒東西可生。
    ⛔ 別等結果出來才發現，每隔一段就量。
    """
    with torch.no_grad():
        z = x.reshape(len(x), -1).float()
        z = z - z.mean(0, keepdim=True)
        v = torch.linalg.svdvals(z) ** 2
        return float((v.sum() ** 2) / (v ** 2).sum() / len(v))


print("stage 1  contrastive e_target ...", flush=True)
rc = np.random.default_rng(SEED + 11)
ed_hist = []
for stp in range(STEPS1):
    traj, mask, s, g, _ = train_batch(rng)
    et = etarget_with(trained_enc, traj, mask, torch.cat([s, g], 1))
    q = q_pooler(torch.stack([sg_c(s), sg_c(g)], 1))
    logits = (F.normalize(q.reshape(B, -1), dim=1) @ F.normalize(et.reshape(B, -1), dim=1).t()) / TEMP
    loss = 0.5 * (F.cross_entropy(logits, lab) + F.cross_entropy(logits.t(), lab))
    c2_item = float("nan")
    if W_CTR2 > 0 and TG is not None:
        # ⭐ 主人 8/26 的設計：同一個 s→g 底下，把【比較差的路】在 u 空間裡推遠。
        #    anchor   ＝ (s,g)
        #    正樣本   ＝ 同題的【短】路
        #    難負樣本 ＝ 同題的【長】路      ← 逼 encoder 非讀形狀不可
        #    易負樣本 ＝ batch 內別題的短路  ← InfoNCE 本來就有的
        ta, ma_, sga, tb, mb_, _sgb = rank_batch(rc, TG, OBS, B)
        ua = etarget_with(trained_enc, ta, ma_, sga)
        ub = etarget_with(trained_enc, tb, mb_, _sgb)
        q2 = q_pooler(torch.stack([sg_c(sga[:, :2]), sg_c(sga[:, 2:])], 1))
        qn = F.normalize(q2.reshape(B, -1), dim=1)
        an = F.normalize(ua.reshape(B, -1), dim=1)
        bn = F.normalize(ub.reshape(B, -1), dim=1)
        # (B, 2B)：前 B 欄是各題的短路（對角＝正樣本），後 B 欄是長路（第 i 欄＝難負樣本）
        l2 = torch.cat([qn @ an.t(), qn @ bn.t()], dim=1) / TEMP
        c2 = F.cross_entropy(l2, lab)
        loss = loss + W_CTR2 * c2; c2_item = c2.item()
    opt1.zero_grad(set_to_none=True); loss.backward(); opt1.step()
    if (stp + 1) % max(STEPS1 // 5, 1) == 0 or stp == 0:
        with torch.no_grad():
            ed = eff_dim(etarget_with(trained_enc, traj, mask, torch.cat([s, g], 1)))
        ed_hist.append(ed)
        print(f"  step {stp+1}  contrastive {loss.item():.4f}  同題項 {c2_item:.4f}"
              f"  u 有效維度 {ed:.3f}"
              f"{'   🚨 塌了' if ed < 0.02 else ''}", flush=True)
freeze(trained_enc)
print(f"  ⇒ encoder 凍結（u 有效維度 起 {ed_hist[0]:.3f} → 末 {ed_hist[-1]:.3f}）", flush=True)

# ⭐ 第三個對照（2026-08-25 稽核指定，⛔ 少了它主指標就不可信）：
#    一顆【完全沒訓練】的隨機 encoder。固定點數重採樣之後它應該掉回 ~0.5。
random_enc = new_encoder(SEED + 31337)
freeze(random_enc)
# ⚠️ 基準：一顆【沒訓練】的 encoder 的有效維度。⛔ 沒有它就不知道上面那個數字算不算塌。
with torch.no_grad():
    _t0, _m0, _s0, _g0, _ = train_batch(np.random.default_rng(4321))
    ED_RAND = eff_dim(etarget_with(random_enc, _t0, _m0, torch.cat([_s0, _g0], 1)))
# ⚠️ f-string 裡的表達式在 Python 3.11 不能跨行 ⇒ 先算好再放進去
_ed_note = ""
if ED_RAND > 0:
    _ed_dir = "掉了" if ed_hist[-1] < ED_RAND else "升了"
    _ed_note = f"（{_ed_dir} {abs(100 * (ed_hist[-1] / ED_RAND - 1)):.0f}%）"
print(f"  ⚠️ 基準：沒訓練的 encoder，u 有效維度 {ED_RAND:.3f}"
      f"   ⇒ 訓練後 {ed_hist[-1]:.3f} {_ed_note}", flush=True)

def pure_u(enc, idx_sorted):
    """純 u，⛔ 不含 (s,g)。線性探針要用這個，否則量 (s,g) 等於直接看答案。"""
    outs = []
    for a in range(0, len(idx_sorted), 256):
        sl = idx_sorted[a:a + 256]
        traj, mask = make_segments(rsort[sl], gsort[sl], VOBS)
        sgp = torch.from_numpy(sg_of(rsort[sl], gsort[sl], VOBS)).to(device)
        with torch.no_grad():
            uu = etarget_with(enc, torch.from_numpy(traj).to(device),
                              torch.from_numpy(mask).to(device), sgp)
        outs.append(uu.reshape(len(sl), -1).cpu().numpy().astype(np.float64))
    return np.concatenate(outs)


def lin_r2(X, Y, seed=0):
    """線性探針的 held-out R²。

    ⭐ 用【線性】而且【不訓練】⇒ 沒有「訓練不足 / 架構不對」這些干擾，
       量到的就是「這個資訊在不在 u 裡面，而且是不是線性可讀」。
    🚨 一定要 held-out：u 有上千維、樣本只有幾千筆，in-sample R² 會虛高到沒有意義。
    """
    Y = np.asarray(Y, np.float64)
    if Y.ndim == 1:
        Y = Y[:, None]
    idx = np.random.default_rng(seed).permutation(len(X))
    k = len(X) // 2
    tr, te = idx[:k], idx[k:]
    # 🚨 2026-08-26 實測：u 有 1024~4096 維、訓練樣本只有 3308 筆 ⇒ 方程式欠定
    #    ⇒ 直接 lstsq 的解在測試集上發散（R² 跑到 -14872）。
    #    ⛔ 好在它爛得很明顯，要是吐出 0.6 這種合理的數字就騙過去了。
    #    修法：先用【訓練集】的 PCA 降到樣本數的八分之一以下，再做最小平方。
    Xc = X - X[tr].mean(0, keepdims=True)
    ncomp = int(min(X.shape[1], max(8, len(tr) // 8)))
    if X.shape[1] > ncomp:
        _, _, Vt = np.linalg.svd(Xc[tr], full_matrices=False)
        Xc = Xc @ Vt[:ncomp].T                    # ⚠️ 投影矩陣只能從【訓練集】算，⛔ 不能用全體
    Xc = np.concatenate([Xc, np.ones((len(Xc), 1))], 1)
    W_, *_ = np.linalg.lstsq(Xc[tr], Y[tr], rcond=None)
    res = Y[te] - Xc[te] @ W_
    ss_res = (res ** 2).sum(0)
    ss_tot = ((Y[te] - Y[tr].mean(0)) ** 2).sum(0)   # ⚠️ 基準用訓練集的均值，⛔ 不是測試集自己的
    return float(np.mean(1 - ss_res / np.maximum(ss_tot, 1e-12)))


def u_of(enc, idx_sorted, mode="enc"):
    """把題庫索引批次轉成 V 的輸入（mode="raw" ⇒ 那把尺，直接吃原始座標）。"""
    outs = []
    for a in range(0, len(idx_sorted), 256):
        sl = idx_sorted[a:a + 256]
        traj, mask = make_segments(rsort[sl], gsort[sl], VOBS)
        sgv = torch.from_numpy(sg_of(rsort[sl], gsort[sl], VOBS)).to(device)
        with torch.no_grad():
            xx = features(enc, torch.from_numpy(traj).to(device),
                          torch.from_numpy(mask).to(device), mode, sgv)
        outs.append(xx.cpu())
    return torch.cat(outs)

remap = {int(v): i for i, v in enumerate(uniq)}
ia = np.array([remap[int(v)] for v in pairs[:, 0]])
ib = np.array([remap[int(v)] for v in pairs[:, 1]])
La, Lb = lsort[pairs[:, 0]], lsort[pairs[:, 1]]
dist_a = np.linalg.norm(VOBS[rsort[pairs[:, 0]]] - VOBS[gsort[pairs[:, 0]]], axis=1)
dist_b = np.linalg.norm(VOBS[rsort[pairs[:, 1]]] - VOBS[gsort[pairs[:, 1]]], axis=1)
GRP_IDX = {int(c): np.flatnonzero(pair_group == c) for c in np.unique(pair_group)}
GRP_KEYS = np.array(list(GRP_IDX.keys()))
_boot_rng = np.random.default_rng(4242)

def correctness(score_a, score_b, gap_min=0):
    """⚠️ gap_min 至少 1：ΔL=0 的配對兩邊都是 False ⇒ 舊版把它們全算「答對」
       ⇒ 一顆【常數輸出】的 V 就能拿 0.517（稽核實測）。"""
    m = np.abs(La - Lb) >= max(gap_min, 1)
    return (score_a[m] > score_b[m]) == (La[m] < Lb[m]), m

def rank_acc(score_a, score_b, gap_min=0, boot=0):
    ok, m = correctness(score_a, score_b, gap_min)
    if m.sum() == 0:
        return None, 0, None
    acc = float(ok.mean())
    ci = None
    if boot:
        # 🚨 配對【不獨立】：同一題展開成 C(n,2) 個配對、同一條路出現在多個配對裡
        #    ⇒ naive binomial CI 會窄約五倍（稽核實測）。按【題目】做 cluster bootstrap。
        full = np.zeros(len(La), bool); full[np.flatnonzero(m)] = ok
        accs = []
        for _ in range(boot):
            pick = _boot_rng.choice(GRP_KEYS, size=len(GRP_KEYS), replace=True)
            idx = np.concatenate([GRP_IDX[int(c)] for c in pick])
            idx = idx[m[idx]]
            if len(idx):
                accs.append(full[idx].mean())
        if accs:
            ci = [float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))]
    return acc, int(m.sum()), ci

def spearman(x, y):
    """midrank（tie 取平均名次）—— L 是整數，tie 很多。"""
    def midrank(v):
        o = np.argsort(v, kind="stable"); r = np.empty(len(v), np.float64)
        r[o] = np.arange(len(v), dtype=np.float64)
        vs = v[o]; i = 0
        while i < len(vs):
            j = i
            while j + 1 < len(vs) and vs[j + 1] == vs[i]:
                j += 1
            if j > i:
                r[o[i:j + 1]] = (i + j) / 2.0
            i = j + 1
        return r
    rx, ry = midrank(np.asarray(x, np.float64)), midrank(np.asarray(y, np.float64))
    rx -= rx.mean(); ry -= ry.mean()
    return float((rx @ ry) / (np.linalg.norm(rx) * np.linalg.norm(ry) + 1e-12))

report = {"env": ENV_NAME, "seed": SEED, "steps1": STEPS1, "steps_v": STEPS_V, "T_FIX": T_FIX,
          "eps": EPS, "eval_pairs": EVAL_PAIRS, "minlen_query": 20, "shuffle_ctrl": SHUFFLE_CTRL,
          "w_mse": W_MSE, "w_rank": W_RANK, "train_pairs": TRAIN_PAIRS, "oracle": bool(ORACLE),
          "sample": SAMPLE, "use_sg": bool(USE_SG), "w_neg": W_NEG, "w_ctr2": W_CTR2, "K": K,
          "sg_query": bool(SG_QUERY),
          "u_eff_dim": {"start": ed_hist[0], "end": ed_hist[-1], "random_enc": ED_RAND},
          "n_pairs": int(len(pairs)), "n_paths": int(len(uniq)),
          "n_groups": int(len(np.unique(pair_group))), "n_val_traj": int(len(np.unique(tsort[uniq]))),
          "sg_spread_median": SG_SPREAD, "runs": {}}
GAPS = [0, 10, 30, 60]
ARMS = [("real", trained_enc, False, "enc", None, "V(u)  訓練過的 encoder"),
        ("shuffled", trained_enc, True, "enc", None, "對照①  標籤打亂"),
        ("random_enc", random_enc, False, "enc", None, "對照②  encoder 完全沒訓練"),
        ("untrained_V", trained_enc, False, "enc", 0, "對照④  V 完全不訓練（u 白送多少）")]
if not SHUFFLE_CTRL:
    ARMS = [ARMS[0]]                         # ⚠️ 關掉對照時 verdict 也會跟著標成不可信，見下面
if ORACLE:
    # 尺①：同 loss、同題庫、同評估、同 V 容量，只差「看到的是原始座標而不是 u」。
    # 🚨 2026-08-26 實測：它反而【低於】V(u) ⇒ 它不是上限。攤平的座標餵 MLP 讀不到
    #    「相鄰點間距」這種序列結構，而 encoder 有 attention ⇒ 這把尺量到的是架構差，不是資訊差。
    #    ⇒ 留著當紀錄，⛔ 別再拿它當上限用。真的上限看尺②（幾何長度，不用學）。
    ARMS.append(("oracle", None, False, "raw", None, "尺①　V 直接吃原始座標（⚠️ 架構受限）"))
U_CACHE = {}
for name, enc, shuf, mode, stp_, tag in ARMS:
    key_id = (mode, id(enc))
    if key_id not in U_CACHE:
        U_CACHE[key_id] = u_of(enc, uniq, mode)
    V, heldout_mse, tr_rank, tr_neg = train_value(enc, shuf, tag, mode, stp_)
    with torch.no_grad():
        sc = V(U_CACHE[key_id].to(device)).squeeze(-1).cpu().numpy()
    sa, sb_ = sc[ia], sc[ib]
    print(f"\n=== {tag} ===", flush=True)
    row = {"heldout_mse": heldout_mse, "trainset_rank_acc": tr_rank,
           "trainset_neg_acc": tr_neg, "mode": mode,
           "spearman_all": spearman(sc, -lsort[uniq]), "rank_acc": {}}
    # ⭐ 主人 8/26：「0.5 到 0.8 區間太小，不適合實際使用」⇒ 換一個有【物理單位】的問法：
    #    拿 V 去【選路】，選出來的那條實際比隨機挑少走幾步、離最好的還差幾步。
    #    ⇒ 單位是「步」，回答的是「V 好不好用」，⛔ 不是「V 排序準不準」。
    _pk, _rd, _bt = [], [], []
    for _gi, _ix in GRP_IDX.items():
        _ps = np.unique(pairs[_ix].reshape(-1))
        if len(_ps) < 2:
            continue
        _Lg = lsort[_ps].astype(np.float64)
        _vg = sc[np.array([remap[int(p)] for p in _ps])]
        _pk.append(_Lg[int(np.argmax(_vg))]); _rd.append(_Lg.mean()); _bt.append(_Lg.min())
    _pk, _rd, _bt = np.array(_pk), np.array(_rd), np.array(_bt)
    _saved, _gap = float((_rd - _pk).mean()), float((_pk - _bt).mean())
    _span = float((_rd - _bt).mean())
    row["steps"] = {"pick": float(_pk.mean()), "random": float(_rd.mean()), "best": float(_bt.mean()),
                    "saved": _saved, "gap": _gap, "n_groups": int(len(_pk)),
                    "captured": (_saved / _span) if _span > 0 else None}
    for gp in GAPS:
        acc, n, ci = rank_acc(sa, sb_, gp, boot=(300 if gp == 0 else 0))
        row["rank_acc"][str(gp)] = {"acc": acc, "n": n, "cluster_ci95": ci}
        cis = f"  cluster-CI95 [{ci[0]:.3f},{ci[1]:.3f}]" if ci else ""
        print(f"  同題內排序準確率  |ΔL|≥{max(gp,1):<3} : {acc:.3f}   (n={n}){cis}", flush=True)
    print(f"  ⭐ 拿 V 選路：隨機挑 {_rd.mean():.1f} 步 → V 挑 {_pk.mean():.1f} 步 "
          f"→ 最好 {_bt.mean():.1f} 步   ⇒ 省了 {_saved:.1f} 步、還差 {_gap:.1f} 步"
          f"（吃掉可省空間的 {100*_saved/_span:.0f}%）", flush=True)
    print(f"  held-out MSE {heldout_mse:.1f}"
          f"{'  ⚠️ w_mse=0 ⇒ 尺度會漂，這欄不可比' if W_MSE <= 0 else ''}"
          f"   train集同題排序 {tr_rank:.3f}   train集辨識別題的路 {tr_neg:.3f}"
          f"（診斷用，⛔ 不是主指標）", flush=True)
    print(f"  全體 Spearman {row['spearman_all']:+.3f} "
          f"⚠️ 附帶指標（跨題排序光靠直線距離就能拿高分）", flush=True)
    report["runs"][name] = row

# ⭐ 探針（2026-08-26 主人核可）：u 到底編碼了什麼？
#    推論要驗的是：encoder 的目標只要求「對得上 s,g」，而 s,g 只有 4 個數字
#    ⇒ u 只要編碼那 4 個數字就交差了 ⇒ 🚨 它沒有理由記住長度。
#    ⇒ 若 (s,g) 的 R² 高、L 的 R² 低 ⇒ 推論成立。
_L_uniq = lsort[uniq].astype(np.float64)
_SG_uniq = sg_of(rsort[uniq], gsort[uniq], VOBS).astype(np.float64)
print("\n=== 探針：u 裡面有什麼（線性、held-out R²）===", flush=True)
_probe = {}
for _nm, _e in (("訓練過的 encoder", trained_enc), ("沒訓練的 encoder", random_enc)):
    _pu = pure_u(_e, uniq)
    _r_sg, _r_l = lin_r2(_pu, _SG_uniq), lin_r2(_pu, _L_uniq)
    _probe[_nm] = {"r2_sg": _r_sg, "r2_L": _r_l}
    print(f"  {_nm:>16}   (s,g) R² {_r_sg:+.3f}    路長 L 的 R² {_r_l:+.3f}", flush=True)
# ⚠️ 基準：直接拿原始座標當特徵，長度的 R² 應該很高 ⇒ 證明「資訊在資料裡」不是在 u 裡
_raw_feat = make_segments(rsort[uniq], gsort[uniq], VOBS)[0].reshape(len(uniq), -1).astype(np.float64)
_probe["原始座標"] = {"r2_sg": lin_r2(_raw_feat, _SG_uniq), "r2_L": lin_r2(_raw_feat, _L_uniq)}
print(f"  {'原始座標（基準）':>16}   (s,g) R² {_probe['原始座標']['r2_sg']:+.3f}"
      f"    路長 L 的 R² {_probe['原始座標']['r2_L']:+.3f}", flush=True)
report["probe_linear_r2"] = _probe

# 🚨 對照⑤（2026-08-26 加）：【取樣模式本身】能洩漏多少長度。
#    量法：把 make_segments 真的吐出來的點，數「不重複的座標有幾個」，拿它當分數。
#    ⇒ 這是【迴歸防護】：不管以後取樣方式怎麼改，這條都會自己叫。
#    ⛔ 別用公式 min(L+1,T) 代替 —— 那只對取整版成立，換個取樣法就失效了。
_lk_seg, _ = make_segments(rsort[uniq], gsort[uniq], VOBS)
_nuniq = np.array([len(np.unique(s, axis=0)) for s in _lk_seg], np.float64)
_lk_score = -_nuniq + np.random.default_rng(7).standard_normal(len(_nuniq)) * 1e-6  # tie 隨機拆
acc_lk, n_lk, ci_lk = rank_acc(_lk_score[ia], _lk_score[ib], 0, boot=300)
report["control_sampling_leak"] = {"acc": acc_lk, "n": n_lk, "cluster_ci95": ci_lk,
                                   "n_unique_median": float(np.median(_nuniq)), "T_FIX": T_FIX}
print(f"\n=== 對照⑤  取樣模式洩漏（數不重複的點）===\n  {acc_lk:.3f}  (n={n_lk})"
      f"  cluster-CI95 [{ci_lk[0]:.3f},{ci_lk[1]:.3f}]"
      f"   不重複點數中位 {np.median(_nuniq):.0f} / {T_FIX}", flush=True)

# ⭐ 尺②（2026-08-26 加）：完全不用訓練 —— 直接把重採樣後【相鄰點的距離】加起來當分數。
#    ⇒ 它回答的正是主人問的那格：「是 u 裡沒有，還是題目本身難？」
#    ⇒ 高 ⇒ 資訊確實在幾何裡，是我們沒讀出來；低 ⇒ 重採樣成 T_FIX 點就已經把長度削掉了。
#    ⚠️ 必須用【未正規化】的座標：(obs-mu)/sd 會把 x/y 各自縮放 ⇒ 距離被扭曲。
_gi = np.linspace(rsort[uniq][:, None], gsort[uniq][:, None], T_FIX,
                  axis=1).round().astype(np.int64).reshape(len(uniq), T_FIX)
_geo = -np.linalg.norm(np.diff(VOBS[_gi], axis=1), axis=2).sum(1)     # 負的幾何長度（短 ⇒ 分高）
acc_g, n_g, ci_g = rank_acc(_geo[ia], _geo[ib], 0, boot=300)
report["ruler_geometric"] = {"acc": acc_g, "n": n_g, "cluster_ci95": ci_g}
print(f"\n=== 尺②  幾何長度（不用學，直接算）===\n  {acc_g:.3f}  (n={n_g})"
      f"  cluster-CI95 [{ci_g[0]:.3f},{ci_g[1]:.3f}]", flush=True)

acc_d, n_d, ci_d = rank_acc(-dist_a, -dist_b, 0, boot=300)
report["baseline_straight_line"] = {"acc": acc_d, "n": n_d, "cluster_ci95": ci_d}
print(f"\n=== 對照③  只用 s,g 直線距離 ===\n  {acc_d:.3f}  (n={n_d})"
      f"  cluster-CI95 [{ci_d[0]:.3f},{ci_d[1]:.3f}]", flush=True)

# ---------- verdict ----------
real = report["runs"]["real"]["rank_acc"]["0"]["acc"]
real_ci = report["runs"]["real"]["rank_acc"]["0"]["cluster_ci95"]
ok_main = real_ci is not None and real_ci[0] > 0.8            # ⚠️ 用 CI 下界，⛔ 不是點估計
notes = []
# ⭐ 2026-08-26 加的第二道：光是贏過 0.5 不算數，要贏過【V 完全不訓練】那條線。
beats_ut = None
if "untrained_V" in report["runs"]:
    _utci = report["runs"]["untrained_V"]["rank_acc"]["0"]["cluster_ci95"]
    _utac = report["runs"]["untrained_V"]["rank_acc"]["0"]["acc"]
    beats_ut = bool(real_ci and _utci and real_ci[0] > _utci[1])
    print(f"\n=== 對照④  V 完全不訓練 ===\n  {_utac:.3f}"
          f"{f'  CI95 [{_utci[0]:.3f},{_utci[1]:.3f}]' if _utci else ''}"
          f"   ← 主指標真正要贏的線", flush=True)
    if not beats_ut:
        notes.append(f"🚨 V(u) 的 CI 下界 {real_ci[0]:.3f} 沒有超過「V 不訓練」的 CI 上界 "
                     f"{_utci[1]:.3f} ⇒ 訓練沒有帶來可辨識的增益")
if not SHUFFLE_CTRL:
    notes.append("⛔ SHUFFLE_CTRL=0 ⇒ 對照組沒跑完 ⇒ 主指標不可信")
    ok_ctrl = False
else:
    sh = report["runs"]["shuffled"]["rank_acc"]["0"]["acc"]
    rd = report["runs"]["random_enc"]["rank_acc"]["0"]["acc"]
    ut = report["runs"].get("untrained_V", {}).get("rank_acc", {}).get("0", {}).get("acc")
    # ⚠️ 容忍度用實測的 cluster-CI 寬度，⛔ 不是拍腦袋的 ±0.05
    tol = max(0.03, (ci_d[1] - ci_d[0]) / 2) if ci_d else 0.05
    # 🚨 2026-08-26：三個對照的【正確基線不一樣】，⛔ 全部拿 0.5 去比是錯的。
    #    隨機 encoder / 直線距離 ⇒ 基線 0.5（它們的輸入本來就沒有長度結構）
    #    標籤打亂             ⇒ 基線是「V 完全不訓練」—— 因為打亂消不掉「u 的幾何本身白送」
    #                            （實測：打亂標籤拿 0.536，CI 下界 0.516 > 0.5 ⇒ 拿 0.5 當基線會誤判）
    checks = {"隨機 encoder": (rd, 0.5), "直線距離": (acc_d, 0.5),
              "標籤打亂": (sh, ut if ut is not None else 0.5)}
    ok_ctrl = True
    # ⚠️ 「標籤打亂」只在【高於】基線時才是問題：打亂本來就該傷害 ⇒ 低於基線是正常的。
    #    ⛔ 8/26 舊版寫成雙邊 ⇒ T_FIX=128/201 被誤判成「對照組不正常」。
    ONE_SIDED = {"標籤打亂"}
    for k, (v, base) in checks.items():
        bad = (v - base > tol) if k in ONE_SIDED else (abs(v - base) > tol)
        if bad:
            ok_ctrl = False
            notes.append(f"🚨 對照「{k}」= {v:.3f}，離它的基線 {base:.3f} 超過 {tol:.3f}"
                         f" ⇒ 還有洩漏或管線有問題")
    # 🚨 取樣洩漏：這條一響，主指標不管印什麼都不算數
    if acc_lk > 0.5 + max(tol, 0.05):
        ok_ctrl = False
        notes.append(f"🚨 對照⑤ 取樣模式本身就能拿 {acc_lk:.3f} ⇒ 長度從【點的重複】洩漏出去了"
                     f"（不重複點中位 {np.median(_nuniq):.0f}/{T_FIX}）⇒ 主指標不算數")
# ⭐ 那把尺：主指標離上限多遠，決定處方是「修 encoder」還是「題目本身就難」
if "oracle" in report["runs"]:
    orc = report["runs"]["oracle"]["rank_acc"]["0"]["acc"]
    orc_ci = report["runs"]["oracle"]["rank_acc"]["0"]["cluster_ci95"]
    report["oracle_gap"] = {"oracle_acc": orc, "real_acc": real, "gap": orc - real}
    print(f"\n=== 尺 vs V(u) ===", flush=True)
    print(f"  尺（原始座標） {orc:.3f}"
          f"{f'  CI95 [{orc_ci[0]:.3f},{orc_ci[1]:.3f}]' if orc_ci else ''}", flush=True)
    print(f"  V(u)           {real:.3f}   ⇒ 差 {orc - real:+.3f}", flush=True)
    # 🚨 2026-08-26：舊版這裡印的是一句【固定字串】，不管數字往哪邊都照印
    #    ⇒ 那是「檢查會通過因為它壞了」的同款形狀。改成由數值決定。
    _g = orc - real
    _msg = ("⇒ 尺① 明顯高於 V(u) ⇒ 資訊掉在 encoder" if _g > 0.05 else
            "⇒ 🚨 V(u) 反而高於尺① ⇒ 這把尺量到的是【架構差】不是資訊差，⛔ 別當上限用" if _g < -0.05 else
            "⇒ 兩者差不多 ⇒ 尺① 分不出東西，改看尺②（幾何長度）")
    print("  " + _msg, flush=True)
report["verdict"] = {"main_pass": bool(ok_main), "controls_sane": bool(ok_ctrl),
                     "beats_untrained_V": beats_ut, "notes": notes}
print(f"\n=> 主指標 {'PASS' if ok_main else 'FAIL'}"
      f"（同題內排序 {real:.3f}，CI 下界要 > 0.8）", flush=True)
print(f"=> 對照組 {'正常' if ok_ctrl else '🚨 不正常'}", flush=True)
if beats_ut is not None:
    print(f"=> 贏過「V 不訓練」 {'✓' if beats_ut else '🚨 沒有'}", flush=True)
for nte in notes:
    print("   " + nte, flush=True)
if not ok_ctrl:
    print("   ⛔ 對照組沒過 ⇒ 主指標不管印什麼都不算數", flush=True)

os.makedirs(OUT_DIR, exist_ok=True)
out = os.path.join(OUT_DIR,
    f"value_u_{ENV_NAME}_T{T_FIX}_K{K}_e{EPS}_p{EVAL_PAIRS}_s1{STEPS1}_sv{STEPS_V}"
    f"_wm{W_MSE:g}_wr{W_RANK:g}_wn{W_NEG:g}_sg{int(USE_SG)}"
    f"_o{int(ORACLE)}_{SAMPLE}_q{int(SG_QUERY)}_s{SEED}.json")
with open(out, "w") as f:
    json.dump(report, f, indent=2, allow_nan=False)          # ⚠️ 裸 NaN 會讓 jq / JS 解析失敗
print(f"存到 {out}", flush=True)
