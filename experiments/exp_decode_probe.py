"""u 裡到底裝了多少路線資訊 —— 直接量，不再用探針推。

🚨 主人 2026-08-28 核可：「這禮拜已經測無數次了，最後一次，測完不要忘記」
⇒ ⛔ 這是最後一次量這件事。跑完必須寫進實驗索引，⛔ 不准有第五次。

## 為什麼要跑（⛔ 別把這段刪掉，這是它存在的理由）

換 encoder 目標那一刀，整個建立在「u 裡沒有路線資訊」上。
而這個前提四天來都是【間接】推出來的：
  ① 線性探針 r2_L 低      ⇒ 🚨 這條 2026-08-28 已作廢：
     [實測] 原始座標（資訊 100% 在場）的 linear r2_L = -0.0412
     ⇒ 路長是座標的非線性函數，線性探針讀不到是【預期行為】，不是發現
  ② 有效維度 ~6           ⇒ 隔一層（量變異不量資訊），而且分母是 batch 64 不是 1024
  ③ flow 塌成點質量        ⇒ 隔一層（隔著 flow 的訓練）
  ④ V(u) 這個 MLP 也讀不出 ⇒ 隔一層（隔著 V 的訓練病）

⇒ ⭐ 這支是唯一【直接】的：訓一顆 decoder 到收斂，量的就是「u 裡拿得出多少」本身。

## 判準（⛔ 開跑前寫死，事後不准改）

單看一個重建誤差沒有意義（0.3 是好是壞？不知道）⇒ 要兩個對照把它夾起來：

    D_sg   只餵 (s,g) 4 個數      → 誤差 A ＝ 不看 u 就猜得到的部分（下界）
    D_u    餵【凍住的】舊 u        → 誤差 B ＝ 受測者
    D_ae   encoder 重新學（自編碼）→ 誤差 C ＝ 這個架構的上限

    資訊回收率 = (A − B) / (A − C)

      < 25%      u ≈ f(s,g)，資訊真的不在 ⇒ 換考題那刀成立
      25% ~ 50%  灰帶 ⇒ 主力矩陣要加一支「舊 encoder + decoder」對照
      > 50%      🚨 前提錯 ⇒ 換考題在解假問題，瓶頸在 flow / head

⭐ Fable 5 的預註冊預測（2026-08-28，開跑前寫死）：回收率 < 25%。
⭐ ルナ沒有獨立預測 —— ①作廢之後我對這格【沒有】立場，這正是要跑它的原因。

## K 掃（主人 2026-08-28 指示「query 不一定要 4，可以掃」）

舊的 K sweep 全部是在【舊考題】下量的（K=1/2/4/16 → r2_L 0.318/0.203/0.145，K16 炸掉）
⇒ 舊考題只要求認出 (s,g)，多的 token 沒有梯度壓力去裝東西 ⇒ 那個結論綁在舊考題上。
⇒ ⭐ 在【重建考題】下掃 K，量的才是「一條路需要多少容量」。
⚠️ 只有 D_ae 掃得動：D_u 的 K 被 ckpt 綁死。

用法：
    LACOT_ENV=pointmaze-medium-stitch-v0 LACOT_TCAP=128 \
    LACOT_DP_CKPT=results/ckpt_medium-stitch_..._s0.pt \
    LACOT_DP_MODE=u|sg|ae LACOT_K=4 python -u experiments/exp_decode_probe.py
"""
import os, sys, json, time
import numpy as np
import torch
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lacot.e_target import PerceiverPooler
from lacot.traj_decoder import TrajDecoder

OGB_DATA = os.environ.get("OGBENCH_DATA_DIR", "/archive/cymaxwelllee/data/ogbench")
device = "cuda" if torch.cuda.is_available() else "cpu"
ENV_NAME = os.environ.get("LACOT_ENV", "pointmaze-medium-stitch-v0")
MODE = os.environ.get("LACOT_DP_MODE", "u")            # u | sg | ae
CKPT = os.environ.get("LACOT_DP_CKPT", "")
SEED = int(os.environ.get("LACOT_SEED", 0))
STEPS = int(os.environ.get("LACOT_DP_STEPS", 3000))
HOLDOUT_MOD = int(os.environ.get("LACOT_DP_HOLDOUT", 20))   # traj_id % 20 == 0 ⇒ held-out
CHECK_P = float(os.environ.get("LACOT_DP_CHECKP", 0.01))   # 塌掉檢查的抽查機率
# ⭐ corruption control 專用：故意把 pos_q 初始化成 encoder 那邊的 0.02，
#    用來驗「塌掉檢查」真的會叫。⛔ 正式跑一律 0 —— 檔名不帶它，因為它不該出現在結果裡。
BREAK_POSQ = int(os.environ.get("LACOT_DP_BREAK", 0))
# ⭐ 存 decoder：V_geo 需要「u → 128 個座標點」這條可微的路，而那顆就是這裡訓出來的。
#    ⛔ 預設不存（七支掃描存七顆沒意義），要用的那一顆明確指定。
SAVE_DEC = int(os.environ.get("LACOT_DP_SAVE", 0))
# 🚨 smoke 用假資料跑出來的檔【不准】落進 results/ —— 2026-08-28 我自己犯過一次。
#    同族檔案混版本正是這個 repo 咬過我們三次的病（Fable 引錯架構、我引錯架構、
#    8/27 引到舊 code 版本的產物）⇒ smoke 一律 LACOT_DP_OUT=<scratchpad>。
OUT_DIR = os.environ.get("LACOT_DP_OUT", "results")
assert MODE in ("u", "sg", "ae"), f"⛔ LACOT_DP_MODE 只能是 u/sg/ae，收到 {MODE}"
if MODE in ("u",):
    assert CKPT, "⛔ mode=u 要餵凍住的 encoder ⇒ 必須給 LACOT_DP_CKPT"

torch.manual_seed(SEED); np.random.seed(SEED)
print(f"device={device}  env={ENV_NAME}  mode={MODE}  seed={SEED}", flush=True)

# ─────────────────────────────────────────────────────────────────────
# 資料 ＋ make_batch
# 🚨 整段【複製自】scratch_lacot_rollout.py:30-121（2026-08-28 當日版本），⛔ 不是重寫。
#    理由：那一份帶著 F6/F7 兩個修好的洩漏（goal 重抽偏差、取樣點數洩漏長度），
#    重寫一份等於重新製造一次同樣的洩漏。⇒ 之後該抽進 lacot/data.py 共用（待辦）。
# ─────────────────────────────────────────────────────────────────────
d = np.load(f"{OGB_DATA}/{ENV_NAME}.npz")
OBS = np.asarray(d["observations"], np.float32)
ACT = np.asarray(d["actions"], np.float32)
TERM = np.asarray(d["terminals"], bool)
N = OBS.shape[0]
ends = np.flatnonzero(TERM); starts = np.concatenate([[0], ends[:-1] + 1])
traj_end = np.empty(N, np.int64)
traj_id = np.empty(N, np.int64)
for i, (s0, e0) in enumerate(zip(starts, ends)):
    traj_end[s0:e0 + 1] = e0
    traj_id[s0:e0 + 1] = i
assert ends[-1] == N - 1, "⛔ 資料集最後一筆不是 terminal ⇒ traj_end 尾巴是未初始化記憶體"
MAX_TRAIN_T = int((ends - starts + 1).max())
mu, sd = OBS.mean(0), OBS.std(0) + 1e-6

K = int(os.environ.get("LACOT_K", 4))
CHUNK = int(os.environ.get("LACOT_CHUNK", 4))
# 🚨 2026-08-28：預設 256 → 128，跟主線 scratch_lacot_rollout.py 對齊。
#    ⚠️ ⛔ 必須跟主線同步 —— T_CAP 是 decoder 的輸出點數：兩份 T_CAP 不同會讓
#    pos_q 的長度不同、同一條路上每個點的間距也不同 ⇒ 這裡訓出來的 decoder 
#    配到主線上就是【另一個東西】，而它不會報錯（形狀對不上時才會，同名不同義時不會）。
#    ⇒ 改主線的 LACOT_TCAP 預設時，這一行要一起改。
T_CAP = min(int(os.environ.get("LACOT_TCAP", 128)), MAX_TRAIN_T)
B, D_MODEL = 64, 256
# ⭐ held-out：整條軌跡切開，⛔ 不是切 row —— 切 row 的話同一條路會同時出現在兩邊。
IS_HELD = (traj_id % HOLDOUT_MOD) == 0
print(f"T_CAP={T_CAP}  K={K}  held-out 軌跡 {IS_HELD.sum()/N:.1%} 的 row", flush=True)


def make_batch(rng, held=False, bs=None):
    """⛔ 抽樣邏輯與主線逐字相同，只多了 held-out 過濾。"""
    bs = bs or B
    rows, goals = [], []
    while len(rows) < bs:
        r = int(rng.integers(0, N)); te = int(traj_end[r])
        if te - r < CHUNK:
            continue
        if IS_HELD[r] != held:            # ⭐ 唯一的新增：訓練/驗證各取各的軌跡
            continue
        _d = rng.random()
        gr = int(round(min(r + 1, te) * _d + te * (1 - _d)))
        gr = max(gr, min(r + CHUNK, te))
        rows.append(r); goals.append(gr)
    rows, goals = np.array(rows), np.array(goals)
    f = np.linspace(rows[:, None].astype(np.float64), goals[:, None].astype(np.float64),
                    T_CAP, axis=1).reshape(bs, T_CAP)
    lo_i = np.floor(f).astype(np.int64)
    hi_i = np.minimum(lo_i + 1, goals[:, None])
    w = (f - lo_i)[..., None]
    traj = ((OBS[lo_i] * (1.0 - w) + OBS[hi_i] * w - mu) / sd).astype(np.float32)
    mask = np.zeros((bs, T_CAP), bool)
    assert mask.shape[1] == T_CAP and not mask.any(), "⛔ 取樣點數不再固定 ⇒ 長度會從 mask 洩漏"
    s = (OBS[rows] - mu) / sd; g = (OBS[goals] - mu) / sd
    T = lambda x: torch.from_numpy(x.astype(np.float32)).to(device)
    return T(traj), torch.from_numpy(mask).to(device), T(s), T(g)


def sota_mlp(i, h, o, n=2):
    """⛔ 逐字同主線 :123-129 —— encoder 要跟 ckpt 對得上，形狀差一層就載不進來。"""
    L, p = [], i
    for _ in range(n):
        lin = nn.Linear(p, h); nn.init.xavier_uniform_(lin.weight); nn.init.zeros_(lin.bias)
        L += [lin, nn.GELU(), nn.LayerNorm(h)]; p = h
    lin = nn.Linear(p, o); nn.init.xavier_uniform_(lin.weight); nn.init.zeros_(lin.bias)
    return nn.Sequential(*L, lin)


# ─────────────────────────────────────────────────────────────────────
# TrajDecoder —— PerceiverPooler 的鏡像：M 個 token → T_CAP 個座標點
# ─────────────────────────────────────────────────────────────────────
# ⭐ TrajDecoder 已抽進 lacot/traj_decoder.py（2026-08-28）——
#    主線的 recon 目標與 V_geo 要用同一顆，⛔ 不能留兩份分岔。
#    搬過去的是逐字同一份（本檔 2026-08-28 的結果就是用它跑的），
#    連 corruption control 的結論註解一起帶走了。

# ─────────────────────────────────────────────────────────────────────
# 三種 context 來源
# ─────────────────────────────────────────────────────────────────────
def build_context_fn():
    """回傳 (fn(traj, mask, s, g) -> [B, M, D_MODEL], 可訓練參數 list, 說明, 模組 dict)。

    🚨 2026-08-28 修：舊版存 ckpt 時是靠 `ctx_fn.__closure__[1] / [0]` 去拿
       traj_enc / e_pooler —— freevar 的順序是【按名字排的】，⛔ 不是按出現順序。
       ⇒ 把變數改名、或在 fn 裡多引用一個外層變數，兩顆就會【對調】存進 ckpt，
         而它不會報錯：兩顆的 state_dict 形狀不同才會炸，這裡是名字對、內容錯
         ⇒ 之後拿這顆 decoder 去解碼會安靜地吐垃圾。
    ⇒ 明確回傳模組 dict，⛔ 不靠 closure 順序。
    """
    if MODE == "sg":
        # 下界：完全不看軌跡，只有起終點各一個 token
        emb = sota_mlp(2, 512, D_MODEL).to(device)
        return (lambda traj, mask, s, g: torch.stack([emb(s), emb(g)], 1),
                list(emb.parameters()), "只有 (s,g) 兩個 token", {"emb": emb})

    traj_enc = sota_mlp(2, 512, 512).to(device)
    e_pooler = PerceiverPooler(512, D_MODEL, K, 2, 4, max_len=max(512, T_CAP)).to(device)

    if MODE == "u":
        sd_ = torch.load(CKPT, map_location=device, weights_only=False)
        traj_enc.load_state_dict(sd_["traj_enc"]); e_pooler.load_state_dict(sd_["e_pooler"])
        ck = sd_.get("cfg", {})
        ck_k = int(ck["K"]) if "K" in ck else None
        assert ck_k in (None, K), f"⛔ ckpt 的 K={ck_k} 跟本次 K={K} 不一致 ⇒ 載進來的不是同一個東西"
        for p in list(traj_enc.parameters()) + list(e_pooler.parameters()):
            p.requires_grad_(False)
        traj_enc.eval(); e_pooler.eval()
        print(f"✅ 載入凍住的 encoder：{os.path.basename(CKPT)}", flush=True)

        def fn(traj, mask, s, g):
            with torch.no_grad():
                Bc, Tc, _ = traj.shape
                return e_pooler(traj_enc(traj.reshape(Bc * Tc, 2)).reshape(Bc, Tc, 512),
                                key_padding_mask=mask)
        return fn, [], "凍住的舊 u", {"traj_enc": traj_enc, "e_pooler": e_pooler}

    # ae：encoder 跟著一起學 ⇒ 這個架構的上限
    def fn(traj, mask, s, g):
        Bc, Tc, _ = traj.shape
        return e_pooler(traj_enc(traj.reshape(Bc * Tc, 2)).reshape(Bc, Tc, 512),
                        key_padding_mask=mask)
    return (fn, list(traj_enc.parameters()) + list(e_pooler.parameters()),
            "自編碼（encoder 可訓）", {"traj_enc": traj_enc, "e_pooler": e_pooler})


ctx_fn, enc_params, desc, ctx_mods = build_context_fn()
dec = TrajDecoder(D_MODEL, T_CAP, pos_std=0.02 if BREAK_POSQ else 1.0).to(device)
dec.check_p = CHECK_P
if BREAK_POSQ:
    print("🚨 BREAK_POSQ=1：pos_q 故意初始化成 0.02（corruption control）", flush=True)
params = list(dec.parameters()) + enc_params
opt = torch.optim.Adam(params, lr=3e-4)
n_par = sum(p.numel() for p in params)
print(f"context = {desc}   可訓練參數 {n_par/1e6:.2f}M", flush=True)


def rmse_parts(pred, tgt):
    """回傳 (整條 RMSE, 內部點 RMSE)。
    ⭐ 端點要分開看：起終點的資訊 u 一定有（cond 就帶著），內部點才是「路線」。"""
    err = (pred - tgt).pow(2).sum(-1)                 # [B, T] 每點的平方距離
    inner = err[:, 1:-1]
    return float(err.mean().sqrt()), float(inner.mean().sqrt())


rng = np.random.default_rng(SEED)
val_rng = np.random.default_rng(10_000 + SEED)
t0 = time.time()
hist = []
for stp in range(STEPS):
    traj, mask, s, g = make_batch(rng, held=False)
    pred = dec(ctx_fn(traj, mask, s, g))
    loss = (pred - traj).pow(2).mean()
    opt.zero_grad(set_to_none=True); loss.backward()
    torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
    if (stp + 1) % max(STEPS // 6, 1) == 0 or stp == 0:
        dec.eval()
        with torch.no_grad():
            vt, vm, vs, vg = make_batch(val_rng, held=True, bs=256)
            vp = dec(ctx_fn(vt, vm, vs, vg))
            all_r, in_r = rmse_parts(vp, vt)
        dec.train()
        hist.append(dict(step=stp + 1, train_mse=float(loss), val_rmse=all_r, val_inner_rmse=in_r))
        print(f"  step {stp+1:>5}  train {float(loss):.4f}   held-out RMSE {all_r:.4f}"
              f"  內部點 {in_r:.4f}   [{time.time()-t0:.0f}s]", flush=True)

# 最終評估：多抽幾批取平均，⛔ 一批 256 題不夠穩
# ⭐ 同時跑 shuffle 對照 —— 把 context 沿 batch 打亂再解碼一次。
#    decoder 若真的在讀 context，打亂後誤差必須明顯變大；
#    ⛔ 若打亂前後一樣，代表它學到的是「不管給什麼都吐同一條平均路」
#    ⇒ 那時的重建誤差【不能】拿來講「u 裡有多少資訊」，它量的是資料的邊際分布。
dec.eval()
alls, inners, shuf_inners, batch_spreads = [], [], [], []
with torch.no_grad():
    for _ in range(8):
        vt, vm, vs, vg = make_batch(val_rng, held=True, bs=256)
        ctx = ctx_fn(vt, vm, vs, vg)
        pred = dec(ctx)
        a, i = rmse_parts(pred, vt)
        alls.append(a); inners.append(i)
        perm = torch.randperm(len(ctx), device=ctx.device)
        _, si = rmse_parts(dec(ctx[perm]), vt)
        shuf_inners.append(si)
        batch_spreads.append(float(pred.std(dim=0).mean()))   # 不同樣本之間的差異
res = dict(env=ENV_NAME, mode=MODE, desc=desc, seed=SEED, K=K, T_CAP=T_CAP, steps=STEPS,
           ckpt=os.path.basename(CKPT) if CKPT else None,
           holdout_mod=HOLDOUT_MOD, n_params=n_par, secs=round(time.time() - t0, 1),
           val_rmse=float(np.mean(alls)), val_rmse_sd=float(np.std(alls)),
           val_inner_rmse=float(np.mean(inners)), val_inner_rmse_sd=float(np.std(inners)),
           shuffled_inner_rmse=float(np.mean(shuf_inners)),
           ctx_usage=float(np.mean(shuf_inners) - np.mean(inners)),   # 打亂 context 變差多少
           batch_spread=float(np.mean(batch_spreads)),                # 不同樣本的輸出差異
           hist=hist)
os.makedirs(OUT_DIR, exist_ok=True)
tag = (f"{OUT_DIR}/decprobe_{ENV_NAME.replace('pointmaze-','').replace('-v0','')}"
       f"_{MODE}_K{K}_T{T_CAP}_st{STEPS}_s{SEED}.json")
with open(tag, "w") as f:
    json.dump(res, f, ensure_ascii=False, indent=1)
if SAVE_DEC:
    # ⚠️ encoder 也要一起存 —— mode=ae 的 decoder 只對【它自己訓出來的】那個 encoder 有意義，
    #    配錯 encoder 的 decoder 會安靜地解碼出垃圾（⛔ 這種錯不會報錯）。
    ck = dict(dec=dec.state_dict(), mode=MODE, K=K, T_CAP=T_CAP, seed=SEED,
              val_inner_rmse=res["val_inner_rmse"])
    if MODE == "ae":
        # ⛔ 按名字拿，⛔ 不按 closure 順序（見 build_context_fn 的 docstring）
        ck["traj_enc"] = ctx_mods["traj_enc"].state_dict()
        ck["e_pooler"] = ctx_mods["e_pooler"].state_dict()
    dpath = tag.replace(f"{OUT_DIR}/decprobe_", f"{OUT_DIR}/decoder_").replace(".json", ".pt")
    torch.save(ck, dpath)
    print(f"存 decoder {dpath}"
          f"（含 encoder）" if MODE == "ae" else f"存 decoder {dpath}", flush=True)
print(f"\n  {MODE}: held-out RMSE {res['val_rmse']:.4f} ± {res['val_rmse_sd']:.4f}"
      f"   內部點 {res['val_inner_rmse']:.4f} ± {res['val_inner_rmse_sd']:.4f}", flush=True)
print(f"  context 打亂後內部點 {res['shuffled_inner_rmse']:.4f}"
      f"  ⇒ 用到的 context 值 {res['ctx_usage']:+.4f}"
      f"   {'🚨 decoder 幾乎沒在讀 context ⇒ 這次的 RMSE 不能拿來談 u' if res['ctx_usage'] < 0.02 else '✓ 有在讀'}",
      flush=True)
print(f"  不同樣本之間的輸出差異 {res['batch_spread']:.4f}"
      f"   {'🚨 幾乎吐同一條路' if res['batch_spread'] < 0.05 else ''}", flush=True)
print(f"寫入 {tag}", flush=True)
