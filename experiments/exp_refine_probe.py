"""§4 機械驗收探針：在跑任何 rollout 之前，先驗梯度爬坡在【真模型】上到底做了什麼。
（handoff 2026-08-28 ⑨；Fable 方案 §4。⛔ rollout 一題一分多鐘，這裡純張量、不碰 env、幾分鐘。）

餵四種 u（B 條一批）：
  flow   flow.sample(cond)                    正常起點（主線實際用的）
  bad    flow.sample + NOISE·N(0,I)           故意弄壞（跟 smoke_refine_grad 同款）
  perm   別題的 u_true（batch 內打亂配對）      內容好、但跟 (s,g) 不符
  true   本題的 u_true                         已經是好答案 ⇒ 爬坡應該【幾乎不動它】
掃 η×λ 格點各爬 STEPS 步，＋三個對照：
  η=0    null control ── ⛔ 它必須【不過】驗收門檻，不然探針是裝飾
  λ=0    只有羅盤（V_geo）
  zeroV  只有結界（flow log p）
判（對 bad/perm 起點，per-sample 中位數）：
  wall、goal 要降到 anchor 的 p90 以內；log p 不准跌出 anchor p10 − 2×IQR。
  anchor ＝ decode(u_true) 的分布 ── ⭐ 跟被評物走同一個 decoder，才是「爬坡可達的好」；
  真軌跡本身的 wall 恆 0（佔據圖由同批 OBS 蓋成，見主線 sanity 註解）⇒ 只印參考、不當門檻。
兩種病的訊號：
  病 A（爬出 flow 認得的範圍）＝ wall/goal 改善但 logp 跌出下限 ⇒ λ 太小
  病 B（V_geo 的最佳解不在資料流形上）＝ true 起點被爬走很遠（‖Δdecode‖ 大、logp 掉）
輸出：results/refineprobe_{tag}.json ＋ console 摘要表（等寬，可直接貼給主人）。

用法（lady 上，A2 的 ckpt）：
  LACOT_LOAD_CKPT=results/ckpt_medium-stitch_..._eorecon_ictr_..._s0.pt \
  LACOT_ENV=pointmaze-medium-stitch-v0 python -u experiments/exp_refine_probe.py
本機管線 smoke（假資料＋隨機權重，只驗跑得通、⛔ 不驗判定）：
  LACOT_PROBE_SMOKE=1 python -u experiments/exp_refine_probe.py
"""
import json, os, sys
import numpy as np
import torch
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lacot.e_target import PerceiverPooler
from lacot.nf_head import Flow
from lacot.refine_grad import GeoValue, grad_refine
from lacot.traj_decoder import TrajDecoder

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
device = "cuda" if torch.cuda.is_available() else "cpu"
SMOKE = int(os.environ.get("LACOT_PROBE_SMOKE", 0))
SEED = int(os.environ.get("LACOT_SEED", 0))
B = int(os.environ.get("LACOT_PROBE_B", 8 if SMOKE else 64))
STEPS = int(os.environ.get("LACOT_PROBE_STEPS", 5 if SMOKE else 100))
NOISE = float(os.environ.get("LACOT_PROBE_NOISE", 2.0))
# ⭐ 分布外跨度模式（Q2 軸）：(s,g) 仍在同一條軌跡上（⇒ u_true anchor 造得出來），
#    但跨度強制 ≥ MINSPAN 步 —— 訓練抽樣的跨度 p99 只有 33 步，MINSPAN=100 就是分布外。
#    ⚠️ exp_span_difficulty 實測：這只是【時間】分布外 —— BFS 難度 p50 仍只有 1 格。
MINSPAN = int(os.environ.get("LACOT_PROBE_MINSPAN", 0))
# ⭐⭐ TASKS 模式（真正的 Q2 探針）：(s,g) 直接用 dev 尺的題（tier2 = BFS 7~11 格，
#    訓練抽法 p90 才 3 格）。這種配對資料裡沒有真軌跡 ⇒「true」起點換成
#    【BFS 參考路（格心折線沿弧長重採樣）的編碼】—— ⚠️ 那是 encoder 沒見過的合成輸入，
#    它的 decode 品質本身就是「拿 BFS 生 e_target」（主人 8/24 的問題）的直接證據。
#    先跑 experiments/exp_dump_dev_tasks.py 產 npz。
TASKS_NPZ = os.environ.get("LACOT_PROBE_TASKS", "")
PROBE_TIER = int(os.environ.get("LACOT_PROBE_TIER", 2))
ETAS = [float(x) for x in os.environ.get("LACOT_PROBE_ETAS", "0.05,0.1,0.5").split(",")]
LAMS = [float(x) for x in os.environ.get("LACOT_PROBE_LAMS", "0.1,0.3,1.0").split(",")]
torch.manual_seed(SEED)
rng = np.random.default_rng(SEED)
print(f"device={device} smoke={SMOKE} B={B} steps={STEPS} noise={NOISE} etas={ETAS} lams={LAMS}",
      flush=True)

# ── 資料 ─────────────────────────────────────────────────────────────────────
if SMOKE:
    # 假資料：L 形走廊上的隨機遊走（形狀對就好，⛔ 判定不 assert）
    ENV_NAME = "fake-smoke"
    _pts = []
    for _ in range(40):
        p = np.array([0.1, 0.1]) + rng.random(2) * 0.05
        seg = [p.copy()]
        for _ in range(int(rng.integers(30, 80))):
            p = p + rng.normal(0, 0.05, 2)
            seg.append(p.copy())
        _pts.append(np.array(seg))
    OBS = np.concatenate(_pts).astype(np.float32)
    TERM = np.zeros(len(OBS), bool)
    _off = np.cumsum([len(s) for s in _pts]) - 1
    TERM[_off] = True
else:
    OGB_DATA = os.environ.get("OGBENCH_DATA_DIR", "/archive/cymaxwelllee/data/ogbench")
    ENV_NAME = os.environ.get("LACOT_ENV", "pointmaze-medium-stitch-v0")
    d = np.load(f"{OGB_DATA}/{ENV_NAME}.npz")
    OBS = np.asarray(d["observations"], np.float32)
    TERM = np.asarray(d["terminals"], bool)
# ↓ 照抄 scratch_lacot_rollout.py（traj_end／mu/sd）
N = OBS.shape[0]; ends = np.flatnonzero(TERM); starts = np.concatenate([[0], ends[:-1] + 1])
traj_end = np.empty(N, np.int64)
for s0, e0 in zip(starts, ends):
    traj_end[s0:e0 + 1] = e0
assert ends[-1] == N - 1, "資料集最後一筆不是 terminal ⇒ traj_end 尾巴是未初始化的記憶體"
mu, sd = OBS.mean(0), OBS.std(0) + 1e-6

# ── ckpt 與 cfg（探針【以 ckpt 的 cfg 為準】建構 ⇒ 不可能不一致）──────────────
if SMOKE:
    cfg = dict(K=4, COND=64, T_CAP=32, CHUNK=4, D_MODEL=64, ENC_OBJ="recon_ictr")
    ck = None
else:
    LOAD_CKPT = os.environ.get("LACOT_LOAD_CKPT", "")
    assert LOAD_CKPT, "⛔ 探針要診斷【訓好的】模型：必須給 LACOT_LOAD_CKPT"
    _lp = LOAD_CKPT if os.path.isabs(LOAD_CKPT) else os.path.join(ROOT, LOAD_CKPT)
    ck = torch.load(_lp, map_location=device, weights_only=False)
    cfg = ck.get("cfg", {})
    assert str(cfg.get("ENC_OBJ", "")).startswith("recon") and "u_dec" in ck, (
        f"⛔ ckpt 的 ENC_OBJ={cfg.get('ENC_OBJ')!r} 沒有 decoder ⇒ V_geo 沒有眼睛，探針無意義")
K = int(cfg["K"]); COND = int(cfg["COND"]); T_CAP = int(cfg["T_CAP"])
D_MODEL = int(cfg.get("D_MODEL", 256)); DIM = K * D_MODEL
print(f"cfg: K={K} COND={COND} T_CAP={T_CAP} D_MODEL={D_MODEL}"
      f" ENC_OBJ={cfg.get('ENC_OBJ')} 來源={'(smoke)' if SMOKE else os.path.basename(_lp)}",
      flush=True)


# ── 模型（構造照抄 scratch_lacot_rollout.py；strict 載入本身就是結構一致的驗證）──
def sota_mlp(i, h, o, n=2):          # ↓ 照抄主線
    L, p = [], i
    for _ in range(n):
        lin = nn.Linear(p, h); nn.init.xavier_uniform_(lin.weight); nn.init.zeros_(lin.bias)
        L += [lin, nn.GELU(), nn.LayerNorm(h)]; p = h
    lin = nn.Linear(p, o); nn.init.xavier_uniform_(lin.weight); nn.init.zeros_(lin.bias)
    return nn.Sequential(*L, lin)


traj_enc = sota_mlp(2, 512, 512).to(device)
e_pooler = PerceiverPooler(512, D_MODEL, K, 2, 4, max_len=max(512, T_CAP)).to(device)
cond_enc = sota_mlp(2, 512, 512).to(device)
cond_head = sota_mlp(1024, 512, COND).to(device)
flow = Flow(token_dim=D_MODEL, seq_len=K, n_blocks=4, cond_dim=COND).to(device)
u_dec = TrajDecoder(D_MODEL, T_CAP).to(device)
if not SMOKE:
    for _name, _mod in (("traj_enc", traj_enc), ("e_pooler", e_pooler), ("cond_enc", cond_enc),
                        ("cond_head", cond_head), ("flow", flow), ("u_dec", u_dec)):
        _mod.load_state_dict(ck[_name])          # strict=True：形狀/鍵不合就當場炸
for _m in (traj_enc, e_pooler, cond_enc, cond_head, flow, u_dec):
    _m.eval()


def condvec(s, g):                   # ↓ 照抄主線
    return cond_head(torch.cat([cond_enc(s), cond_enc(g)], 1))


def encode(traj, mask):              # ↓ 照抄主線（e_pooler 那行）
    Bc, Tc = traj.shape[:2]
    return e_pooler(traj_enc(traj.reshape(Bc * Tc, 2)).reshape(Bc, Tc, 512),
                    key_padding_mask=mask)


# ── 批（插值段照抄 make_batch 的 F7 修法版。⚠️ 這是第四份副本【標明來源】——
#      抽 canonical 進 lacot/data.py 的債在 handoff 續帶區，⛔ 別讓第五份出現）──
def probe_batch(rng, B):
    rows, goals = [], []
    while len(rows) < B:
        r = int(rng.integers(0, N)); te = int(traj_end[r])
        if te - r < max(8, MINSPAN):     # 太短的段沒有「路線」可言（MINSPAN 模式：跨度不夠就換）
            continue
        if MINSPAN > 0:
            gr = int(rng.integers(r + MINSPAN, te + 1))   # 分布外：跨度 ≥ MINSPAN 步
        else:
            _d = rng.random()
            gr = int(round(min(r + 1, te) * _d + te * (1 - _d)))
            gr = max(gr, min(r + 8, te))
        rows.append(r); goals.append(gr)
    rows, goals = np.array(rows), np.array(goals)
    _sp = goals - rows
    print(f"  抽到的跨度：p50 {int(np.median(_sp))} p90 {int(np.percentile(_sp, 90))} 步"
          f"（訓練分布 p99 ≈ 33 步{'，MINSPAN 模式＝分布外' if MINSPAN else ''}）", flush=True)
    f = np.linspace(rows[:, None].astype(np.float64), goals[:, None].astype(np.float64),
                    T_CAP, axis=1).reshape(B, T_CAP)
    lo_i = np.floor(f).astype(np.int64)
    hi_i = np.minimum(lo_i + 1, goals[:, None])
    w = (f - lo_i)[..., None]
    traj = ((OBS[lo_i] * (1.0 - w) + OBS[hi_i] * w - mu) / sd).astype(np.float32)
    mask = np.zeros((B, T_CAP), bool)
    s = (OBS[rows] - mu) / sd; g = (OBS[goals] - mu) / sd
    T = lambda x: torch.from_numpy(x.astype(np.float32)).to(device)
    return T(traj), torch.from_numpy(mask).to(device), T(s), T(g)


class ZeroGeo:
    """「只有結界」對照：V≡0 但保持對 u 的計算圖（grad 恆 0，⛔ 不會炸 autograd）。"""

    def __call__(self, pts, s, g, per_term=False):
        v = pts.sum((1, 2)) * 0.0
        if per_term:
            z = v.detach()
            return v, dict(wall=z, goal=z, start=z, length=z)
        return v


GEO = GeoValue(OBS, mu, sd, res=8, device=device)
if not SMOKE:
    _gh = GEO.health()
    assert _gh["ok"], "⛔ 幾何 value 沒過健康檢查 ⇒ " + "；".join(_gh["reasons"])
    print(f"GEO health ✓  格心 round-trip {_gh['mapping_err']:.2e}"
          f"  盒內隨機點穿牆中位 {_gh['wall_median_random']:.4f}", flush=True)

def arc_resample(p, n):
    """[L,2] 折線沿弧長均勻取 n 點（跟主線 subgoal 的弧長原則同款）。"""
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    t = np.linspace(0.0, cum[-1], n)
    return np.stack([np.interp(t, cum, p[:, k]) for k in (0, 1)], 1)


# ── 四種起點 ─────────────────────────────────────────────────────────────────
if TASKS_NPZ:
    _tp = TASKS_NPZ if os.path.isabs(TASKS_NPZ) else os.path.join(ROOT, TASKS_NPZ)
    tz = np.load(_tp)
    _sel = np.flatnonzero(tz["tier"] == PROBE_TIER)
    B = len(_sel)
    print(f"TASKS 模式：tier{PROBE_TIER} {B} 題  BFS 距離"
          f" {int(tz['bfs_dist'][_sel].min())}~{int(tz['bfs_dist'][_sel].max())} 格"
          f"（⚠️「true」起點＝BFS 參考路的編碼，是 encoder 沒見過的合成輸入）", flush=True)
    _paths = np.stack([arc_resample(tz["path_xy"][i][: int(tz["path_len"][i])], T_CAP)
                       for i in _sel])
    traj = torch.from_numpy(((_paths - mu) / sd).astype(np.float32)).to(device)
    mask = torch.zeros(B, T_CAP, dtype=torch.bool, device=device)
    S = torch.from_numpy(((tz["init_xy"][_sel] - mu) / sd).astype(np.float32)).to(device)
    G = torch.from_numpy(((tz["goal_xy"][_sel] - mu) / sd).astype(np.float32)).to(device)
    with torch.no_grad():
        _bref = GEO.wall_depth(traj).mean(1)
    print(f"  BFS 參考路本身：穿牆中位 {float(_bref.median()):.4f}"
          f"（格心折線，理應貼 0 —— 大了就是 dump 或映射錯）", flush=True)
with torch.no_grad():
    if not TASKS_NPZ:
        traj, mask, S, G = probe_batch(rng, B)
    U_TRUE = encode(traj, mask)                      # 本題的 u_true（TASKS 模式＝BFS 參考路編碼）
    CONDV = condvec(S, G)
    U_FLOW = flow.sample(B, CONDV)
    U_BAD = U_FLOW + NOISE * torch.randn_like(U_FLOW)
    _perm = torch.from_numpy(rng.permutation(B)).to(device)
    assert (_perm != torch.arange(B, device=device)).float().mean() > 0.8
    U_PERM = U_TRUE[_perm]                           # 別題的 u_true（cond 不換）
SOURCES = {"flow": U_FLOW, "bad": U_BAD, "perm": U_PERM, "true": U_TRUE}


def measure(u):
    """per-sample：wall（沿路平均穿牆深度）、goal/start 距、log p、路長。"""
    with torch.no_grad():
        pts = u_dec(u)
        _, t = GEO(pts, S, G, per_term=True)
        lp = flow.log_prob(u, CONDV)
    return {k: v.detach().cpu().numpy() for k, v in
            dict(wall=t["wall"], goal=t["goal"], start=t["start"],
                 length=t["length"], logp=lp).items()}


# ── anchor：decode(u_true) 的分布（跟被評物同一個 decoder ⇒ 爬坡可達的「好」）──
A = measure(U_TRUE)
ANCH = dict(wall_p90=float(np.percentile(A["wall"], 90)),
            goal_p90=float(np.percentile(A["goal"], 90)),
            logp_p10=float(np.percentile(A["logp"], 10)),
            logp_iqr=float(np.percentile(A["logp"], 75) - np.percentile(A["logp"], 25)))
LOGP_FLOOR = ANCH["logp_p10"] - 2.0 * ANCH["logp_iqr"]
with torch.no_grad():
    _wd_data = GEO.wall_depth(traj).mean(1)
print(f"anchor（decode(u_true)）：wall p90 {ANCH['wall_p90']:.4f}  goal p90 {ANCH['goal_p90']:.4f}"
      f"  logp p10 {ANCH['logp_p10']:.1f}  IQR {ANCH['logp_iqr']:.1f} ⇒ 下限 {LOGP_FLOOR:.1f}\n"
      f"（參考：真軌跡本身 wall 中位 {float(_wd_data.median()):.4f} —— 恆 0、不當門檻）",
      flush=True)

# ── 掃描 ─────────────────────────────────────────────────────────────────────
CONFIGS = ([("null", 0.0, 0.3, GEO)] +                      # η=0（不動）
           [("compass", e, 0.0, GEO) for e in ETAS] +       # λ=0 只有羅盤
           [("barrier", e, 1.0, ZeroGeo()) for e in ETAS] + # V≡0 只有結界
           [("grid", e, l, GEO) for e in ETAS for l in LAMS])
ARMS = []
for src, u0 in SOURCES.items():
    for kind, eta, lam, geo in CONFIGS:
        if kind != "grid" and src not in ("bad", "true"):
            continue                                        # 對照組跑 bad/true 就夠了
        u1, hist = grad_refine(u0, CONDV, u_dec, flow, geo, S, G,
                               steps=STEPS, eta=eta, lam=lam, trace=True)
        m0, m1 = measure(u0), measure(u1)
        with torch.no_grad():
            dmove = float((u_dec(u1) - u_dec(u0)).norm(dim=-1).mean())
        fin = {k: float(np.median(m1[k])) for k in m1}
        passes = dict(wall=fin["wall"] <= ANCH["wall_p90"],
                      goal=fin["goal"] <= ANCH["goal_p90"],
                      logp=fin["logp"] >= LOGP_FLOOR)
        ARMS.append(dict(src=src, kind=kind, eta=eta, lam=lam,
                         start={k: float(np.median(m0[k])) for k in m0},
                         final=fin, dmove=dmove, hist=hist,
                         passes=passes, all_pass=all(passes.values())))
        print(f"  {src:5s} {kind:7s} η={eta:<4} λ={lam:<4} "
              f"wall {np.median(m0['wall']):.3f}→{fin['wall']:.3f} "
              f"goal {np.median(m0['goal']):.3f}→{fin['goal']:.3f} "
              f"logp {np.median(m0['logp']):.0f}→{fin['logp']:.0f} "
              f"‖Δdec‖ {dmove:.3f} {'✓' if ARMS[-1]['all_pass'] else '✗'}", flush=True)

# ── 判定 ─────────────────────────────────────────────────────────────────────
null_bad = next(a for a in ARMS if a["kind"] == "null" and a["src"] == "bad")
probe_valid = not null_bad["all_pass"]      # η=0 必須不過，不然 anchor 太鬆＝探針是裝飾
grid_bad = [a for a in ARMS if a["kind"] == "grid" and a["src"] in ("bad", "perm")]
by_el = {}
for a in grid_bad:
    by_el.setdefault((a["eta"], a["lam"]), []).append(a)
full = {k: v for k, v in by_el.items() if all(x["all_pass"] for x in v)}
recommend = (min(full, key=lambda k: np.mean([x["final"]["goal"] for x in full[k]]))
             if full else None)
true_grid = [a for a in ARMS if a["kind"] == "grid" and a["src"] == "true"]
true_stay = float(np.median([a["dmove"] for a in true_grid])) if true_grid else float("nan")
verdict = dict(
    probe_valid=bool(probe_valid), recommend=recommend and list(recommend),
    n_full_pass=len(full), true_dmove_med=true_stay,
    diseaseA="logp 跌出下限的格點數 = %d / %d" % (
        sum(1 for a in grid_bad if not a["passes"]["logp"]), len(grid_bad)),
    diseaseB_hint="true 起點被爬動的中位 ‖Δdecode‖ = %.3f（大 ⇒ V 的極值不在好答案上）" % true_stay)
print("\n=== 判定 ===", flush=True)
print(f"η=0 對照不過門檻（探針有效）：{'✓' if probe_valid else '✗ ⛔ anchor 太鬆，探針是裝飾'}",
      flush=True)
print(f"壞 u 全 pass 的 (η,λ)：{len(full)} 組 ⇒ 推薦 {recommend}", flush=True)
print(f"{verdict['diseaseA']}   {verdict['diseaseB_hint']}", flush=True)

tag = (f"{ENV_NAME}_st{STEPS}_B{B}_n{NOISE}"
       + (f"_span{MINSPAN}" if MINSPAN else "")
       + (f"_tasks-t{PROBE_TIER}" if TASKS_NPZ else "") + f"_s{SEED}"
       + ("" if SMOKE else "_" + os.path.basename(_lp).removeprefix("ckpt_").removesuffix(".pt")))
out = os.path.join(ROOT, "results", f"refineprobe_{tag}.json")
if SMOKE:
    import tempfile
    out = os.path.join(tempfile.gettempdir(), f"refineprobe_{tag}.json")  # ⛔ smoke 不碰 results/
json.dump(dict(meta=dict(env=ENV_NAME, steps=STEPS, B=B, noise=NOISE, seed=SEED,
                         minspan=MINSPAN, tasks=TASKS_NPZ or None, tier=PROBE_TIER if TASKS_NPZ else None,
                         etas=ETAS, lams=LAMS, smoke=SMOKE,
                         ckpt=None if SMOKE else os.path.basename(_lp), cfg=dict(cfg)),
               anchor=ANCH, logp_floor=LOGP_FLOOR, arms=ARMS, verdict=verdict),
          open(out, "w"), indent=1, default=float)
print(f"\n寫出 {out}", flush=True)
if SMOKE:
    print("SMOKE PASS（只驗管線；判定數字不作數）", flush=True)
