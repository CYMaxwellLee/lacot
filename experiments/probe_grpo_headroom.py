"""GRPO rung 0 探針：獎品量測（pass@G − pass@1）＋ C8 乘法閘校準。

出處＝docs/DESIGN-2026-09-06-grpo-thoughts.md（設計卡 v0）：
  §3.4「GRPO 買的東西＝把 pass@G 裡已存在但低機率的成功搬進 pass@1 ⇒ 獎品大小在
  訓練前就可量」；§4 rung 0（(i) reward 分佈與退化群比例 (ii) pass@G vs pass@1 gap
  (iii) C8 閘門檻校準 (iv) reward fn 單元測）；§2.1 reward 三項＋C8 閘的公式。

量什麼（每顆 ckpt）：
  1. reward 模組（GrpoReward，rung 1 直接重用）：
       r(z) = N(z) · ( w1·r_legal + w2·r_reach + w3·r_hit ),  w1=w2=w3=1/3（任務規格）
       r_legal = (1/T) Σ_t 1[free(p_t)]                    （decode 逐點自由格比例）
       r_reach = 1 − min( D(p_T)/D(s), 1 )                 （D=從 goal cell 的 grid_bfs；∞⇒0）
       r_hit   = 1[ D(p_T) ≤ 1 格 ]
       N(z)    = 1[arclen ≥ ρ_len·L_BFS(s,g)] · 1[max_t‖p_{t+1}−p_t‖ ≤ δ_step]（C8 乘法閘）
  2. headroom：N=64 個 (s,g)（同 probe 底座抽樣＝probe_branch_divergence.py:104-115
     逐字同款、同 seed ⇒ 跟該探針同一批題）、每題 G=16 條 z（帶 intent 的 eval cond、
     同 eval 語意）：mean r@1（單抽期望）、mean max r@G（最好那條）、headroom＝差；
     成功率版 pass@1 vs pass@G（r 換成命中二值）。
  3. C8 閘有效性：被 N(z) 閘掉的比例（拆 arclen/step/both）＋「短計畫刷分」被閘實例。

── C8 校準的表示法（⭐ 本探針的第一個實測發現，跑 smoke 就撞到）───────────────
  任務規格：門檻從【真軌跡分佈】校準：floor=真分佈 p5、step-cap=真分佈 p95。
  「真軌跡」有兩種表示法，數字差 3 倍：
    (a) raw 插值窗（T_CAP 點時間均勻內插）：max 鄰步距 p95≈0.033 —— 這是【插值 artifact】
        （128 點攤在時間軸上，步距天生極小），⛔ 不是 decoder 行為的量尺；
    (b) roundtrip decode（真軌跡 encode→u_dec decode）：max 鄰步距 p95≈0.098 ——
        這才是 reward 實際讀的那個空間（rung 1 的閘只會面對 decode 出來的計畫）。
    `[measured 2026-09-05 smoke]`：用 (a) 校準 δ_step ⇒ 100% 的 flow 樣本、連
    「真軌跡經 decoder」本身都被 step 閘咬掉 —— 閘連 ground truth 都不放行，尺壞了。
    flow 樣本的步距紋理與 roundtrip 相同（p95 0.0987 vs 0.0979）⇒ 步距是 decode
    表示法的性質，不是計畫好壞的性質。
  ⇒ ⭐ 本探針把「真軌跡分佈」讀在 decode 空間：閘門檻＝【roundtrip decode 的真分佈】
    p5 / p95（per-ckpt —— 各顆有自己的 encoder/decoder）。precedent＝
    probe_z_geodesic.py:308-310（真控制組＝roundtrip decode，⛔ 不是 raw 點，同一條理由）。
    raw 空間的 p5/p95 照樣印在報告當對照（校準發現本身是 rung 0 交付物 (iii)）。

── 預釘判準（⛔ 跑之前寫死，跑完照抄輸出，不事後挑）────────────────────────────
  ① 判準主尺＝【成功率版 headroom（raw hit）】＝ pass@G − pass@1，其中 hit 按 §2.1 的
     r_hit 原式（⛔ 不含 C8 閘 —— 任務規格「r 換成『命中』二值」的字面替換；也因此
     這把主尺【不依賴】上面的閘校準選擇）。gated-hit 版（hit·N(z)）並排報告當
     robustness 檢查；兩版若跨判準線，明講。
       headroom ≥ .15 ⇒ 「獎品夠大、rung 1 值得開」
       headroom < .05 ⇒ 「獎品太小、這條臂降級」
       其間          ⇒ 「邊際」
  ② 儀器 gate（per-ckpt；任一沒過 ⇒ 該顆 INSTRUMENT INVALID，判準①不放行）：
     U1 常數計畫（貼起點自由格）r == 0                        （arclen floor 咬）
     U2 teleport 計畫（前半貼 s、後半貼 g）r == 0             （step cap 咬）
     U3 BFS route 假想計畫【三項 reward 本體】raw ≥ 0.90 且 hit 全 1
        （route 折線不是 decode 產物 ⇒ 只驗三項、⛔ 不過閘 —— 閘的定義域＝decode 空間）
     U4 真軌跡 roundtrip decode：C8 通過率 ≥ 0.85
        （校準原則＝現任行為大多通過；p5×p95 邊際各 95%、聯合留 correlation 餘裕。
        〔儀器迭代記錄 2026-09-06、full run 前〕v1 另有「mean r ≥ 0.70」條款，smoke 讀到
        0.686：缺的那塊全來自 roundtrip 的端點命中（decoder recon 端點誤差）——那是
        【decoder 的性質】不是【reward 儀器的性質】，判別力已由 U5 扛 ⇒ 拆掉 mean-r
        條款、mean r 與 roundtrip hit 降為診斷列照印。⛔ 判準①主尺不受此影響。）
     U5 判別力：mean r(roundtrip 真) − mean r(matched 隨機 u decode) ≥ 0.30

── eval 語意 file:line 出處（全部唯讀重建，⛔ 不 import 主檔）──────────────────
  - (s,g) 窗抽樣＝scratch_lacot_rollout.py:465-478（make_batch 抽法）；
    T_CAP 線性插值＝:497-502；正規化 mu/sd 全資料集＝:37-38。
  - eval 錨＝route：_intent_route_zn（:1871-1882）→ route_intent（lacot/intent.py:85-94）
    在 GeoEnergy res=8 資料佔據圖上 BFS 最短路、重採樣 T_A；T_A 從 ckpt 權重反推
    （probe_branch_divergence.py:161 同款、⛔ 不猜 env 預設）。
  - cond＝condvec(s_n,g_n,ix)（:1070-1081）、ix=intent_ad.cond_global(anc)（:1084-1088）；
    INTENT=embed（ckpt 檔名 _ite_）⇒ flow_cond 恆等（:1091-1103 的 2D 分支）。
  - 取樣＝flow.sample（sample_plan :1155-1161 在 GUID_W=0 分支＝原生 flow.sample）；
    專用 torch.Generator（設計卡 §3.1 加固技；三顆同 seed ⇒ 同 z、配對可比）。
  - decode＝_dec（:832-846）在 DEC_START≠hard 分支＝u_dec(u)；三顆 ckpt 皆無 s_embed/
    vq/fsq 鍵（build() assert）；_intent_inv 對 embed 恆等（:1168-1176）。
  - 權重載法＝eval 的 LOAD_EMA=1（:1393-1401）：base 載入後用 ck["ema"] 覆蓋
    cond_enc/cond_head/flow/intent_ad；u_dec/traj_enc/e_pooler 無 ema 影子、載 plain
    （:1402-1406）。precedent＝probe_branch_divergence.py:156-179（已驗）。
  - roundtrip encode＝etarget（:659-661；probe_z_geodesic.py:183-186 同款重建）。
  - 佔據圖／BFS＝GeoEnergy(OBS,mu,sd,res=8)＋lacot/subgoal.py grid_bfs（單一來源）。
  - ⛔ 不套 E_geo refine（norf 顆、且 GRPO 的 policy＝flow 本身 ⇒ 獎品要量 prior 原樣）。

⛔ 唯讀：不改任何既有檔；不訓練、不 sbatch、不上網。CPU。
⚠️ 量測模式註記：本探針量的是【intent-on eval cond】的 headroom（任務規格釘死）。
   R-zero 模式（設計卡主臂）的 headroom 是另一格，本探針不量、不外推。

跑法：
    cd ~/Projects/lacot
    OGBENCH_DATA_DIR=$HOME/data/ogbench MUJOCO_GL=osmesa \
    $HOME/venvs/lacot-rocm/bin/python experiments/probe_grpo_headroom.py \
        2>&1 | tee experiments/probe_grpo_headroom_report.txt
"""
import hashlib
import os
import platform
import sys
import time

import numpy as np
import torch
from torch import nn

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
# ⛔ 只 import 純定義模組（無頂層副作用）—— probe_z_geodesic / probe_branch_divergence 同款紀律
from lacot.nf_head import Flow                    # noqa: E402
from lacot.intent import route_intent             # noqa: E402
from lacot.intent_embed import IntentAdapter      # noqa: E402
from lacot.refine_grad import GeoEnergy           # noqa: E402
from lacot.subgoal import grid_bfs                # noqa: E402
from lacot.traj_decoder import TrajDecoder        # noqa: E402
from lacot.e_target import PerceiverPooler        # noqa: E402

T0 = time.time()
torch.set_num_threads(int(os.environ.get("LACOT_THREADS", "8")))   # zeldajr 是共用 gateway，別吃滿
device = torch.device("cpu")

ENV_NAME = os.environ.get("LACOT_ENV", "pointmaze-large-stitch-v0")
OGB_DATA = os.environ.get("OGBENCH_DATA_DIR", os.path.expanduser("~/data/ogbench"))
N_PAIRS = int(os.environ.get("LACOT_N_PAIRS", 64))     # 任務規格 N=64
G_SAMP = int(os.environ.get("LACOT_G", 16))            # 任務規格 G=16
SEED = int(os.environ.get("LACOT_SEED", 0))            # (s,g) 抽樣 seed＝probe 底座同款
CALIB_N = int(os.environ.get("LACOT_CALIB_N", 256))    # C8 校準集（獨立於量測集）
CALIB_SEED = 12345                                     # 校準集專用 rng（⛔ 不跟量測集共用流）
FLOW_SEED = 20260906                                   # flow 取樣專用 generator（三顆同 seed ⇒ 同 z、配對可比）
W_LEGAL = W_REACH = W_HIT = 1.0 / 3.0                  # 任務規格 w1=w2=w3=1/3

# 三顆量測對象（任務規格；不在就跳過並註明）
CKPTS = {
    "f27n s40 (st8000)":
        "ckpt_large-stitch_self_K8_c256_ch4_st8000_T128_ep2_gu_eorecon_ictr_tch0.5"
        "_btf27n_emw0.999_wu500_s1from_ite_dssoft_norf_cd0.1_bci_s40.pt",
    "idpxm s40 (st11429, idp0.3)":
        "ckpt_large-stitch_self_K8_c256_ch4_st11429_T128_ep2_gu_eorecon_ictr_tch0.5"
        "_btidpxm_emw0.999_wu500_s1from_ite_idp0.3_dssoft_norf_cd0.1_bci_s40.pt",
    "f27nL s40 (st11429)":
        "ckpt_large-stitch_self_K8_c256_ch4_st11429_T128_ep2_gu_eorecon_ictr_tch0.5"
        "_btf27nL_emw0.999_wu500_s1from_ite_dssoft_norf_cd0.1_bci_s40.pt",
}

# 預釘判準數值（跑完照抄，⛔ 不事後改）
VERDICT_BIG = 0.15
VERDICT_SMALL = 0.05
U3_MIN_ROUTE_RAW = 0.90
U4_MIN_PASS = 0.85
U5_MIN_MARGIN = 0.30
NONTRIV_L = 3          # 次要切片：L_BFS ≥ 3 格＝非平凡題（診斷列、⛔ 不進判準）


def hr(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78, flush=True)


def q3(x):
    return np.percentile(np.asarray(x, np.float64), [25, 50, 75])


def sota_mlp(i, h, o, n=2):
    """出處＝scratch_lacot_rollout.py:520-526（逐行同構、載權重用 —— init 會被 ckpt 覆蓋）。"""
    L, p = [], i
    for _ in range(n):
        L += [nn.Linear(p, h), nn.GELU(), nn.LayerNorm(h)]
        p = h
    return nn.Sequential(*L, nn.Linear(p, o))


# ═══ 0. 資料＋佔據圖（切窗/正規化/建圖全走既有慣例）═══════════════════════════
hr("0. 資料與佔據圖（config 行 —— 每個數字的機器與設定）")
print(f"  host={platform.node()}  device=cpu  torch_threads={torch.get_num_threads()}")
print(f"  env={ENV_NAME}  N_PAIRS={N_PAIRS}  G={G_SAMP}  sg_seed={SEED}  flow_seed={FLOW_SEED}"
      f"  calib_n={CALIB_N} (seed {CALIB_SEED})  w=(1/3,1/3,1/3)")
print("  模式：intent-on eval cond（route 錨）、EMA 權重、無 refine、decode=u_dec")
d = np.load(f"{OGB_DATA}/{ENV_NAME}.npz")
OBS = np.asarray(d["observations"], np.float32)
TERM = np.asarray(d["terminals"], bool)
N = OBS.shape[0]
ends = np.flatnonzero(TERM)
starts = np.concatenate([[0], ends[:-1] + 1])
traj_end = np.empty(N, np.int64)
for s0, e0 in zip(starts, ends):
    traj_end[s0:e0 + 1] = e0
assert ends[-1] == N - 1, "⛔ 最後一筆不是 terminal（同 rollout.py:35 自檢）"
mu, sd = OBS.mean(0), OBS.std(0) + 1e-6      # rollout.py:37-38（全資料集）
print(f"  OBS{OBS.shape}  episodes={len(ends)}")

GEO = GeoEnergy(OBS, mu, sd, res=8, device="cpu")     # E 圖同款（rollout.py:1843 的 GEO）
OCC = (GEO.dist[0, 0].numpy() == 0.0)
FREE = np.argwhere(OCC)
LO = np.asarray(GEO.lo, np.float64)
SPAN = np.asarray(GEO.hi - GEO.lo, np.float64)
SHAPE = np.asarray(GEO.shape, np.int64)
CELL_NORM = float(np.mean(SPAN / (SHAPE - 1)))        # 一格的正規化邊長（兩維平均；_E_CELL_XY 同慣例、去 sd）
gh = GEO.health()
print(f"  佔據圖 {tuple(int(s) for s in SHAPE)}  free={int(OCC.sum())}/{int(OCC.size)}"
      f"  CELL_NORM={CELL_NORM:.5f}  health ok={gh['ok']}")
if not gh["ok"]:
    print(f"  ⚠️ GeoEnergy.health() 沒過：{gh['reasons']}（照實報）")


def cells_of(pts):
    """[...,2] 正規化座標 → cell index（round+clip、⛔ 不 snap —— 合法率量的就是落不落自由格；
    出處＝probe_z_geodesic.py:214-222 同款設計選擇）。"""
    idx = np.round((np.asarray(pts, np.float64) - LO) / SPAN * (SHAPE - 1)).astype(np.int64)
    return np.clip(idx, 0, SHAPE - 1)


def cell_one_snap(z, counter):
    """單點版＋snap 保底（給 s/g 這種真資料點用；snap 到了要計數，⛔ 不靜默）。"""
    idx = cells_of(z[None])[0]
    c = (int(idx[0]), int(idx[1]))
    if OCC[c]:
        return c
    counter[0] += 1
    nn_idx = FREE[np.abs(FREE - idx).sum(1).argmin()]
    return (int(nn_idx[0]), int(nn_idx[1]))


def xy_to_cell_raw(xy):
    """route_intent 注入用：【原始】座標 → cell（先 mu/sd 再走 snap 版；＝_e_xy_to_cell :1851-1859）。"""
    z = (np.asarray(xy, np.float64)[:2] - mu) / sd
    idx = np.clip(np.round((z - LO) / SPAN * (SHAPE - 1)).astype(int), 0, SHAPE - 1)
    c = (int(idx[0]), int(idx[1]))
    if OCC[c]:
        return c
    nn_idx = FREE[np.abs(FREE - idx).sum(1).argmin()]
    return (int(nn_idx[0]), int(nn_idx[1]))


def cell_to_zn(c):
    """cell → 格心【正規化】座標（＝_e_cell_to_zn :1866-1867）。"""
    return LO + np.asarray(c, np.float64) / (SHAPE - 1) * SPAN


# ═══ 1. Reward 模組（rung 1 直接重用）═══════════════════════════════════════
_BFS_CACHE = {}     # per-goal dist map，跨 ckpt 共用（圖只有一張）


class GrpoReward:
    """設計卡 §2.1 的 reward 本體＋C8 乘法閘。零模擬器：佔據圖＋BFS＋座標。

    介面（rung 1 拿去就用）：
      calibrate(ratios, max_steps)   → 真分佈（decode 空間）釘 ρ_len（p5）與 δ_step（p95）
      dist_from(g_cell)              → 從 goal cell 的 BFS dist map（per-goal 快取）
      score(pts, s_cell, g_cell)     → dict（r 與全部拆項＋§2.2 偵測欄）；pts [B,T,2] 正規化 np
      score_terms(...)               → 只算三項本體（不過閘；U3 這種非 decode 折線用）
    """

    def __init__(self, occ, w=(W_LEGAL, W_REACH, W_HIT)):
        self.occ = occ
        self.w = w
        self.rho_len = None       # arclen floor 比值（×L_BFS_norm）
        self.delta_step = None    # 鄰步距上限（正規化單位）

    def calibrate(self, ratios, max_steps):
        self.rho_len = float(np.percentile(ratios, 5))
        self.delta_step = float(np.percentile(max_steps, 95))
        return self.rho_len, self.delta_step

    def dist_from(self, g_cell):
        g_cell = tuple(g_cell)
        if g_cell not in _BFS_CACHE:
            _BFS_CACHE[g_cell] = grid_bfs(self.occ, g_cell)
        return _BFS_CACHE[g_cell]

    @staticmethod
    def arclen(pts):
        return np.linalg.norm(np.diff(pts, axis=-2), axis=-1).sum(-1)

    @staticmethod
    def max_step(pts):
        return np.linalg.norm(np.diff(pts, axis=-2), axis=-1).max(-1)

    def score_terms(self, pts, s_cell, g_cell):
        """三項 reward 本體（無閘）。回 dict：raw / r_legal / r_reach / r_hit / d_end / L_bfs。"""
        pts = np.asarray(pts, np.float64)
        B = pts.shape[0]
        dist = self.dist_from(g_cell)
        D_s = dist.get(tuple(s_cell), None)
        idx = cells_of(pts)
        free = self.occ[idx[..., 0], idx[..., 1]]
        r_legal = free.mean(1)
        # 末點 D（⛔ 不 snap：落牆格/不連通 ⇒ ∞ ⇒ reach=0, hit=0 —— 設計卡「D(p_T)=∞⇒0」）
        d_end = np.array([dist.get((int(idx[b, -1, 0]), int(idx[b, -1, 1])), np.inf)
                          for b in range(B)], np.float64)
        if D_s is None:
            r_reach = np.zeros(B)                    # s 都到不了 g 的題 ⇒ 呼叫端應已排除；防禦性 0
        elif D_s == 0:
            r_reach = (d_end == 0).astype(np.float64)
        else:
            r_reach = np.where(np.isinf(d_end), 0.0, 1.0 - np.minimum(d_end / max(D_s, 1) , 1.0))
        r_hit = (d_end <= 1).astype(np.float64)
        raw = self.w[0] * r_legal + self.w[1] * r_reach + self.w[2] * r_hit
        return dict(raw=raw, r_legal=r_legal, r_reach=r_reach, r_hit=r_hit, d_end=d_end,
                    L_bfs=(np.nan if D_s is None else float(D_s)))

    def score(self, pts, s_cell, g_cell):
        assert self.rho_len is not None, "⛔ 先 calibrate 再 score（閘門檻沒釘就打分＝未定義行為）"
        pts = np.asarray(pts, np.float64)
        t = self.score_terms(pts, s_cell, g_cell)
        L_bfs_norm = (0.0 if np.isnan(t["L_bfs"]) else t["L_bfs"] * CELL_NORM)
        al = self.arclen(pts)
        ms = self.max_step(pts)
        gate_arclen = al >= self.rho_len * L_bfs_norm
        gate_step = ms <= self.delta_step
        N_gate = (gate_arclen & gate_step).astype(np.float64)
        t.update(N=N_gate, r=N_gate * t["raw"], gate_arclen=gate_arclen, gate_step=gate_step,
                 arclen=al, max_step=ms)
        return t


# ═══ 2. 抽窗（量測集＝probe 底座同款；校準集獨立 rng）═════════════════════════
def sample_windows(n, seed, chunk):
    """出處＝probe_branch_divergence.py:104-115（rollout.py:465-478 同款）。"""
    rng = np.random.default_rng(seed)
    rows, goals = [], []
    while len(rows) < n:
        r = int(rng.integers(0, N))
        te = int(traj_end[r])
        if te - r < chunk:
            continue
        _d = rng.random()
        gr = int(round(min(r + 1, te) * _d + te * (1 - _d)))
        rows.append(r)
        goals.append(max(gr, min(r + chunk, te)))
    return np.array(rows), np.array(goals)


def build_traj(rows, goals, t_cap):
    """出處＝probe_branch_divergence.py:117-122（rollout.py:497-502 逐字）。"""
    n = len(rows)
    f = np.linspace(rows[:, None].astype(np.float64), goals[:, None].astype(np.float64),
                    t_cap, axis=1).reshape(n, t_cap)
    lo_i = np.floor(f).astype(np.int64)
    hi_i = np.minimum(lo_i + 1, goals[:, None])
    w = (f - lo_i)[..., None]
    return ((OBS[lo_i] * (1.0 - w) + OBS[hi_i] * w - mu) / sd).astype(np.float32)


# ═══ 3. 重建三顆模型（eval 載法：base＋EMA 覆蓋；u_dec/traj_enc/e_pooler plain）═
def build(name, fn):
    path = os.path.join(REPO_ROOT, "results", fn)
    if not os.path.exists(path):
        return None
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = ck["cfg"]
    K, COND, D_MODEL, T_CAP, CHUNK = cfg["K"], cfg["COND"], cfg["D_MODEL"], cfg["T_CAP"], cfg["CHUNK"]
    problems = []
    if "s_embed" in ck:
        problems.append("有 s_embed（DEC_START=hard）—— 本探針沒實作 hard decode 分支")
    if "vq" in ck or "fsq" in ck:
        problems.append("有 vq/fsq 段 —— 本探針沒實作 snap")
    if "intent_ad" not in ck:
        problems.append("沒有 intent_ad —— 無法做『帶 intent 的 eval cond』")
    if cfg["ENC_OBJ"] != "recon_ictr":
        problems.append(f"ENC_OBJ={cfg['ENC_OBJ']}（驗過的是 recon_ictr）")
    if problems:
        print(f"  ⛔ {name} 超出驗證範圍，跳過：{problems}")
        return None
    ta = ck["intent_ad"]["mlp.0.weight"].shape[1] // 2   # 從權重反推 T_A（probe_branch:161 同款）
    ad = IntentAdapter(ta, K)
    enc = sota_mlp(2, 512, 512)
    head = sota_mlp(1024 + ad.cond_extra_dim, 512, COND)                 # rollout.py:996
    fl = Flow(token_dim=D_MODEL, seq_len=K, n_blocks=4, cond_dim=COND)   # rollout.py:999
    dec = TrajDecoder(D_MODEL, T_CAP)                                    # rollout.py:734 同構
    tenc = sota_mlp(2, 512, 512)                                         # rollout.py:559（traj_enc）
    pool = PerceiverPooler(512, D_MODEL, K, 2, 4, max_len=max(512, T_CAP))
    for k, m in (("cond_enc", enc), ("cond_head", head), ("flow", fl), ("intent_ad", ad),
                 ("u_dec", dec), ("traj_enc", tenc), ("e_pooler", pool)):
        m.load_state_dict(ck[k])   # strict=True（預設）：鍵/形狀不合直接炸＝架構 assert
    assert "ema" in ck, f"⛔ {name} 沒有 ema 段"
    n_ema = []
    for k, sdct in ck["ema"].items():
        if k in ("cond_enc", "cond_head", "flow", "intent_ad"):
            {"cond_enc": enc, "cond_head": head, "flow": fl, "intent_ad": ad}[k].load_state_dict(sdct)
            n_ema.append(k)
    for m in (enc, head, fl, ad, dec, tenc, pool):
        m.eval()
        for p in m.parameters():
            p.requires_grad_(False)
    def _fp(sdct):
        h = hashlib.md5()
        for kk in sorted(sdct):
            h.update(kk.encode())
            h.update(sdct[kk].numpy().tobytes())
        return h.hexdigest()[:10]

    print(f"  ✓ {name}: base+EMA 覆蓋 {sorted(n_ema)}、u_dec/traj_enc/e_pooler plain、T_A={ta}")
    print(f"    stage-1 權重指紋 traj_enc/e_pooler/u_dec = {_fp(ck['traj_enc'])}/"
          f"{_fp(ck['e_pooler'])}/{_fp(ck['u_dec'])}   flow(ema)={_fp(ck['ema']['flow'])}"
          f"   ({os.path.basename(fn)})")

    def condvec(s, g, ix=None):
        """rollout.py:1070-1081（embed 分支）：ix=None ⇒ 尾巴拼零。"""
        x = torch.cat([enc(s), enc(g)], 1)
        if ix is None:
            ix = x.new_zeros(x.shape[0], ad.cond_extra_dim)
        return head(torch.cat([x, ix], 1))

    def etarget(traj_t, mask):
        """rollout.py:659-661／probe_z_geodesic.py:183-186 同款。"""
        Bc, Tc, _ = traj_t.shape
        return pool(tenc(traj_t.reshape(Bc * Tc, 2)).reshape(Bc, Tc, 512), key_padding_mask=mask)

    return dict(name=name, flow=fl, condvec=condvec, icond=ad.cond_global, dec=dec,
                etarget=etarget, K=K, D=D_MODEL, T_CAP=T_CAP, CHUNK=CHUNK, TA=ta)


hr("1. 重建模型並載入 EMA 權重（eval 語意）")
MODELS = []
SKIPPED = []
for nm, fn in CKPTS.items():
    m = build(nm, fn)
    if m is None:
        if not os.path.exists(os.path.join(REPO_ROOT, "results", fn)):
            print(f"  ⚠️ {nm}: ckpt 不在（可能還在烤），跳過並註明。檔名 pattern={fn}")
        SKIPPED.append(nm)
    else:
        MODELS.append(m)
assert MODELS, "⛔ 三顆全都不在/不可用 —— 沒東西可量，停手。"
T_CAP = MODELS[0]["T_CAP"]
CHUNK = MODELS[0]["CHUNK"]
K, D_MODEL = MODELS[0]["K"], MODELS[0]["D"]
assert all(m["T_CAP"] == T_CAP and m["CHUNK"] == CHUNK for m in MODELS)


# ═══ 4. 共用題集與校準集 ═════════════════════════════════════════════════════
hr("2. 題集（量測 64 題＝probe 底座；校準 256 窗獨立）")
rows, goals = sample_windows(N_PAIRS, SEED, CHUNK)
traj = build_traj(rows, goals, T_CAP)
S_raw, G_raw = OBS[rows].astype(np.float64), OBS[goals].astype(np.float64)
S_n = torch.tensor((OBS[rows] - mu) / sd, dtype=torch.float32)
G_n = torch.tensor((OBS[goals] - mu) / sd, dtype=torch.float32)
snap_eval = [0]
s_cells = [cell_one_snap(traj[i, 0], snap_eval) for i in range(N_PAIRS)]
g_cells = [cell_one_snap(traj[i, -1], snap_eval) for i in range(N_PAIRS)]
_tmp = GrpoReward(OCC)
pair_L = np.array([np.nan if (L := _tmp.dist_from(g_cells[i]).get(s_cells[i], None)) is None
                   else float(L) for i in range(N_PAIRS)])
valid_pair = ~np.isnan(pair_L)
nontriv = valid_pair & (pair_L >= NONTRIV_L)
print(f"  量測集：{N_PAIRS} 題（snap {snap_eval[0]}、不連通剔 {int((~valid_pair).sum())}）"
      f"  L_BFS 格數 p25/p50/p75={q3(pair_L[valid_pair]).round(1)}"
      f"  同格 L=0 佔 {(pair_L[valid_pair] == 0).mean():.0%}"
      f"  非平凡（L≥{NONTRIV_L}）{int(nontriv.sum())} 題")

c_rows, c_goals = sample_windows(CALIB_N, CALIB_SEED, CHUNK)
c_traj = build_traj(c_rows, c_goals, T_CAP)
snap_cal = [0]
c_s_cells = [cell_one_snap(c_traj[i, 0], snap_cal) for i in range(CALIB_N)]
c_g_cells = [cell_one_snap(c_traj[i, -1], snap_cal) for i in range(CALIB_N)]
c_L = np.array([np.nan if (L := _tmp.dist_from(c_g_cells[i]).get(c_s_cells[i], None)) is None
                else float(L) for i in range(CALIB_N)])
c_ok = ~np.isnan(c_L)
c_pos = c_ok & (c_L > 0)          # ratio 分母要 L>0
raw_seg = np.linalg.norm(np.diff(c_traj, axis=1), axis=-1)         # [CALIB_N, T-1]
raw_ratio = raw_seg.sum(1)[c_pos] / (c_L[c_pos] * CELL_NORM)
raw_maxstep = raw_seg.max(1)
print(f"  校準集：{CALIB_N} 窗（snap {snap_cal[0]}、不連通 {int((~c_ok).sum())}、"
      f"同格 L=0 佔 {(c_L[c_ok] == 0).mean():.0%}）")
print(f"  【raw 插值空間、對照用】arclen/L_BFS p5={np.percentile(raw_ratio, 5):.3f}"
      f"  逐窗 max 步 p95={np.percentile(raw_maxstep, 95):.4f}"
      f"（⛔ 不進閘 —— smoke 實測用它會閘掉 100% 的 decode 產物、含真軌跡 roundtrip，見檔頭）")


# ═══ 5. per-ckpt：C8 校準（decode 空間）→ 儀器 U1–U5 → headroom ═══════════════
RESULTS = {}
for M in MODELS:
    t_ck = time.time()
    hr(f"3. 【{M['name']}】 C8 校準 → 儀器 gate → headroom")
    RWD = GrpoReward(OCC)

    # — C8 校準：真軌跡 roundtrip decode 的分佈（本顆自己的 encoder/decoder）—
    with torch.no_grad():
        mask0 = torch.zeros(CALIB_N, T_CAP, dtype=torch.bool)     # 全 False＝全真點（rollout.py:500）
        et = M["etarget"](torch.from_numpy(c_traj), mask0)
        rt = torch.cat([M["dec"](et[a0:a0 + 256]) for a0 in range(0, CALIB_N, 256)], 0).numpy()
    rt_seg = np.linalg.norm(np.diff(rt.astype(np.float64), axis=1), axis=-1)
    rt_ratio = rt_seg.sum(1)[c_pos] / (c_L[c_pos] * CELL_NORM)
    rt_maxstep = rt_seg.max(1)
    rho_len, delta_step = RWD.calibrate(rt_ratio, rt_maxstep)
    print(f"  ★ 校準值（roundtrip decode 真分佈、進報告）：ρ_len = {rho_len:.3f}（arclen/L_BFS p5、"
          f"n={len(rt_ratio)}）  δ_step = {delta_step:.4f}（逐窗 max 步 p95、n={len(rt_maxstep)}）")
    print(f"    分佈：ratio p5/p50/p95={np.percentile(rt_ratio, [5, 50, 95]).round(2)}"
          f"  max步 p50/p95={np.percentile(rt_maxstep, [50, 95]).round(4)}"
          f"  （raw 空間對照：ratio p5={np.percentile(raw_ratio, 5):.3f}"
          f"、max步 p95={np.percentile(raw_maxstep, 95):.4f} —— 差 {delta_step / np.percentile(raw_maxstep, 95):.1f}×）")

    # — 儀器 gate U1–U5 —
    u1_ok, u2_ok = [], []
    _demo_cand = np.flatnonzero(valid_pair & (pair_L > 4))
    demo_i = int(_demo_cand[0] if len(_demo_cand) else np.flatnonzero(valid_pair)[0])
    for i in np.flatnonzero(valid_pair)[:16]:
        if pair_L[i] <= 1:
            continue
        const_plan = np.tile(cell_to_zn(s_cells[i])[None], (T_CAP, 1))[None]
        u1_ok.append(float(RWD.score(const_plan, s_cells[i], g_cells[i])["r"][0]) == 0.0)
        tele = np.concatenate([np.tile(cell_to_zn(s_cells[i])[None], (T_CAP // 2, 1)),
                               np.tile(cell_to_zn(g_cells[i])[None], (T_CAP - T_CAP // 2, 1))])[None]
        u2_ok.append(float(RWD.score(tele, s_cells[i], g_cells[i])["r"][0]) == 0.0)
    U1 = all(u1_ok) and len(u1_ok) > 0
    U2 = all(u2_ok) and len(u2_ok) > 0
    sc_d = RWD.score(np.tile(cell_to_zn(s_cells[demo_i])[None], (T_CAP, 1))[None],
                     s_cells[demo_i], g_cells[demo_i])
    sc_t = RWD.score(np.concatenate([np.tile(cell_to_zn(s_cells[demo_i])[None], (T_CAP // 2, 1)),
                                     np.tile(cell_to_zn(g_cells[demo_i])[None], (T_CAP // 2, 1))])[None],
                     s_cells[demo_i], g_cells[demo_i])
    print(f"  U1 常數計畫 r==0（{len(u1_ok)} 題）：{'PASS' if U1 else 'FAIL'}"
          f"   例：legal={sc_d['r_legal'][0]:.2f} raw={sc_d['raw'][0]:.3f} → r={sc_d['r'][0]:.1f}"
          f"（arclen {sc_d['arclen'][0]:.3f} < 門檻 {rho_len * sc_d['L_bfs'] * CELL_NORM:.3f}）")
    print(f"  U2 teleport r==0（{len(u2_ok)} 題）：{'PASS' if U2 else 'FAIL'}"
          f"   例：legal={sc_t['r_legal'][0]:.2f} hit={sc_t['r_hit'][0]:.0f} raw={sc_t['raw'][0]:.3f}"
          f" → r={sc_t['r'][0]:.1f}（max_step={sc_t['max_step'][0]:.3f} > δ={delta_step:.3f}）")

    route_raws, route_hits = [], []
    for i in np.flatnonzero(valid_pair)[:32]:
        a = route_intent(OCC, S_raw[i], G_raw[i], xy_to_cell_raw, cell_to_zn, T_CAP)
        if a is None:
            continue
        tt = RWD.score_terms(np.asarray(a, np.float64)[None], s_cells[i], g_cells[i])
        route_raws.append(float(tt["raw"][0]))
        route_hits.append(float(tt["r_hit"][0]))
    U3 = (len(route_raws) > 0 and np.mean(route_raws) >= U3_MIN_ROUTE_RAW
          and np.all(np.array(route_hits) == 1))
    print(f"  U3 route 假想計畫三項本體 mean raw={np.mean(route_raws):.3f} ≥ {U3_MIN_ROUTE_RAW}"
          f" 且 hit 全 1（{len(route_raws)} 題、⛔ 不過閘 —— 閘定義域=decode 空間）："
          f"{'PASS' if U3 else 'FAIL'}")

    rt_r, rt_pass, rt_hit = [], [], []
    for i in np.flatnonzero(c_ok):
        sc = RWD.score(rt[i][None].astype(np.float64), c_s_cells[i], c_g_cells[i])
        rt_r.append(float(sc["r"][0]))
        rt_pass.append(float(sc["N"][0]))
        rt_hit.append(float(sc["r_hit"][0]))
    rt_r, rt_pass, rt_hit = np.array(rt_r), np.array(rt_pass), np.array(rt_hit)
    U4 = rt_pass.mean() >= U4_MIN_PASS
    print(f"  U4 真軌跡 roundtrip：C8 通過率={rt_pass.mean():.1%} ≥ {U4_MIN_PASS:.0%}"
          f"（n={len(rt_r)}）：{'PASS' if U4 else 'FAIL'}"
          f"   診斷：mean r={rt_r.mean():.3f}、roundtrip hit={rt_hit.mean():.1%}"
          f"（＝decoder 端點保真度 ⇒ pass@* 讀數的天花板之一）")

    with torch.no_grad():
        _c0 = M["condvec"](S_n[:8], G_n[:8], None)
        _u_pool = M["flow"].sample(8, _c0, generator=torch.Generator().manual_seed(77))
        U_MEAN, U_STD = _u_pool.mean(0, keepdim=True), _u_pool.std(0, keepdim=True) + 1e-6
        u_rand = U_MEAN + U_STD * torch.randn(64, K, D_MODEL,
                                              generator=torch.Generator().manual_seed(4242))
        pts_rand = M["dec"](u_rand).numpy()
    _vp = np.flatnonzero(valid_pair)
    rand_r = np.array([float(RWD.score(pts_rand[j][None].astype(np.float64),
                                       s_cells[int(_vp[j % len(_vp)])],
                                       g_cells[int(_vp[j % len(_vp)])])["r"][0])
                       for j in range(64)])
    u5_margin = rt_r.mean() - rand_r.mean()
    U5 = u5_margin >= U5_MIN_MARGIN
    print(f"  U5 判別力 mean r(roundtrip 真)−mean r(隨機u)={rt_r.mean():.3f}−{rand_r.mean():.3f}"
          f"={u5_margin:+.3f} ≥ {U5_MIN_MARGIN}：{'PASS' if U5 else 'FAIL'}")

    instrument_valid = U1 and U2 and U3 and U4 and U5
    if instrument_valid:
        print("  ✓ U1–U5 全過 —— 這把尺對這顆、當天、在這台是活的。")
    else:
        print("  ⛔ INSTRUMENT INVALID —— 本顆 headroom 數字照印但【不可當結論】、判準不放行。")

    # — headroom 量測（intent-on eval cond、同 eval 語意）—
    ix_rows, n_noroute = [], 0
    for i in range(N_PAIRS):
        a = route_intent(OCC, S_raw[i], G_raw[i], xy_to_cell_raw, cell_to_zn, M["TA"])
        if a is None:
            n_noroute += 1
            ix_rows.append(None)
        else:
            ix_rows.append(torch.from_numpy(np.asarray(a, np.float32))[None])
    with torch.no_grad():
        conds = []
        for i in range(N_PAIRS):
            ix = None if ix_rows[i] is None else M["icond"](ix_rows[i])
            conds.append(M["condvec"](S_n[i:i + 1], G_n[i:i + 1], ix))
        cond_rep = torch.cat(conds, 0).repeat_interleave(G_SAMP, dim=0)      # [N*G, COND]
        gen = torch.Generator().manual_seed(FLOW_SEED)                       # 三顆同 z、配對可比
        u = M["flow"].sample(N_PAIRS * G_SAMP, cond_rep, generator=gen)
        pts = torch.cat([M["dec"](u[a0:a0 + 256]) for a0 in range(0, len(u), 256)], 0).numpy()
    pts = pts.reshape(N_PAIRS, G_SAMP, T_CAP, 2).astype(np.float64)

    per = dict(r=[], raw=[], N=[], hit=[], ghit=[], legal=[], g_arc=[], g_stp=[],
               ratio=[], mstep=[])
    hack_best = None      # 「短計畫刷分」實例：被 arclen 閘咬、ungated 分最高的樣本
    keep = []
    for i in range(N_PAIRS):
        if not valid_pair[i]:
            continue
        keep.append(i)
        sc = RWD.score(pts[i], s_cells[i], g_cells[i])
        per["r"].append(sc["r"])
        per["raw"].append(sc["raw"])
        per["N"].append(sc["N"])
        per["hit"].append(sc["r_hit"])
        per["ghit"].append(sc["r_hit"] * sc["N"])
        per["legal"].append(sc["r_legal"])
        per["g_arc"].append(~sc["gate_arclen"])
        per["g_stp"].append(~sc["gate_step"])
        per["ratio"].append(sc["arclen"] / max(sc["L_bfs"] * CELL_NORM, 1e-9)
                            if sc["L_bfs"] > 0 else np.full(G_SAMP, np.inf))
        per["mstep"].append(sc["max_step"])
        for j in range(G_SAMP):
            if not sc["gate_arclen"][j]:
                # 「刷分」形狀優先：hit=0（沒到）而 legal 高（貼自由格刷分）；
                # 排序 key＝(是否 hit==0, legal) —— 真刷分排前面、近失樣本墊後。
                key = (sc["r_hit"][j] == 0, float(sc["r_legal"][j]))
                if hack_best is None or key > hack_best["key"]:
                    hack_best = dict(key=key, pair=i, j=j, raw=float(sc["raw"][j]),
                                     legal=float(sc["r_legal"][j]), reach=float(sc["r_reach"][j]),
                                     hit=float(sc["r_hit"][j]), arclen=float(sc["arclen"][j]),
                                     L=float(sc["L_bfs"]), mstep=float(sc["max_step"][j]),
                                     d_end=float(sc["d_end"][j]))
    R = {k: np.stack(v) for k, v in per.items()}                     # [n_valid, G]
    keep = np.array(keep)
    nv = R["r"].shape[0]
    sub = nontriv[keep]                                              # 非平凡題切片

    r1, rG = R["r"].mean(), R["r"].max(1).mean()
    hd_r = R["r"].max(1) - R["r"].mean(1)
    p1, pG = R["hit"].mean(), R["hit"].max(1).mean()
    hd_p = R["hit"].max(1) - R["hit"].mean(1)
    gp1, gpG = R["ghit"].mean(), R["ghit"].max(1).mean()
    hd_gp = R["ghit"].max(1) - R["ghit"].mean(1)
    gate_share = 1.0 - R["N"].mean()
    only_arc = (R["g_arc"] & ~R["g_stp"]).mean()
    only_stp = (R["g_stp"] & ~R["g_arc"]).mean()
    both_g = (R["g_arc"] & R["g_stp"]).mean()
    degen = (R["r"].std(1) < 1e-9)
    degen0 = degen & (R["r"].mean(1) < 1e-9)

    RESULTS[M["name"]] = dict(valid=instrument_valid, hd_p=hd_p, hd_gp=hd_gp, hack=hack_best,
                              rho=rho_len, dstep=delta_step)
    print(f"\n  headroom（n={nv} 題×G={G_SAMP}；route 無路 fallback 拼零 {n_noroute} 題；"
          f"{time.time() - t_ck:.0f}s）")
    print(f"    reward 版： mean r@1={r1:.3f}   mean max r@G={rG:.3f}   "
          f"headroom={hd_r.mean():.3f} ± {hd_r.std(ddof=1) / np.sqrt(nv):.3f}")
    print(f"    成功率版（raw hit）：  pass@1={p1:.3f}   pass@G={pG:.3f}   "
          f"headroom={hd_p.mean():.3f} ± {hd_p.std(ddof=1) / np.sqrt(nv):.3f}")
    print(f"    成功率版（gated hit）：pass@1={gp1:.3f}   pass@G={gpG:.3f}   "
          f"headroom={hd_gp.mean():.3f} ± {hd_gp.std(ddof=1) / np.sqrt(nv):.3f}")
    if sub.sum() >= 4:
        print(f"    ── 非平凡題切片（L_BFS≥{NONTRIV_L}、n={int(sub.sum())}；診斷列⛔不進判準）：")
        print(f"       raw hit  pass@1={R['hit'][sub].mean():.3f}  pass@G={R['hit'][sub].max(1).mean():.3f}"
              f"  headroom={(R['hit'][sub].max(1) - R['hit'][sub].mean(1)).mean():.3f}"
              f" ± {(R['hit'][sub].max(1) - R['hit'][sub].mean(1)).std(ddof=1) / np.sqrt(sub.sum()):.3f}")
        print(f"       reward   r@1={R['r'][sub].mean():.3f}  max r@G={R['r'][sub].max(1).mean():.3f}"
              f"  headroom={(R['r'][sub].max(1) - R['r'][sub].mean(1)).mean():.3f}")
    print(f"    C8 閘（§2.2 偵測欄）：閘掉 {gate_share:.1%}（只 arclen {only_arc:.1%}"
          f"／只 step {only_stp:.1%}／雙閘 {both_g:.1%}）  平均合法率={R['legal'].mean():.1%}")
    print(f"       樣本 arclen/L_BFS 比值 p5/p50={np.percentile(R['ratio'][np.isfinite(R['ratio'])], [5, 50]).round(2)}"
          f"（floor={rho_len:.2f}）  樣本 max 步 p50/p95={np.percentile(R['mstep'], [50, 95]).round(3)}"
          f"（cap={delta_step:.3f}）")
    print(f"    退化群（G 條 r 全同 ⇒ Â≡0 零梯度）：{degen.mean():.1%}（其中全 0 群 {degen0.mean():.1%}）")

# ═══ 6. C8 閘有效性實例（跨顆挑 ungated 分最高的被 arclen 閘樣本）══════════════
hr("4. C8 閘有效性 —— 「短計畫刷分」被閘實例")
best = None
for nm, R in RESULTS.items():
    h = R["hack"]
    if h is not None and (best is None or h["key"] > best[2]["key"]):
        best = (nm, R, h)
if best is not None:
    nm, RR, h = best
    shape_lbl = ("短計畫刷分形狀（hit=0、貼自由格拿 legal 分）" if h["hit"] == 0
                 else "⚠️ 本輪被 arclen 閘咬到的都是『偏短但方向對』的近失樣本 —— 沒抽到 hit=0 的"
                      "純刷分形狀；示範用其中 ungated 分最高的一條，純刷分形狀見 U1 手造例")
    print(f"  來源：{nm}、題 #{h['pair']}、樣本 #{h['j']}（模型自己抽出來的 z，⛔ 非手造）")
    print(f"  形狀：{shape_lbl}")
    _de = "∞" if np.isinf(h["d_end"]) else f"{h['d_end']:.0f}格"
    print(f"    合法率={h['legal']:.2f}  reach={h['reach']:.2f}  hit={h['hit']:.0f}"
          f"  末點距 g={_de}  ⇒ 未閘分數 raw={h['raw']:.3f}")
    print(f"    但 arclen={h['arclen']:.3f} < ρ_len×L_BFS_norm＝{RR['rho']:.3f}×{h['L']:.0f}格"
          f"×{CELL_NORM:.4f}＝{RR['rho'] * h['L'] * CELL_NORM:.3f}")
    print(f"    ⇒ N(z)=0、閘後 r=0.000 —— C8 整包歸零、一分不給（設計卡 §2.1 的 N3 教訓）")
else:
    print("  三顆的取樣中都沒有出現 arclen 閘咬住的樣本（閘只在極端退化解上有事做）——")
    print("  用 U1 的手造常數計畫當實例：legal=1.00、raw=0.333 → 閘後 r=0.000（每顆 U1 已驗）。")

# ═══ 7. 判準（預釘、照抄）════════════════════════════════════════════════════
hr("5. 判準（預釘於檔頭、照抄輸出）")
print(f"  尺＝成功率版 headroom（raw hit）＝ pass@G − pass@1；"
      f"≥ {VERDICT_BIG} ⇒「獎品夠大、rung 1 值得開」；< {VERDICT_SMALL} ⇒「獎品太小、這條臂降級」；其間＝邊際")
for nm, R in RESULTS.items():
    hd = R["hd_p"].mean()
    hdg = R["hd_gp"].mean()
    verdict = ("獎品夠大、rung 1 值得開" if hd >= VERDICT_BIG
               else ("獎品太小、這條臂降級" if hd < VERDICT_SMALL else "邊際"))
    v_g = ("獎品夠大、rung 1 值得開" if hdg >= VERDICT_BIG
           else ("獎品太小、這條臂降級" if hdg < VERDICT_SMALL else "邊際"))
    agree = "（gated 版同判）" if verdict == v_g else f"（⚠️ gated 版跨線：{hdg:.3f} ⇒「{v_g}」）"
    hold = "" if R["valid"] else "【⛔ INSTRUMENT INVALID —— 本判定不放行】"
    print(f"  {nm:30s} headroom={hd:.3f} ⇒ 「{verdict}」{agree}{hold}")
for nm in SKIPPED:
    print(f"  {nm:30s} —— ckpt 不在/不可用，本輪未量（跳過並註明）")

print(f"\n  附註：本量測＝intent-on eval cond、in-distribution 同軌窗題集（probe 底座）。")
print(f"  題集 {(pair_L[valid_pair] == 0).mean():.0%} 為同格（L=0）平凡題 —— 會稀釋 headroom，")
print(f"  非平凡切片已另列。設計卡 §5.1 風險 3 的藥（stitch/teleport 難題集）是後續格，")
print(f"  本探針不外推到那裡。")
print(f"\n耗時 {time.time() - T0:.1f}s")
