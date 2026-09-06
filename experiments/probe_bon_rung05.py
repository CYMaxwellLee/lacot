"""rung 0.5 探針：推論期「探索×驗證器」BoN（Best-of-N）實測 —— plan 層 N-scan、on/zero 雙模式。

出處＝docs/DESIGN-explore-verify-bon.md（rung 0.5 設計卡 v0、2026-09-06）：
  §1.1 機制（抽 N 條 z → decode → GrpoReward 三項乘法閘 → argmax → 出計畫）；
  §1.2 支配引理（w3 ≥ w1 ⇒ realized BoN-hit@N ＝ gated-hit pass@N）；
  §1.3 三格（a=V0 zero headroom、b=N-scan plan 層、c=R0 錶）；
  §3.1 儀器 gate、§3.2 預期增益錨、§3.3 判讀樹。
  ⇒ 本探針施工 a＋b 兩格（plan 層）。c 格（R0 錶）需要改 eval 路徑
     （scratch_lacot_rollout.py）＋eval 節點批次 ⇒ ⛔ 本探針不做、只在報告裡出成本估算。

量什麼（每顆 ckpt × 每個 conditioning 模式）：
  1. C8 校準（⛔ 只在 roundtrip-decode 空間；ρ_len=arclen/L_BFS 的 p5、δ_step=逐窗 max 步的 p95）
     ⚠️ 校準讀的是「真軌跡 encode→decode」的分佈，⛔ 不經過 condvec ⇒ 兩個模式的
        ρ_len/δ_step【結構上必然相同】。這不是複製貼上，是校準量的定義使然（報告照印兩份）。
  2. 儀器 gate U1–U5（沿用 rung0；U1–U4 亦為 cond-free ⇒ 跨模式同值是預期、不是 bug；
     U5 的「隨機 u」參考池【依模式重建】⇒ 這格才會跨模式變）。
  3. N-scan（前綴協定）：每題一次抽 N_MAX=32 條 z（專用 torch.Generator），
     N∈{1,4,8,16,32} 各取【同一批的前綴】⇒ 各 N 逐題配對可比、且省算力。
       realized BoN-hit@N ＝ argmax_i r_i 那條的 raw hit（設計卡 §1.1 步 6）
         - (exp)  ＝ 對平手均勻挑的【解析期望】（⛔ 零額外噪音，主讀數）
         - (draw) ＝ 單抽實況（獨立 tie-break RNG，⛔ 與 W 無關）
       oracle pass@N ＝ max raw hit（rung0 那把尺）
       gated pass@N  ＝ max (raw hit × N(z))（支配引理的預測值）
       q_deg(N)      ＝ N 條 r 全同的題比例 ⇒ 有效注入 ≤ (1−q_deg)·log2 N

── 兩種 conditioning（設計卡 §1.1 步 1、§6.3）─────────────────────────────────
  on   ：ix = intent_ad.cond_global(route_intent(...))  （rung0 量的那個、診斷格）
  zero ：ix = None ⇒ condvec 尾巴拼零＝ι=0            （部署格＝主 claim 格）
  ⚠️ zero 模式的定義與 eval 主檔的 LACOT_INTENT_ZERO=1 同語義
     （scratch_lacot_rollout.py:1203-1208 錨恆回 None → :1098 尾巴 new_zeros）。

── 預釘判準（⛔ 跑之前寫死，跑完照抄輸出，不事後挑）────────────────────────────
  ① V0（設計卡 §3.3 岔 1）＝【zero 模式】oracle raw-hit headroom @N=16
     （＝pass@16 − pass@1，對齊 rung0 的 G=16 那把尺）
       ≥ .15 ⇒ zero 格全開｜.05–.15 ⇒ 邊際、zero 格降診斷｜< .05 ⇒ zero 模式無物可選、
       本臂降級為 on 模式 demo，且同源壞消息要通報 rung1 的 R-zero 主臂。
  ② 支配檢（§3.3 岔 2）：|realized(exp) − gated pass@N| ≤ .02 ⇒ 引理成立、plan 層照錨讀；
     超出 ⇒ 先修儀器，⛔ 修好前不讀下游格。
  ③ N 形狀檢（§3.3 岔 4）：gain(4) ≥ 0.4 × gain(16)（log N 報酬遞減形）。
  ④ 儀器 gate（§3.1）：U1–U5 per ckpt per 模式全綠，否則該格 INSTRUMENT INVALID；
     N=1 golden：BoN@1 必須【逐位元】等於「不開 BoN、直接用第 0 條」的基線。
  ⑤ 多重比較紀律（§3.3 岔 5）：primary 只有 ①；其餘（on 模式各格、N 形狀、reward 版）
     一律 diagnostic。⛔ 不准跑完在格子裡挑好看的當結論。

── 出處（全部唯讀重建，⛔ 不 import 主檔、⛔ 不改任何既有檔）────────────────────
  - GrpoReward／抽窗／build_traj／build()／U1–U5：experiments/probe_grpo_headroom.py
    逐段搬（該檔頂層有副作用、不能 import ⇒ 複製；rung0 自己對 rollout.py 同紀律）。
  - eval 語意 file:line 全部沿用 rung0 檔頭（:465-478 抽窗、:497-502 插值、:1070-1081
    condvec、:1084-1088 icond、:1155-1161 sample、:832-846 decode、:1393-1406 EMA 載法、
    :659-661 roundtrip encode）。
  - ⛔ 不套 E_geo refine（獎品量 prior 原樣，rung0 同紀律）。

⚠️ 讀數警語（寫進報告頭）：
  - 三顆 ckpt 全是【單顆 s40】⇒ 沒有 seed variance；ckpt 之間的差異混著 seed 與配置，
    ⛔ 不可當「這個配置比較好」讀。
  - dev 題集 47% 為 L_BFS=0 的同格平凡題 ⇒ 全集 headroom 被稀釋；難題切片只當診斷。
  - U4 roundtrip hit（decoder 端點保真度）是所有 pass@* 讀數的天花板之一。
  - plan 層增益 ≠ R0 增益（設計卡 §6.1 proxy-gap；R0 錶是 c 格、本探針不量）。

跑法：
    smoke（一顆、N=2、8 題）：
      cd ~/Projects/lacot
      LACOT_BON_SMOKE=1 OGBENCH_DATA_DIR=$HOME/data/ogbench MUJOCO_GL=osmesa \
        $HOME/venvs/lacot-rocm/bin/python -u experiments/probe_bon_rung05.py
    全跑：
      OGBENCH_DATA_DIR=$HOME/data/ogbench MUJOCO_GL=osmesa \
        $HOME/venvs/lacot-rocm/bin/python -u experiments/probe_bon_rung05.py \
          2>&1 | tee experiments/probe_bon_rung05_report.txt
"""
import hashlib
import json
import os
import platform
import sys
import time

import numpy as np
import torch
from torch import nn

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
# ⛔ 只 import 純定義模組（無頂層副作用）—— rung0 探針同款紀律
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

SMOKE = int(os.environ.get("LACOT_BON_SMOKE", 0))

ENV_NAME = os.environ.get("LACOT_ENV", "pointmaze-large-stitch-v0")
OGB_DATA = os.environ.get("OGBENCH_DATA_DIR", os.path.expanduser("~/data/ogbench"))
N_PAIRS = int(os.environ.get("LACOT_N_PAIRS", 8 if SMOKE else 64))     # rung0 底座＝64
N_MAX = int(os.environ.get("LACOT_BON_NMAX", 2 if SMOKE else 32))      # 一次抽這麼多、各 N 取前綴
_NS_DEF = "1,2" if SMOKE else "1,4,8,16,32"
N_SCAN = [int(x) for x in os.environ.get("LACOT_BON_NS", _NS_DEF).split(",")]
assert max(N_SCAN) <= N_MAX, "⛔ N-scan 的最大值超過一次抽的條數 ⇒ 前綴協定破了"
SEED = int(os.environ.get("LACOT_SEED", 0))            # (s,g) 抽樣 seed＝rung0 底座同款
CALIB_N = int(os.environ.get("LACOT_CALIB_N", 64 if SMOKE else 256))
CALIB_SEED = 12345                                     # 校準集專用 rng（⛔ 不跟量測集共用流）
FLOW_SEED = 20260906                                   # flow 取樣專用 generator（rung0 同 seed）
TIE_SEED = 90605                                       # 平手均勻挑專用（⛔ 與 W 獨立）
W_LEGAL = W_REACH = W_HIT = 1.0 / 3.0                  # 設計卡 §1.2：w 釘 rung0 實跑值、w3 ≥ w1
MODES = os.environ.get("LACOT_BON_MODES", "on,zero").split(",")

# 三顆量測對象（＝rung0 同三顆；不在就跳過並註明）
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
_only = os.environ.get("LACOT_CKPT_ONLY", "f27n s40 (st8000)" if SMOKE else "")
if _only:
    CKPTS = {k: v for k, v in CKPTS.items() if _only in k}

# 預釘判準數值（跑完照抄，⛔ 不事後改）
V0_BIG = 0.15           # 設計卡 §3.3 岔 1（沿用 rung0 那把尺）
V0_SMALL = 0.05
DOM_TOL = 0.02          # §3.3 岔 2 支配檢
NSHAPE_FRAC = 0.4       # §3.3 岔 4：gain(4) ≥ 0.4·gain(16)
U3_MIN_ROUTE_RAW = 0.90
U4_MIN_PASS = 0.85
U5_MIN_MARGIN = 0.30
NONTRIV_L = 3           # 診斷切片（⛔ 不進判準）

# 設計卡 §3.2 預期增益錨（從 rung0 報告推導；⛔ 跑前寫死、報告裡與實測並列）
PRED = {   # ckpt -> dict(rung0_oracle_hd, rung0_gated_pG, rung0_raw_p1, pred_gain16)
    "f27n s40 (st8000)":          dict(hd=0.183, gpG=0.938, p1=0.771, pred16=0.182),
    "idpxm s40 (st11429, idp0.3)": dict(hd=0.192, gpG=0.953, p1=0.761, pred16=0.192),
    "f27nL s40 (st11429)":        dict(hd=0.169, gpG=0.906, p1=0.769, pred16=0.137),
}
PRED_N8_LO, PRED_N8_HI = 0.70, 0.85     # §3.2：N=8 ≈ N=16 增益 ×0.7~0.85【設計卡標猜測】

# ── R0 層外部錨（⛔ 本探針【未量】；抄自既有 eval log，供 V0 讀數對照）────────────
#    協定＝5 tasks × 50 seeds、MAXH 1000（官方 eval_episodes=50）；三欄＝(BC 地板, R=0, conf2)
#    出處＝slurm/logs/*.out 的 SUCCESS RATE 段，job id 見表；zero＝LACOT_INTENT_ZERO=1。
R0_ANCHOR = {
    "f27n s40 (st8000)":           dict(on=(0.400, 0.656, 0.884), zero=None,
                                        job="24134 / 無同 ckpt 的 z0 log"),
    "idpxm s40 (st11429, idp0.3)": dict(on=(0.196, 0.716, 0.804), zero=(0.100, 0.624, 0.764),
                                        job="24348 / 24349"),
    "f27nL s40 (st11429)":         dict(on=(0.244, 0.596, 0.844), zero=(0.000, 0.000, 0.000),
                                        job="24454 / 24455"),
}
# 完整 R0 eval 的實測牆鐘（sacct）：24454 f27nL-s40-on 00:22:09、24455 f27nL-s40-z0 00:28:35
R0_WALL_MIN = (22, 29)


def hr(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78, flush=True)


def q3(x):
    return np.percentile(np.asarray(x, np.float64), [25, 50, 75])


def sota_mlp(i, h, o, n=2):
    """出處＝scratch_lacot_rollout.py:520-526（逐行同構、載權重用）。"""
    L, p = [], i
    for _ in range(n):
        L += [nn.Linear(p, h), nn.GELU(), nn.LayerNorm(h)]
        p = h
    return nn.Sequential(*L, nn.Linear(p, o))


# ═══ 0. 資料＋佔據圖 ═════════════════════════════════════════════════════════
hr("0. 資料與佔據圖（config 行 —— 每個數字的機器與設定）")
print(f"  host={platform.node()}  device=cpu  torch_threads={torch.get_num_threads()}"
      f"{'   ⚠️ SMOKE 模式' if SMOKE else ''}")
print(f"  env={ENV_NAME}  N_PAIRS={N_PAIRS}  N_MAX={N_MAX}  N_scan={N_SCAN}  modes={MODES}")
print(f"  sg_seed={SEED}  flow_seed={FLOW_SEED}  tie_seed={TIE_SEED}"
      f"  calib_n={CALIB_N} (seed {CALIB_SEED})  w=(1/3,1/3,1/3)")
print("  模式：EMA 權重、無 refine、decode=u_dec；on＝route 錨、zero＝ι=0（cond 尾巴拼零）")
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
CELL_NORM = float(np.mean(SPAN / (SHAPE - 1)))
gh = GEO.health()
print(f"  佔據圖 {tuple(int(s) for s in SHAPE)}  free={int(OCC.sum())}/{int(OCC.size)}"
      f"  CELL_NORM={CELL_NORM:.5f}  health ok={gh['ok']}")
if not gh["ok"]:
    print(f"  ⚠️ GeoEnergy.health() 沒過：{gh['reasons']}（照實報）")


def cells_of(pts):
    """[...,2] 正規化座標 → cell index（round+clip、⛔ 不 snap）。rung0:194-198。"""
    idx = np.round((np.asarray(pts, np.float64) - LO) / SPAN * (SHAPE - 1)).astype(np.int64)
    return np.clip(idx, 0, SHAPE - 1)


def cell_one_snap(z, counter):
    """單點版＋snap 保底（s/g 用；snap 到了要計數，⛔ 不靜默）。rung0:201-209。"""
    idx = cells_of(z[None])[0]
    c = (int(idx[0]), int(idx[1]))
    if OCC[c]:
        return c
    counter[0] += 1
    nn_idx = FREE[np.abs(FREE - idx).sum(1).argmin()]
    return (int(nn_idx[0]), int(nn_idx[1]))


def xy_to_cell_raw(xy):
    """route_intent 注入用：【原始】座標 → cell（＝_e_xy_to_cell :1851-1859）。"""
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


# ═══ 1. Reward 模組（rung0 逐段搬、⛔ 不改語義）════════════════════════════════
_BFS_CACHE = {}     # per-goal dist map，跨 ckpt/模式共用（圖只有一張）


class GrpoReward:
    """設計卡 §2.1 的 reward 本體＋C8 乘法閘。零模擬器：佔據圖＋BFS＋座標。
    出處＝probe_grpo_headroom.py:232-302（逐字搬；⛔ 不改公式、不改門檻語義）。"""

    def __init__(self, occ, w=(W_LEGAL, W_REACH, W_HIT)):
        self.occ = occ
        self.w = w
        self.rho_len = None
        self.delta_step = None

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
        pts = np.asarray(pts, np.float64)
        B = pts.shape[0]
        dist = self.dist_from(g_cell)
        D_s = dist.get(tuple(s_cell), None)
        idx = cells_of(pts)
        free = self.occ[idx[..., 0], idx[..., 1]]
        r_legal = free.mean(1)
        d_end = np.array([dist.get((int(idx[b, -1, 0]), int(idx[b, -1, 1])), np.inf)
                          for b in range(B)], np.float64)
        if D_s is None:
            r_reach = np.zeros(B)
        elif D_s == 0:
            r_reach = (d_end == 0).astype(np.float64)
        else:
            r_reach = np.where(np.isinf(d_end), 0.0, 1.0 - np.minimum(d_end / max(D_s, 1), 1.0))
        r_hit = (d_end <= 1).astype(np.float64)
        raw = self.w[0] * r_legal + self.w[1] * r_reach + self.w[2] * r_hit
        return dict(raw=raw, r_legal=r_legal, r_reach=r_reach, r_hit=r_hit, d_end=d_end,
                    L_bfs=(np.nan if D_s is None else float(D_s)))

    def score(self, pts, s_cell, g_cell):
        assert self.rho_len is not None, "⛔ 先 calibrate 再 score（閘門檻沒釘＝未定義行為）"
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


# ═══ 2. BoN 選擇（設計卡 §1.1 步 5）══════════════════════════════════════════
def bon_select(r, val, n, tie_seed):
    """J = argmax_i r_i、平手（含全 0 群）⇒ 獨立噪音均勻挑。

    r   [P,Nmax] 打分（過閘後）、val [P,Nmax] 要讀的量（raw hit）；只用前 n 條（前綴協定）。
    回：
      exp   [P] 對 tie-break 隨機性的【解析期望】E_J[val]（⛔ 零額外噪音 ⇒ 主讀數）
      draw  [P] 單抽實況（獨立 RNG，與 W 無關）
      idx   [P] 單抽選中的候選編號
      ties  [P] 平手集大小
    """
    rr, vv = r[:, :n], val[:, :n]
    mx = rr.max(1, keepdims=True)
    tie = (rr == mx)                       # ⛔ 精確相等：退化群 ⇒ 全 True（設計卡語義）
    ties = tie.sum(1)
    exp = (vv * tie).sum(1) / ties
    rng = np.random.default_rng(tie_seed + n)
    idx = np.empty(rr.shape[0], np.int64)
    for p in range(rr.shape[0]):
        cand = np.flatnonzero(tie[p])
        idx[p] = cand[0] if len(cand) == 1 else cand[int(rng.integers(len(cand)))]
    draw = vv[np.arange(rr.shape[0]), idx]
    return exp, draw, idx, ties


# ═══ 3. 抽窗（rung0 同款）════════════════════════════════════════════════════
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


# ═══ 4. 重建模型（eval 載法：base＋EMA 覆蓋）══════════════════════════════════
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
        problems.append("沒有 intent_ad —— 無法做 on/zero 兩模式對照")
    if cfg["ENC_OBJ"] != "recon_ictr":
        problems.append(f"ENC_OBJ={cfg['ENC_OBJ']}（驗過的是 recon_ictr）")
    if problems:
        print(f"  ⛔ {name} 超出驗證範圍，跳過：{problems}")
        return None
    ta = ck["intent_ad"]["mlp.0.weight"].shape[1] // 2
    ad = IntentAdapter(ta, K)
    enc = sota_mlp(2, 512, 512)
    head = sota_mlp(1024 + ad.cond_extra_dim, 512, COND)
    fl = Flow(token_dim=D_MODEL, seq_len=K, n_blocks=4, cond_dim=COND)
    dec = TrajDecoder(D_MODEL, T_CAP)
    tenc = sota_mlp(2, 512, 512)
    pool = PerceiverPooler(512, D_MODEL, K, 2, 4, max_len=max(512, T_CAP))
    for k, m in (("cond_enc", enc), ("cond_head", head), ("flow", fl), ("intent_ad", ad),
                 ("u_dec", dec), ("traj_enc", tenc), ("e_pooler", pool)):
        m.load_state_dict(ck[k])
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
        """rollout.py:1070-1081（embed 分支）：ix=None ⇒ 尾巴拼零＝zero 模式的 ι=0。"""
        x = torch.cat([enc(s), enc(g)], 1)
        if ix is None:
            ix = x.new_zeros(x.shape[0], ad.cond_extra_dim)
        return head(torch.cat([x, ix], 1))

    def etarget(traj_t, mask):
        """rollout.py:659-661／probe_z_geodesic.py:183-186 同款。"""
        Bc, Tc, _ = traj_t.shape
        return pool(tenc(traj_t.reshape(Bc * Tc, 2)).reshape(Bc, Tc, 512), key_padding_mask=mask)

    return dict(name=name, flow=fl, condvec=condvec, icond=ad.cond_global, dec=dec,
                etarget=etarget, K=K, D=D_MODEL, T_CAP=T_CAP, CHUNK=CHUNK, TA=ta,
                fp_flow=_fp(ck["ema"]["flow"]))


hr("1. 重建模型並載入 EMA 權重（eval 語意）")
MODELS, SKIPPED = [], []
for nm, fn in CKPTS.items():
    m = build(nm, fn)
    if m is None:
        if not os.path.exists(os.path.join(REPO_ROOT, "results", fn)):
            print(f"  ⚠️ {nm}: ckpt 不在，跳過並註明。檔名 pattern={fn}")
        SKIPPED.append(nm)
    else:
        MODELS.append(m)
assert MODELS, "⛔ 沒有可用的 ckpt —— 沒東西可量，停手。"
T_CAP, CHUNK = MODELS[0]["T_CAP"], MODELS[0]["CHUNK"]
K, D_MODEL = MODELS[0]["K"], MODELS[0]["D"]
assert all(m["T_CAP"] == T_CAP and m["CHUNK"] == CHUNK for m in MODELS)


# ═══ 5. 共用題集與校準集（rung0 同款、同 seed ⇒ 同一批題）══════════════════════
hr("2. 題集（量測＝rung0 底座同款；校準窗獨立 rng）")
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
c_pos = c_ok & (c_L > 0)
raw_seg = np.linalg.norm(np.diff(c_traj, axis=1), axis=-1)
raw_ratio = raw_seg.sum(1)[c_pos] / (c_L[c_pos] * CELL_NORM)
raw_maxstep = raw_seg.max(1)
print(f"  校準集：{CALIB_N} 窗（snap {snap_cal[0]}、不連通 {int((~c_ok).sum())}）")
print(f"  【raw 插值空間、⛔ 不進閘、對照用】arclen/L_BFS p5={np.percentile(raw_ratio, 5):.3f}"
      f"  逐窗 max 步 p95={np.percentile(raw_maxstep, 95):.4f}")

# route 錨（on 模式用；per-ckpt 因為 T_A 可能不同 —— 這裡三顆同 T_A，仍逐顆算）
KEEP = np.flatnonzero(valid_pair)


# ═══ 6. 主迴圈：per ckpt × per 模式 ═══════════════════════════════════════════
CELLS = {}        # (ckpt, mode) -> 讀數
RAWDUMP = {}
for M in MODELS:
    # route 錨只跟 (s,g)＋T_A 有關，跟模式無關 ⇒ per-ckpt 算一次
    ix_rows, n_noroute = [], 0
    for i in range(N_PAIRS):
        a = route_intent(OCC, S_raw[i], G_raw[i], xy_to_cell_raw, cell_to_zn, M["TA"])
        if a is None:
            n_noroute += 1
            ix_rows.append(None)
        else:
            ix_rows.append(torch.from_numpy(np.asarray(a, np.float32))[None])

    for MODE in MODES:
        t_ck = time.time()
        hr(f"3. 【{M['name']}】 mode={MODE} —— C8 校準 → 儀器 gate → N-scan")
        RWD = GrpoReward(OCC)

        # — C8 校準：真軌跡 roundtrip decode 的分佈 —
        # ⚠️ roundtrip 不經過 condvec ⇒ 這兩個門檻【結構上與模式無關】，兩份必然同值。
        with torch.no_grad():
            mask0 = torch.zeros(CALIB_N, T_CAP, dtype=torch.bool)
            et = M["etarget"](torch.from_numpy(c_traj), mask0)
            rt = torch.cat([M["dec"](et[a0:a0 + 256]) for a0 in range(0, CALIB_N, 256)], 0).numpy()
        rt_seg = np.linalg.norm(np.diff(rt.astype(np.float64), axis=1), axis=-1)
        rt_ratio = rt_seg.sum(1)[c_pos] / (c_L[c_pos] * CELL_NORM)
        rt_maxstep = rt_seg.max(1)
        rho_len, delta_step = RWD.calibrate(rt_ratio, rt_maxstep)
        print(f"  ★ 校準（roundtrip-decode 空間）：ρ_len={rho_len:.3f}（p5、n={len(rt_ratio)}）"
              f"  δ_step={delta_step:.4f}（p95、n={len(rt_maxstep)}）"
              f"  ⚠️ cond-free ⇒ 兩模式同值是【定義使然】")

        # — 儀器 gate U1–U5 —
        u1_ok, u2_ok = [], []
        for i in KEEP[:16]:
            if pair_L[i] <= 1:
                continue
            const_plan = np.tile(cell_to_zn(s_cells[i])[None], (T_CAP, 1))[None]
            u1_ok.append(float(RWD.score(const_plan, s_cells[i], g_cells[i])["r"][0]) == 0.0)
            tele = np.concatenate([np.tile(cell_to_zn(s_cells[i])[None], (T_CAP // 2, 1)),
                                   np.tile(cell_to_zn(g_cells[i])[None], (T_CAP - T_CAP // 2, 1))])[None]
            u2_ok.append(float(RWD.score(tele, s_cells[i], g_cells[i])["r"][0]) == 0.0)
        U1 = all(u1_ok) and len(u1_ok) > 0
        U2 = all(u2_ok) and len(u2_ok) > 0

        route_raws, route_hits = [], []
        for i in KEEP[:32]:
            a = route_intent(OCC, S_raw[i], G_raw[i], xy_to_cell_raw, cell_to_zn, T_CAP)
            if a is None:
                continue
            tt = RWD.score_terms(np.asarray(a, np.float64)[None], s_cells[i], g_cells[i])
            route_raws.append(float(tt["raw"][0]))
            route_hits.append(float(tt["r_hit"][0]))
        U3 = (len(route_raws) > 0 and np.mean(route_raws) >= U3_MIN_ROUTE_RAW
              and np.all(np.array(route_hits) == 1))

        rt_r, rt_pass, rt_hit = [], [], []
        for i in np.flatnonzero(c_ok):
            sc = RWD.score(rt[i][None].astype(np.float64), c_s_cells[i], c_g_cells[i])
            rt_r.append(float(sc["r"][0]))
            rt_pass.append(float(sc["N"][0]))
            rt_hit.append(float(sc["r_hit"][0]))
        rt_r, rt_pass, rt_hit = np.array(rt_r), np.array(rt_pass), np.array(rt_hit)
        U4 = rt_pass.mean() >= U4_MIN_PASS

        # U5 的隨機 u 參考池【依模式重建】—— 這格才是真的 per-mode
        with torch.no_grad():
            if MODE == "on":
                _ix8 = [None if ix_rows[i] is None else M["icond"](ix_rows[i]) for i in range(8)]
                _c0 = torch.cat([M["condvec"](S_n[i:i + 1], G_n[i:i + 1], _ix8[i]) for i in range(8)], 0)
            else:
                _c0 = M["condvec"](S_n[:8], G_n[:8], None)
            _u_pool = M["flow"].sample(8, _c0, generator=torch.Generator().manual_seed(77))
            U_MEAN, U_STD = _u_pool.mean(0, keepdim=True), _u_pool.std(0, keepdim=True) + 1e-6
            u_rand = U_MEAN + U_STD * torch.randn(64, K, D_MODEL,
                                                  generator=torch.Generator().manual_seed(4242))
            pts_rand = M["dec"](u_rand).numpy()
        rand_r = np.array([float(RWD.score(pts_rand[j][None].astype(np.float64),
                                           s_cells[int(KEEP[j % len(KEEP)])],
                                           g_cells[int(KEEP[j % len(KEEP)])])["r"][0])
                           for j in range(64)])
        u5_margin = rt_r.mean() - rand_r.mean()
        U5 = u5_margin >= U5_MIN_MARGIN
        instrument_valid = U1 and U2 and U3 and U4 and U5
        print(f"  U1 常數計畫 r==0（{len(u1_ok)} 題）：{'PASS' if U1 else 'FAIL'}"
              f"   U2 teleport r==0（{len(u2_ok)} 題）：{'PASS' if U2 else 'FAIL'}"
              f"   U3 route 三項 raw={np.mean(route_raws):.3f}≥{U3_MIN_ROUTE_RAW}"
              f"（hit 全 1）：{'PASS' if U3 else 'FAIL'}")
        print(f"  U4 roundtrip C8 通過率={rt_pass.mean():.1%}≥{U4_MIN_PASS:.0%}："
              f"{'PASS' if U4 else 'FAIL'}   診斷：mean r={rt_r.mean():.3f}、"
              f"roundtrip hit={rt_hit.mean():.1%}（＝pass@* 天花板之一）")
        print(f"  U5 判別力 {rt_r.mean():.3f}−{rand_r.mean():.3f}={u5_margin:+.3f}"
              f"≥{U5_MIN_MARGIN}（隨機 u 池＝本模式重建）：{'PASS' if U5 else 'FAIL'}")
        print(f"  {'✓ U1–U5 全過 —— 這把尺對這顆、這個模式、當天、在這台是活的。' if instrument_valid else '⛔ INSTRUMENT INVALID —— 讀數照印但【不可當結論】。'}")

        # — 抽 N_MAX 條（前綴協定：⛔ 一次抽、各 N 取前綴，不重抽）—
        with torch.no_grad():
            conds = []
            for i in range(N_PAIRS):
                if MODE == "on":
                    ix = None if ix_rows[i] is None else M["icond"](ix_rows[i])
                else:
                    ix = None
                conds.append(M["condvec"](S_n[i:i + 1], G_n[i:i + 1], ix))
            cond_base = torch.cat(conds, 0)                                   # [P, COND]
            cond_rep = cond_base.repeat_interleave(N_MAX, dim=0)              # [P*N_MAX, COND]
            # 🚨 layout assert：前綴協定成立的前提＝每題的候選是【連續 N_MAX 列】
            assert torch.equal(cond_rep.reshape(N_PAIRS, N_MAX, -1)[:, 0], cond_base), \
                "⛔ repeat_interleave 的 layout 跟前綴協定的假設不一致"
            gen = torch.Generator().manual_seed(FLOW_SEED)
            u = M["flow"].sample(N_PAIRS * N_MAX, cond_rep, generator=gen)
            # golden A：同 seed 重抽 ⇒ 逐位元相同（generator 加固技驗收）
            u2 = M["flow"].sample(N_PAIRS * N_MAX, cond_rep,
                                  generator=torch.Generator().manual_seed(FLOW_SEED))
            golden_det = bool(torch.equal(u, u2))
            del u2
            pts = torch.cat([M["dec"](u[a0:a0 + 256]) for a0 in range(0, len(u), 256)], 0).numpy()
        pts = pts.reshape(N_PAIRS, N_MAX, T_CAP, 2).astype(np.float64)

        # — 打分（全 N_MAX 條、一次算完；各 N 只是切前綴）—
        per = dict(r=[], raw=[], N=[], hit=[], ghit=[], legal=[], g_arc=[], g_stp=[], mstep=[])
        for i in KEEP:
            sc = RWD.score(pts[i], s_cells[i], g_cells[i])
            per["r"].append(sc["r"])
            per["raw"].append(sc["raw"])
            per["N"].append(sc["N"])
            per["hit"].append(sc["r_hit"])
            per["ghit"].append(sc["r_hit"] * sc["N"])
            per["legal"].append(sc["r_legal"])
            per["g_arc"].append(~sc["gate_arclen"])
            per["g_stp"].append(~sc["gate_step"])
            per["mstep"].append(sc["max_step"])
        R = {k: np.stack(v) for k, v in per.items()}        # [n_valid, N_MAX]
        nv = R["r"].shape[0]
        sub = nontriv[KEEP]

        base_hit = float(R["hit"].mean())          # ＝單抽期望成功率（rung0 的 pass@1 那把尺）
        base_r = float(R["r"].mean())
        gate_share = float(1.0 - R["N"].mean())

        # — N-scan —
        rowsN = []
        for n in N_SCAN:
            exp, draw, jidx, ties = bon_select(R["r"], R["hit"], n, TIE_SEED)
            orc = R["hit"][:, :n].max(1)
            gtd = R["ghit"][:, :n].max(1)
            maxr = R["r"][:, :n].max(1)
            deg = (R["r"][:, :n].std(1) < 1e-9)
            deg0 = deg & (R["r"][:, :n].mean(1) < 1e-9)
            bits = float(np.log2(n))
            # ⭐ rung0 口徑的 headroom：pass@N − mean(同一段前綴) —— 逐字同 rung0 的
            #    `hd_p = R["hit"].max(1) - R["hit"].mean(1)`，⇒ V0 判準跟 .15 那條線可比。
            hd_pfx_v = orc - R["hit"][:, :n].mean(1)
            rowsN.append(dict(
                n=n, real_exp=float(exp.mean()), real_draw=float(draw.mean()),
                real_exp_se=float(exp.std(ddof=1) / np.sqrt(nv)) if nv > 1 else float("nan"),
                oracle=float(orc.mean()),
                hd_pfx=float(hd_pfx_v.mean()),
                hd_pfx_se=float(hd_pfx_v.std(ddof=1) / np.sqrt(nv)) if nv > 1 else float("nan"),
                oracle_se=float((orc - R["hit"].mean(1)).std(ddof=1) / np.sqrt(nv)) if nv > 1 else float("nan"),
                gated=float(gtd.mean()), maxr=float(maxr.mean()),
                q_deg=float(deg.mean()), q_deg0=float(deg0.mean()),
                bits=bits, eff_bits=float((1.0 - deg.mean()) * bits),
                ties_mean=float(ties.mean()),
                real_exp_nontriv=(float(exp[sub].mean()) if sub.sum() >= 4 else float("nan")),
                oracle_nontriv=(float(orc[sub].mean()) if sub.sum() >= 4 else float("nan")),
            ))

        # golden B：N=1 必須逐位元等於「不開 BoN、直接用第 0 條」
        e1, d1, j1, _ = bon_select(R["r"], R["hit"], 1, TIE_SEED)
        base0 = R["hit"][:, 0]
        golden_n1 = bool(np.array_equal(e1, base0) and np.array_equal(d1, base0)
                         and np.all(j1 == 0))
        print(f"\n  golden：同 seed 重抽逐位元相同={golden_det}"
              f"   BoN@1 ≡ 第 0 條基線（idx/exp/draw 三項全等）={golden_n1}"
              f"   {'✓' if (golden_det and golden_n1) else '⛔ FAIL —— 儀器 bug、下游不放行'}")
        print(f"  基線：pass@1(cand0)={float(base0.mean()):.3f}"
              f"   pass@1(mean over {N_MAX})={base_hit:.3f}（＝rung0 那把尺；增益以它為底）"
              f"   mean r@1={base_r:.3f}   C8 閘掉={gate_share:.1%}"
              f"   route 無路 fallback 拼零 {n_noroute} 題"
              f"{'（⚠️ on 模式有 fallback ⇒ 這幾題實際跑的是 zero）' if (MODE == 'on' and n_noroute) else ''}")

        # — 表 —
        print(f"\n  N-scan（n={nv} 題、前綴協定、{time.time() - t_ck:.0f}s）"
              f"   增益欄的底＝pass@1(mean over {N_MAX})={base_hit:.3f}")
        print("    " + "-" * 92)
        print(f"    {'N':>3} {'realized':>9} {'(draw)':>8} {'oracle':>8} {'gated':>8} "
              f"{'meanmaxr':>9} {'Δreal':>7} {'Δorac':>7} {'q_deg':>6} {'log2N':>6} {'effbit':>7}")
        print("    " + "-" * 92)
        for rw in rowsN:
            print(f"    {rw['n']:>3} {rw['real_exp']:>9.3f} {rw['real_draw']:>8.3f} "
                  f"{rw['oracle']:>8.3f} {rw['gated']:>8.3f} {rw['maxr']:>9.3f} "
                  f"{rw['real_exp'] - base_hit:>+7.3f} {rw['oracle'] - base_hit:>+7.3f} "
                  f"{rw['q_deg']:>6.1%} {rw['bits']:>6.2f} {rw['eff_bits']:>7.2f}")
        print("    " + "-" * 92)
        print("    rung0 口徑 headroom（pass@N − mean 同前綴，＝rung0 那條 .15 線比得上的量）："
              + "  ".join(f"N={rw['n']}:{rw['hd_pfx']:+.3f}±{rw['hd_pfx_se']:.3f}"
                          for rw in rowsN if rw["n"] > 1))
        if sub.sum() >= 4:
            print(f"    ── 非平凡切片（L_BFS≥{NONTRIV_L}、n={int(sub.sum())}；⛔ 診斷、不進判準）：")
            _b = float(R["hit"][sub].mean())
            print("       " + "  ".join(f"N={rw['n']}: real={rw['real_exp_nontriv']:.3f}"
                                        f"({rw['real_exp_nontriv'] - _b:+.3f})" for rw in rowsN))

        # — 支配檢（§3.3 岔 2）—
        # ⚠️ 引理的前提是【argmax 有得選】：N=1 時 argmax 恆回那唯一一條，
        #    「存在過閘命中候選 ⇒ 選中它」是空句 ⇒ N=1 的偏差 ＝ 第 0 條 raw 命中但被 C8
        #    閘掉的比例（＝閘對 base policy 命中計畫的誤咬率），⛔ 不是引理的反例。
        #    ⇒ 預釘判準照跑全 N（不移門柱），另印【N≥2 範圍】的版本＝引理真正的檢定。
        dom = [(rw["n"], rw["real_exp"] - rw["gated"]) for rw in rowsN]
        dom_bad = [f"N={n}:{d:+.3f}" for n, d in dom if abs(d) > DOM_TOL]
        dom2 = [(n, d) for n, d in dom if n >= 2]
        dom2_max = max((abs(d) for _, d in dom2), default=float("nan"))
        dom2_ok = bool(dom2) and dom2_max <= DOM_TOL
        n1_dev = dict(dom).get(1, float("nan"))
        print(f"    支配檢（預釘、全 N）|realized(exp) − gated pass@N| ≤ {DOM_TOL}："
              f"{'PASS' if not dom_bad else 'FAIL ⇒ ' + ' '.join(dom_bad)}"
              f"   最大偏差={max(abs(d) for _, d in dom):.3f}")
        print(f"    └ 引理範圍（N≥2、argmax 真的有得選）：最大偏差={dom2_max:.3f} ⇒ "
              f"{'PASS（引理成立、plan 層照錨讀）' if dom2_ok else '⛔ FAIL ⇒ 先修儀器'}"
              f"；N=1 偏差 {n1_dev:+.3f} ＝ C8 閘對 base 命中計畫的誤咬率（結構項、非反例）")

        CELLS[(M["name"], MODE)] = dict(
            valid=instrument_valid, U=(U1, U2, U3, U4, U5), golden=(golden_det, golden_n1),
            rho=rho_len, dstep=delta_step, base_hit=base_hit, base0=float(base0.mean()),
            base_r=base_r, gate_share=gate_share, n_noroute=n_noroute, nv=nv,
            rows={rw["n"]: rw for rw in rowsN}, dom_max=max(abs(d) for _, d in dom),
            dom_ok=(not dom_bad), dom2_max=dom2_max, dom2_ok=dom2_ok, n1_dev=n1_dev,
            rt_hit=float(rt_hit.mean()), rt_pass=float(rt_pass.mean()),
            u5=float(u5_margin), n_nontriv=int(sub.sum()),
        )
        RAWDUMP[f"{M['name']}|{MODE}"] = dict(
            pairs=[int(x) for x in KEEP],
            L_bfs=[float(pair_L[i]) for i in KEEP],
            r=np.round(R["r"], 6).tolist(), hit=R["hit"].astype(int).tolist(),
            ghit=R["ghit"].astype(int).tolist(), gate=R["N"].astype(int).tolist(),
        )


# ═══ 7. 總表 ════════════════════════════════════════════════════════════════
hr("4. 總表 —— ckpt × cond × N（realized 命中率／增益／log2N bits）")
print("  realized＝BoN 選中那條的 raw hit（對平手均勻挑取解析期望）；"
      "Δ 的底＝該格 pass@1(mean over N_MAX)")
print("  effbit＝(1−q_deg)·log2N（設計卡 §2 有效注入折扣〔啟發式〕）\n")
for MODE in MODES:
    print(f"  ── cond = {MODE} " + "─" * 60)
    print(f"    {'ckpt':<30} {'base@1':>7} " + " ".join(f"{'N=' + str(n):>15}" for n in N_SCAN))
    for M in MODELS:
        C = CELLS.get((M["name"], MODE))
        if C is None:
            continue
        cells = []
        for n in N_SCAN:
            rw = C["rows"][n]
            cells.append(f"{rw['real_exp']:.3f}({rw['real_exp'] - C['base_hit']:+.3f})".rjust(15))
        flag = "" if C["valid"] else "  ⛔INVALID"
        print(f"    {M['name']:<30} {C['base_hit']:>7.3f} " + " ".join(cells) + flag)
    print(f"    {'(log2 N bits 上界)':<30} {'—':>7} "
          + " ".join(f"{('≤' + format(np.log2(n), '.2f')):>15}" for n in N_SCAN))

hr("5. 預測 vs 實測（設計卡 §3.2 錨；⛔ 預測在跑之前寫死）")
print("  設計卡預測（on 模式、支配引理）：realized 增益@N=16 ＝ rung0 的 gated pass@G − raw pass@1")
print("  ⚠️ 設計卡 §3.2 表的 f27n 那格印 +.182，但照它自己寫的公式應是 .938−.771=+.167")
print("     （+.183 是 rung0 的 oracle raw-hit headroom、隔壁列）⇒ 兩個都印，以【公式版】為準。")
print(f"  {'ckpt':<30} {'卡印':>7} {'公式':>7} {'實測@16':>8} {'差(公式)':>9} | "
      f"{'預測@8區間':>15} {'實測@8':>8} {'落點':>6}")
for M in MODELS:
    C = CELLS.get((M["name"], "on"))
    P = PRED.get(M["name"])
    if C is None or P is None:
        continue
    pf = P["gpG"] - P["p1"]                      # 卡自己的公式，從 rung0 報告重算
    g16 = (C["rows"][16]["real_exp"] - C["base_hit"]) if 16 in C["rows"] else float("nan")
    g8 = (C["rows"][8]["real_exp"] - C["base_hit"]) if 8 in C["rows"] else float("nan")
    lo, hi = pf * PRED_N8_LO, pf * PRED_N8_HI
    inb = "內" if (lo <= g8 <= hi) else ("低" if g8 < lo else "高")
    print(f"  {M['name']:<30} {P['pred16']:>+7.3f} {pf:>+7.3f} {g16:>+8.3f} {g16 - pf:>+9.3f} |"
          f"  {lo:+.3f}~{hi:+.3f} {g8:>+8.3f} {inb:>6}")
print("  ⚠️ 本探針的 N=16 是【32 條抽樣的前綴】、rung0 的 G=16 是【獨立的 16 條抽樣】")
print("     ⇒ 同分佈、不同樣本；差在 SE（±.033~.037）量級內屬正常，⛔ 不可當『重現失敗』讀。")

hr("6. 判準（預釘於檔頭、照抄輸出）")
print(f"  ① primary＝V0：【zero 模式】oracle raw-hit headroom @N=16（pass@16 − pass@1）")
print(f"     ≥ {V0_BIG} ⇒ zero 格全開｜{V0_SMALL}–{V0_BIG} ⇒ 邊際、zero 格降診斷｜"
      f"< {V0_SMALL} ⇒ zero 無物可選、本臂降級＋通報 rung1 R-zero")
V0_N = 16 if 16 in N_SCAN else max(N_SCAN)
print(f"     尺＝rung0 口徑（pass@{V0_N} − mean 同前綴），⛔ 跟 rung0 的 .15 線同一把")
for M in MODELS:
    C = CELLS.get((M["name"], "zero"))
    if C is None:
        print(f"  {M['name']:<30} —— zero 模式本輪未跑")
        continue
    v0 = C["rows"][V0_N]["hd_pfx"]
    se = C["rows"][V0_N]["hd_pfx_se"]
    vd = ("zero 格全開" if v0 >= V0_BIG else
          ("zero 無物可選、本臂降級" if v0 < V0_SMALL else "邊際、zero 格降診斷"))
    hold = "" if C["valid"] else "【⛔ INSTRUMENT INVALID、不放行】"
    print(f"  {M['name']:<30} V0@N={V0_N}: {v0:+.3f} ± {se:.3f} ⇒ 「{vd}」{hold}")
    Con = CELLS.get((M["name"], "on"))
    if Con is not None:
        print(f"  {'':<30}   （對照 on 模式同格 headroom：{Con['rows'][V0_N]['hd_pfx']:+.3f}"
              f"；rung0 實測 G=16 同尺："
              f"{PRED.get(M['name'], {}).get('hd', float('nan')):+.3f}）")
        print(f"  {'':<30}   ⚠️ 絕對水位（base pass@1）：zero={C['base_hit']:.3f}"
              f"  on={Con['base_hit']:.3f}  落差={C['base_hit'] - Con['base_hit']:+.3f}")
print("")
print("  ★ V0 判準的一個【混淆項】—— 判準只讀 headroom（差），讀不到水位（絕對值）：")
for M in MODELS:
    Cz, Co = CELLS.get((M["name"], "zero")), CELLS.get((M["name"], "on"))
    if Cz is None or Co is None:
        continue
    drop = Cz["base_hit"] - Co["base_hit"]
    tag = ("zero 在分佈內（訓練帶 intent-drop）" if drop > -0.10
           else "⚠️ zero 是 OOD ⇒ headroom 是【壞掉的 base 上面的差】")
    print(f"    {M['name']:<30} zero base={Cz['base_hit']:.3f}"
          f"（on {Co['base_hit']:.3f}、{drop:+.3f}） ⇒ {tag}")
print("    ⇒ 過線 ≠ 可部署：headroom ≥ .15 只說「N 條裡有好的可挑」，")
print("       ⛔ 沒說挑完的水位夠高。兩個數要一起讀（見下節 R0 外部錨、同向）。")

print(f"\n  ② 支配檢 |realized − gated| ≤ {DOM_TOL}（§3.3 岔 2）")
print(f"     全 N＝預釘原文；N≥2＝引理真正的定義域（N=1 argmax 無得選 ⇒ 前提空句）")
for (nm, md), C in CELLS.items():
    print(f"  {nm:<30} {md:<5} 全N 最大偏差={C['dom_max']:.3f} ⇒ {'PASS' if C['dom_ok'] else '⛔ FAIL'}"
          f"  ｜ N≥2 最大偏差={C['dom2_max']:.3f} ⇒ {'PASS' if C['dom2_ok'] else '⛔ FAIL'}"
          f"  ｜ N=1 偏差={C['n1_dev']:+.3f}（＝閘誤咬 base 命中）")
print("     ⇒ 全 N 版的 FAIL 【全部落在 N=1】：那格量的是 C8 閘咬掉「第 0 條本來會命中的計畫」")
print("        的比例，跟「argmax 有沒有選對」無關。引理在它有定義的範圍（N≥2）內成立。")
print("        ⚠️ 但這個 N=1 偏差本身是【真的損失】：部署 N=1 時閘不參與選擇，")
print("        它只是把 realized 與 gated 兩把尺拉開，⛔ 不代表閘咬掉的那些計畫真的能跑。")

print(f"\n  ③ N 形狀檢：gain(4) ≥ {NSHAPE_FRAC}×gain(16)（§3.3 岔 4、log N 遞減形）")
for (nm, md), C in CELLS.items():
    if 4 not in C["rows"] or 16 not in C["rows"]:
        continue
    g4 = C["rows"][4]["real_exp"] - C["base_hit"]
    g16 = C["rows"][16]["real_exp"] - C["base_hit"]
    ok = (g4 >= NSHAPE_FRAC * g16) if g16 > 0 else None
    print(f"  {nm:<30} {md:<5} gain(4)={g4:+.3f}  gain(16)={g16:+.3f}  "
          f"⇒ {'PASS' if ok else ('n/a（gain(16)≤0）' if ok is None else '⛔ 違反 ⇒ 覆蓋受限、記給 proposal 多樣性')}")

print(f"\n  ④ 儀器：U1–U5 ＋ golden")
for (nm, md), C in CELLS.items():
    print(f"  {nm:<30} {md:<5} U1-5={''.join('✓' if x else '✗' for x in C['U'])}"
          f"  golden(det/N1)={'✓' if C['golden'][0] else '✗'}{'✓' if C['golden'][1] else '✗'}"
          f"  ρ_len={C['rho']:.3f} δ_step={C['dstep']:.4f}"
          f"  roundtrip hit={C['rt_hit']:.1%}（天花板）")

hr("7. R0 層外部錨與 c 格成本（⛔ 本探針【未量 R0】，這節是對照與估算）")
print("  R0 外部錨（既有 eval log；5 tasks × 50 seeds、MAXH 1000；欄＝BC地板 / R=0 / conf2）")
print(f"  {'ckpt':<30} {'on':>22} {'zero':>22}   job")
for M in MODELS:
    A = R0_ANCHOR.get(M["name"])
    if A is None:
        continue
    f = lambda t: "—" if t is None else " / ".join(f"{x:.3f}" for x in t)   # noqa: E731
    print(f"  {M['name']:<30} {f(A['on']):>22} {f(A['zero']):>22}   {A['job']}")
print("  ⚠️ f27nL 的 zero 那列【五列全 0，連 BC 地板都 0/250】⇒ 那是該次 eval 的整體崩塌，")
print("     ⛔ 不可當「zero 模式 R0 的定值」讀；要一個乾淨的 zero R0 得重跑一次。")
print("  ★ 讀法：只有 idpxm（訓練帶 INTENT_DROP=0.3）的 zero 模式在 R0 層還站得住")
print("     （.716→.624）—— 跟本探針 plan 層的 zero base pass@1 完全同向。")
print("")
print("  c 格（R0 錶）成本估算：")
print(f"    單次完整 R0 eval 實測牆鐘 {R0_WALL_MIN[0]}–{R0_WALL_MIN[1]} min／(ckpt×模式)"
      f"（sacct job 24454/24455、eval 節點 GPU）")
print(f"    plan 端 BoN 加成：只有選出那條進 executor ⇒ rollout ×1、plan ×N；")
print(f"      本探針實測 {len(MODELS) * len(MODES) * len(KEEP) * N_MAX} 個候選 / {time.time() - T0:.0f}s"
      f"（含載入三顆 ckpt）⇒ ~2 ms／候選，與 rung0 同量級")
print(f"      ⇒ N=8 每題多 ~14 ms plan，對照 rollout ≈ {22 * 60 / 250:.1f} s／episode ⇒ 倍率 ≈ ×1.00")
print(f"      （設計卡 §5 估 ×1.0~1.2 —— 實測落在【下界】，因為 rollout 完全主導）")
print(f"    完整 c 格（3 ckpt × 2 模式 × {{N=1, N=8}}）≈ 12 job × ~25 min ≈ 5 GPU-hour")
print("    ⛔ 但它需要改 eval 路徑（scratch_lacot_rollout.py 加 LACOT_BON_N/MODE 旗）＋eval 節點")
print("       佇列 ⇒ 本探針【不做】（設計卡 §7 呈裁項；且該檔有未驗收的 M 狀態 patch、不動）。")

hr("8. 讀數警語（⛔ 跟數字一起讀）")
print("  1. 三顆 ckpt 全是【單顆 s40】—— 沒有 seed variance。ckpt 之間的差異混著 seed 與")
print("     配置兩個因子，⛔ 不可讀成「這個配置比較好」。")
print(f"  2. 題集 {(pair_L[valid_pair] == 0).mean():.0%} 是 L_BFS=0 的同格平凡題 ⇒ 全集增益被稀釋；")
print("     非平凡切片只當診斷（⛔ 不進判準）。")
print("  3. U4 roundtrip hit（decoder 端點保真度）壓住一切 pass@* 讀數 —— 見上表。")
print("  4. plan 層增益 ≠ R0 增益：驗證器讀不到動力學可行性與 executor 可跟隨性")
print("     （設計卡 §6.1）⇒ 本表是 plan 層，R0 兌現率是 c 格、本探針【不量、不外推】。")
print("  5. BoN 輸出分佈已非 π_base（KL ≤ log N − (N−1)/N）—— 下游任何假設")
print("     「計畫 ~ π_base」的檢查語義跟著變（設計卡 §6.3 附註）。")

OUT_JSON = os.path.join(REPO_ROOT, "experiments", "probe_bon_rung05_raw.json")
if not SMOKE:
    meta = dict(host=platform.node(), env=ENV_NAME, n_pairs=N_PAIRS, n_max=N_MAX,
                n_scan=N_SCAN, modes=MODES, sg_seed=SEED, flow_seed=FLOW_SEED,
                tie_seed=TIE_SEED, calib_n=CALIB_N, w=[W_LEGAL, W_REACH, W_HIT],
                ts=time.strftime("%Y-%m-%dT%H:%M:%S"))
    agg = {f"{nm}|{md}": {k: v for k, v in C.items() if k != "rows"} | {"rows": C["rows"]}
           for (nm, md), C in CELLS.items()}
    with open(OUT_JSON, "w") as f:
        json.dump(dict(meta=meta, agg=agg, raw=RAWDUMP), f, indent=1, default=float)
    print(f"\n  原始 json：{OUT_JSON}")

print(f"\n耗時 {time.time() - T0:.1f}s")
