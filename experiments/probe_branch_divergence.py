"""分支散度探針：量「intent 條件通路是死是活」。

背景：idp（LACOT_INTENT_DROP=0.3 訓的）顆在 eval-time guidance（LACOT_INTENT_GUID_W=2）
下【完全沒有變化】。假說＝intent-invariant 不動點：v(·,intent) 與 v(·,∅) 兩支已經收斂到
幾乎相同 ⇒ guidance 放大的是一個零向量。這支探針直接量那個差。

量什麼：TARFlow 的「每步速度場」＝每個 token 的仿射參數 p_t=(mu_t, alpha_t)
（lacot/nf_head.py:129-143 的 sample 方向）。沿【帶 intent 那一支】生成的 u 軌跡走，
在每個 block×token 步上，用【同一段 u_{<t}】另外算兩份 cond 的參數：

    d_zero = ‖p_int − p_zero‖ / (‖p_zero‖ + 1e-8)     intent 拼零（guidance 的引導方向）
    d_swap = ‖p_int − p_swap‖ / (‖p_zero‖ + 1e-8)     換成別的樣本的錨（「換小抄」）
    d_sg   = ‖p_int − p_sg‖   / (‖p_zero‖ + 1e-8)     換整組 (s,g)＝條件通路的活性對照

⭐ 三個都用同一個分母 ⇒ 直接可比（同尺度）。
⭐ d_sg 是【儀器對照】：d_zero≈0 有兩種讀法 ——「intent 方向死了」跟「flow 根本不看 cond」。
   d_sg 大而 d_zero 小才排得掉第二種；d_sg 也小 ⇒ 診斷不同，本探針如實說。
⭐ 另外拆一層：cond 層的相對差（‖cond_int−cond_zero‖/‖cond_zero‖）—— 分辨塌在
   intent adapter／cond_head（cond 就一樣了）還是塌在 flow（cond 有差、參數沒差）。

⛔ 唯讀：不 import experiments/scratch_lacot_rollout.py（它 import 即跑 2400+ 行主流程）。
   需要的架構與載入方式是讀懂該檔後在這裡重建的，每個決定附 file:line 出處。
⛔ 不訓練、不改任何既有檔、不 sbatch、不上網。

跑法：
    cd ~/Projects/lacot
    OGBENCH_DATA_DIR=$HOME/data/ogbench MUJOCO_GL=osmesa \
    $HOME/venvs/lacot-rocm/bin/python experiments/probe_branch_divergence.py \
        2>&1 | tee experiments/probe_branch_divergence_report.txt
"""
import os
import sys
import time

import numpy as np
import torch
from torch import nn

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
# ⛔ 只 import 純定義模組（無頂層副作用）
from lacot.nf_head import Flow                              # noqa: E402
from lacot.intent import hindsight_intent                   # noqa: E402
from lacot.intent_embed import IntentAdapter                # noqa: E402
from lacot.refine_grad import GeoEnergy                     # noqa: E402

T0 = time.time()
device = torch.device("cpu")
ENV_NAME = os.environ.get("LACOT_ENV", "pointmaze-large-stitch-v0")
OGB_DATA = os.environ.get("OGBENCH_DATA_DIR", os.path.expanduser("~/data/ogbench"))
N_SAMP = int(os.environ.get("LACOT_N_SAMP", 64))
SEED = int(os.environ.get("LACOT_SEED", 0))
PRE = ("ckpt_large-stitch_self_K8_c256_ch4_st8000_T128_ep2_gu_eorecon_ictr_tch0.5_"
       "bt{bt}_emw0.999_wu500_s1from_ite{idp}_dssoft_norf_cd0.1_bci_s40.pt")
CKPTS = {"A f27n (無 dropout)": PRE.format(bt="f27n", idp=""),
         "B idp0.3 (p=0.3)":    PRE.format(bt="idp1", idp="_idp0.3")}

# ─── 預釘判準（⛔ 先寫在這裡，跑完照抄輸出，不事後挑）───────────────────────
CRIT_RATIO = 0.30        # B 的 d_zero 中位 / A 的 d_zero 中位 < 0.3 ⇒「B 兩支已幾乎重合」成立
CRIT_SWAP_FLAT = 0.01    # d_swap 中位 < 0.01 ⇒ 判定「對換小抄沒有反應」
GATE_SG_ALIVE = 0.05     # d_sg 中位 < 0.05 ⇒ 條件通路本身量不出活性 ⇒ 本探針對該顆判讀無效


def hr(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78, flush=True)


def sota_mlp(i, h, o, n=2):
    """出處＝experiments/scratch_lacot_rollout.py:520-526（逐行同構，載權重才對得上）。"""
    L, p = [], i
    for _ in range(n):
        L += [nn.Linear(p, h), nn.GELU(), nn.LayerNorm(h)]
        p = h
    return nn.Sequential(*L, nn.Linear(p, o))


def q(x):
    return np.percentile(np.asarray(x, np.float64), [25, 50, 75])


# ═══ 0. 資料：切窗／正規化逐字對齊主線 ═══════════════════════════════════════
hr("0. 資料（切窗／正規化 file:line 出處）")
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
mu, sd = OBS.mean(0), OBS.std(0) + 1e-6          # rollout.py:38（全資料集，⛔ 不用片段算）
print(f"  env={ENV_NAME}  OBS{OBS.shape}  episodes={len(ends)}  N_SAMP={N_SAMP}  seed={SEED}")

CFG0 = torch.load(os.path.join(REPO_ROOT, "results", list(CKPTS.values())[0]),
                  map_location=device, weights_only=False)["cfg"]
K, COND, D_MODEL, T_CAP, CHUNK = (CFG0["K"], CFG0["COND"], CFG0["D_MODEL"],
                                  CFG0["T_CAP"], CFG0["CHUNK"])
print(f"  ckpt cfg：K={K} COND={COND} D_MODEL={D_MODEL} T_CAP={T_CAP} CHUNK={CHUNK}"
      f"  ENC_OBJ={CFG0['ENC_OBJ']}  EMA_W={CFG0['EMA_W']}")

# 抽窗：rollout.py:465-478 的規則（起點均勻、goal 在 [r+1, te] 均勻後 clamp ≥ r+CHUNK）
rng = np.random.default_rng(SEED)
rows, goals = [], []
while len(rows) < N_SAMP:
    r = int(rng.integers(0, N))
    te = int(traj_end[r])
    if te - r < CHUNK:
        continue
    _d = rng.random()
    gr = int(round(min(r + 1, te) * _d + te * (1 - _d)))
    rows.append(r)
    goals.append(max(gr, min(r + CHUNK, te)))
rows, goals = np.array(rows), np.array(goals)
# 軌跡：rollout.py:497-502 的固定 T_CAP 點線性插值（⛔ 不取整 —— 點數洩漏那個修）
f = np.linspace(rows[:, None].astype(np.float64), goals[:, None].astype(np.float64),
                T_CAP, axis=1).reshape(N_SAMP, T_CAP)
lo_i = np.floor(f).astype(np.int64)
hi_i = np.minimum(lo_i + 1, goals[:, None])
w = (f - lo_i)[..., None]
traj = ((OBS[lo_i] * (1.0 - w) + OBS[hi_i] * w - mu) / sd).astype(np.float32)
S = torch.tensor((OBS[rows] - mu) / sd, dtype=torch.float32)
G = torch.tensor((OBS[goals] - mu) / sd, dtype=torch.float32)
print(f"  窗長（步）p25/p50/p75 = {q(goals - rows).round(0)}")

# 錨（hindsight，＝訓練預設 LACOT_INTENT_SRC=hindsight）：rollout.py:338-384 的
# 佔據圖與 cell↔正規化座標轉換，逐行搬過來（⛔ 正規化空間，不做 mu/sd）。
_ig = GeoEnergy(OBS, mu, sd, res=8, device="cpu")
_iocc = (_ig.dist[0, 0].numpy() == 0.0)
_ifree = np.argwhere(_iocc)
_ilo = np.asarray(_ig.lo, np.float64)
_ispan = np.asarray(_ig.hi - _ig.lo, np.float64)
_ishape = np.asarray(_ig.shape, np.int64)


def _i_zn_to_cell(z):
    idx = np.clip(np.round((np.asarray(z, np.float64)[:2] - _ilo) / _ispan * (_ishape - 1)).astype(int),
                  0, _ishape - 1)
    c = tuple(idx)
    return c if _iocc[c] else tuple(_ifree[int(np.abs(_ifree - idx).sum(1).argmin())])


def _i_cell_to_zn(c):
    return _ilo + np.asarray(c, np.float64) / (_ishape - 1) * _ispan


TA = 32                        # 由 ckpt 的 intent_ad.mlp.0.weight 反推驗證（見 build()）
ANC = torch.from_numpy(np.stack(
    [hindsight_intent(traj[i], _i_zn_to_cell, _i_cell_to_zn, TA)[0] for i in range(N_SAMP)]))
ROLL = torch.roll(torch.arange(N_SAMP), 1)        # 「別的樣本」＝環狀位移 1（配對、可重現）
print(f"  佔據圖 {tuple(_ig.shape)}、錨 T_A={TA}、anchors{tuple(ANC.shape)}"
      f"   swap 對象＝roll(1)（⛔ 不重抽，配對可比）")


# ═══ 1. 重建模型 + 載權重（ema，載法同 eval）═════════════════════════════════
def build(name, fn):
    path = os.path.join(REPO_ROOT, "results", fn)
    assert os.path.exists(path), f"⛔ 找不到 {path}"
    ck = torch.load(path, map_location=device, weights_only=False)
    ta = ck["intent_ad"]["mlp.0.weight"].shape[1] // 2      # ⭐ 從權重反推，⛔ 不猜 env 預設
    assert ta == TA, f"⛔ {name} 的 T_A={ta} ≠ {TA}"
    ad = IntentAdapter(TA, K)
    enc = sota_mlp(2, 512, 512)
    head = sota_mlp(1024 + ad.cond_extra_dim, 512, COND)    # rollout.py:967-969
    fl = Flow(token_dim=D_MODEL, seq_len=K, n_blocks=4, cond_dim=COND)   # rollout.py:971-972
    for k, m in (("cond_enc", enc), ("cond_head", head), ("flow", fl), ("intent_ad", ad)):
        m.load_state_dict(ck[k])
    # ⭐ LOAD_EMA=1 的語義：影子權重覆蓋供點鏈（rollout.py:1319-1326）
    assert "ema" in ck, f"⛔ {name} 沒有 ema 段"
    for k, sdct in ck["ema"].items():
        if k in ("cond_enc", "cond_head", "flow", "intent_ad"):
            {"cond_enc": enc, "cond_head": head, "flow": fl, "intent_ad": ad}[k].load_state_dict(sdct)
    for m in (enc, head, fl, ad):
        m.eval()
        for p in m.parameters():
            p.requires_grad_(False)
    print(f"  {name}: 載入 4 模組 + ema 覆蓋 {sorted(set(ck['ema']) & {'cond_enc','cond_head','flow','intent_ad'})}"
          f"   ({os.path.basename(fn)})")

    def condvec(s, g, ix=None):
        """出處＝rollout.py:1032-1048（embed 分支）：ix=None ⇒ 尾巴拼零。"""
        x = torch.cat([enc(s), enc(g)], 1)
        if ix is None:
            ix = x.new_zeros(x.shape[0], ad.cond_extra_dim)
        return head(torch.cat([x, ix], 1))

    return dict(name=name, flow=fl, condvec=condvec, icond=ad.cond_global)


hr("1. 重建模型並載入 EMA 權重")
MODELS = [build(n, fn) for n, fn in CKPTS.items()]


# ═══ 2. 沿 int 支的取樣軌跡，收每個 block×token 步的四份參數 ═════════════════
@torch.no_grad()
def branch_params(flow, conds, z):
    """⛔ 主幹（u 的更新）只用 conds['int'] ⇒ 走的就是正常帶 intent 的取樣軌跡；
    其餘分支在【同一段 u_{<t}】上另算一次參數（＝guidance 在 w≈1 附近會看到的狀態）。
    逆序／perm／flip 同步全照 Flow.sample（nf_head.py:219-237），flip 用 flow._cond_for_block。"""
    B = z.shape[0]
    u = z
    rec = {k: [] for k in conds}
    blk_of = []
    for i in reversed(range(len(flow.blocks))):
        if i < len(flow.blocks) - 1:
            u = flow.perm.inverse(u)
        blk = flow.blocks[i]
        pre = {k: blk._prefix(B, flow._cond_for_block(c, i), u.device) for k, c in conds.items()}
        z_in, u_new = u, torch.zeros_like(u)
        for t in range(z_in.shape[1]):
            e = None if t == 0 else blk.embed(u_new[:, :t])
            for k in conds:
                seq = pre[k] if t == 0 else torch.cat([pre[k], e], dim=1)
                m_, a_ = blk._params(seq)
                rec[k].append(torch.cat([m_[:, -1], a_[:, -1]], -1))     # [B, 2*D]
                if k == "int":
                    u_new[:, t] = z_in[:, t] * torch.exp(a_[:, -1]) + m_[:, -1]
            blk_of.append(i)
    return {k: torch.stack(v) for k, v in rec.items()}, np.array(blk_of)   # [Steps,B,2D]


hr("2. 逐 block×token 步的分支散度")
torch.manual_seed(20260905)
Z = torch.randn(N_SAMP, K, D_MODEL)          # ⭐ 同一份 z 給兩顆 ⇒ 配對比較
RES = {}
for M in MODELS:
    ix = M["icond"](ANC)
    conds = {"int":  M["condvec"](S, G, ix),
             "zero": M["condvec"](S, G, None),
             "swap": M["condvec"](S, G, ix[ROLL]),
             "sg":   M["condvec"](S[ROLL], G[ROLL], ix[ROLL])}
    P, blk_of = branch_params(M["flow"], conds, Z)
    den = P["zero"].norm(dim=-1) + 1e-8                       # [Steps,B]
    d = {k: ((P["int"] - P[k]).norm(dim=-1) / den).numpy() for k in ("zero", "swap", "sg")}
    cl = ((conds["int"] - conds["zero"]).norm(dim=-1) / (conds["zero"].norm(dim=-1) + 1e-8)).numpy()
    RES[M["name"]] = dict(d=d, blk=blk_of, condrel=cl)
    print(f"\n  【{M['name']}】 steps={P['int'].shape[0]}（4 blocks × {K} tokens）× {N_SAMP} 樣本")
    print(f"    cond 層  ‖cond_int−cond_zero‖/‖cond_zero‖  中位 {np.median(cl):.4f}"
          f"   (p25/p75 {q(cl)[0]:.4f}/{q(cl)[2]:.4f})")
    for k, lab in (("zero", "d_zero  intent→拼零 "), ("swap", "d_swap  換別人的錨"),
                   ("sg", "d_sg    換整組(s,g) ")):
        a, b, c = q(d[k].ravel())
        print(f"    {lab}  中位 {b:.5f}   p25/p75 {a:.5f}/{c:.5f}   max {d[k].max():.5f}")
    print("    逐 block（取樣逆序 3→0；括號＝該 block 的 d_zero / d_swap / d_sg 中位）：")
    for i in reversed(range(4)):
        m = blk_of == i
        print(f"      block {i}:  {np.median(d['zero'][m]):.5f} / "
              f"{np.median(d['swap'][m]):.5f} / {np.median(d['sg'][m]):.5f}")


# ═══ 3. 判準（預釘、照抄）══════════════════════════════════════════════════
hr("3. 判準（預釘於腳本頂端，數值照抄）")
nA, nB = [M["name"] for M in MODELS]
mzA, mzB = np.median(RES[nA]["d"]["zero"]), np.median(RES[nB]["d"]["zero"])
msA, msB = np.median(RES[nA]["d"]["swap"]), np.median(RES[nB]["d"]["swap"])
gA, gB = np.median(RES[nA]["d"]["sg"]), np.median(RES[nB]["d"]["sg"])
ratio = mzB / (mzA + 1e-12)
print(f"  gate（儀器活性）d_sg 中位 ≥ {GATE_SG_ALIVE}：{nA} {gA:.5f} "
      f"{'通過' if gA >= GATE_SG_ALIVE else '未通過'}   |   {nB} {gB:.5f} "
      f"{'通過' if gB >= GATE_SG_ALIVE else '未通過'}")
if min(gA, gB) < GATE_SG_ALIVE:
    print("  ⚠️ 有一顆的條件通路本身量不出活性 ⇒ 該顆的 d_zero 小【不能】單獨讀成"
          "「intent 方向死了」（也可能是 flow 整體不看 cond）。下面的判準對該顆判讀無效。")
print(f"\n  判準①  d_zero 中位比值 B/A < {CRIT_RATIO}")
print(f"          A={mzA:.5f}   B={mzB:.5f}   比值 {ratio:.4f}   ⇒ "
      f"{'符合（B 兩支已幾乎重合）' if ratio < CRIT_RATIO else '不符合（兩顆的分支散度相近）'}")
print(f"  判準②  d_swap 中位 < {CRIT_SWAP_FLAT} ⇒ 判「對換錨沒有反應」")
print(f"          A={msA:.5f} {'符合' if msA < CRIT_SWAP_FLAT else '不符合'}   |   "
      f"B={msB:.5f} {'符合' if msB < CRIT_SWAP_FLAT else '不符合'}")
print(f"\n  結論（只陳述量到的）：judgement = "
      f"{'①成立' if ratio < CRIT_RATIO else '①不成立'}；"
      f"B 的 intent 分支散度為 A 的 {ratio * 100:.1f}%；"
      f"B 對換錨的反應中位 {msB:.5f}（A {msA:.5f}）。")
print(f"\n  cond 層 vs 參數層（塌在哪一段）：")
for M in MODELS:
    cl = np.median(RES[M["name"]]["condrel"])
    dz = np.median(RES[M["name"]]["d"]["zero"])
    print(f"    {M['name']}:  cond 相對差 {cl:.4f}  →  參數相對差 {dz:.5f}"
          f"   （比值 {dz / (cl + 1e-12):.4f}）")
print(f"\n耗時 {time.time() - T0:.1f}s")
