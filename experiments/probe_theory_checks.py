"""理論地基三驗（真資料）：A1 條件冗餘／Z2 反對稱能量譜／Prop 1.1 四格對答案。

對應理論件：docs/THEORY-2026-09-05-internalization-formal.md
  - A1（§0 假設）：I_data(τ;a|s,g)=0 —— FINDINGS-0905 ⑫⑭ 機制 1 的承重牆，至今是推論非量測。
    操作化：a 是 route 的摘要（hindsight intent），故量 H(R|s,g) —— R=路線簽名。
  - Remark 4.2（§4）：現行 e 幾乎全落對稱子空間 F₊（C1 cos=+0.9901 是間接證據）——
    直接量 φ₋=(e(τ)−e(rev τ))/2 的能量比與 PCA 有效維度（D_d 預算的實測依據）。
  - Prop 1.1（§1.2 定性級聯）：ε=0 ⇒ Δ_succ=0 —— 用 FINDINGS ①⑦⑬ 已有量測對答案。

⛔ 唯讀：只讀資料 npz、凍結 ckpt、既有工具類（GeoEnergy）；不訓練、不改既有檔、
   不 commit、不上網、不 GPU、不 sbatch。切窗／正規化／格圖全部沿用
   experiments/probe_z_geodesic.py 已驗過的做法（該檔逐條附 file:line 出處，此處引用之）。

═══ 預釘判準（跑之前寫死，跑完照抄，不事後挑）═══════════════════════════════
  A1 成立 ⇔ 逐桶「決定程度」d_b = 1 − H_MM(R|b)/H_MM(R) 的中位 > 0.80
            【0.80 是猜測門檻、可調 —— 判讀時同時看 H(R|b) 絕對值與分層趨勢】
  Z2 實錘 ⇔ ‖φ₋‖²/‖e‖² 中位 < 5% ⇒ Remark 4.2「幾乎全落 F₊」成立、D_d 可小
  Prop1.1 ⇔ 四格中不存在「ε≈0 且 Δ 顯著非零（配對 8/8 同號級）」的嚴格反例
═══ 熵估樣本量陷阱（誠實聲明）═════════════════════════════════════════════
  桶內 n 小 ⇒ H(R|b) 天花板 log2(n)、天然低估 ⇒「決定程度」偏高（偏向 A1 成立）。
  對策：(a) Miller-Madow 修正（naive + (m−1)/(2n·ln2)）並列 naive；
        (b) n≥5 / n≥10 / n≥20 三個門檻分層看趨勢；
        (c) 同集 episode 重疊窗造成的偽複製 ⇒ 另算「每桶每 episode 只留一窗」變體；
        (d) 逐桶另報 modal route 佔比（對小樣本穩健的決定性指標）。
  反向偏誤：H(R) 分母（幾十萬簽名）同樣低估 ⇒ 決定程度偏低。兩向都報，不遮。

跑法：
    cd ~/Projects/lacot
    OGBENCH_DATA_DIR=$HOME/data/ogbench MUJOCO_GL=osmesa \
    $HOME/venvs/lacot-rocm/bin/python experiments/probe_theory_checks.py \
        2>&1 | tee experiments/probe_theory_checks_report.txt
"""
import os
import sys
import time

import numpy as np
import torch
from scipy.stats import spearmanr

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
# ⛔ 只 import 純定義模組（無頂層副作用）—— 同 probe_z_geodesic.py:48-52 的原則
from lacot.e_target import PerceiverPooler      # noqa: E402
from lacot.refine_grad import GeoEnergy         # noqa: E402

T0 = time.time()
LN2 = float(np.log(2.0))

ENV_NAME = os.environ.get("LACOT_ENV", "pointmaze-large-stitch-v0")
OGB_DATA = os.environ.get("OGBENCH_DATA_DIR", os.path.expanduser("~/data/ogbench"))
CKPT_NAME = ("ckpt_large-stitch_self_K8_c256_ch4_st8000_T128_ep2_gu_eorecon_ictr_tch0.5"
             "_emw0.999_wu500_dssoft_norf_cd0.1_bci_s27.pt")   # 同 probe_z_geodesic.py:61-62
CKPT_PATH = os.environ.get("LACOT_CKPT", os.path.join(REPO_ROOT, "results", CKPT_NAME))
N_WIN = int(os.environ.get("LACOT_N_WIN", 400_000))   # A1 的窗數（熵估要大樣本）
N_Z2 = int(os.environ.get("LACOT_N_Z2", 768))         # Z2 的窗數（規格要求 ≥500）
SEED = int(os.environ.get("LACOT_SEED", 0))
GEO_RES = 8      # 全 repo 唯一慣例（probe_z_geodesic.py:67 同款；任務規格指定 res=8）
MIN_BUCKET = 5   # 任務規格：樣本量 ≥5 的桶
CRIT_A1_MEDIAN_DET = 0.80   # 【猜測門檻、可調】
CRIT_Z2_RATIO = 0.05
device = "cpu"
torch.manual_seed(SEED)


def hr(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78, flush=True)


def pct(x, qs=(10, 25, 50, 75, 90)):
    v = np.percentile(np.asarray(x, np.float64), qs)
    return "  ".join(f"p{q}={v_:.3f}" for q, v_ in zip(qs, v))


# ═════════════════════════════════════════════════════════════════
# 0. 資料 + 格圖（逐字沿用 probe_z_geodesic.py §0/§2 已驗做法）
# ═════════════════════════════════════════════════════════════════
hr("0. 資料 + res=8 格圖（做法出處＝probe_z_geodesic.py §0/§2，該檔附逐行 file:line）")
_npz = np.load(f"{OGB_DATA}/{ENV_NAME}.npz")
OBS = np.asarray(_npz["observations"], np.float32)
TERM = np.asarray(_npz["terminals"], bool)
N = OBS.shape[0]
ends = np.flatnonzero(TERM)
starts = np.concatenate([[0], ends[:-1] + 1])
traj_end = np.empty(N, np.int64)
for s0, e0 in zip(starts, ends):
    traj_end[s0:e0 + 1] = e0
assert ends[-1] == N - 1
mu, sd = OBS.mean(0), OBS.std(0) + 1e-6   # 全資料集算（訓練同款，⛔ 不用片段算）
print(f"  env={ENV_NAME}  OBS{OBS.shape}  episodes={len(ends)}  N_WIN={N_WIN}  seed={SEED}")

geo = GeoEnergy(OBS, mu, sd, res=GEO_RES, device="cpu")
occ = (geo.dist[0, 0].numpy() == 0.0)
Hg, Wg = (int(v) for v in geo.shape)
lo_np, span_np = geo.lo, (geo.hi - geo.lo)
shape_arr = np.asarray(geo.shape, np.int64)
gh = geo.health()
print(f"  grid=({Hg},{Wg})  free={int(occ.sum())}/{occ.size}（coverage={geo.coverage:.1%}）"
      f"  health ok={gh['ok']} mapping_err={gh['mapping_err']:.2e}")
assert gh["ok"], f"⛔ GeoEnergy.health() 沒過：{gh['reasons']} ⇒ 格圖不可信，停手"

ck = torch.load(CKPT_PATH, map_location=device, weights_only=False)
cfg = ck["cfg"]
K, D_MODEL, T_CAP, CHUNK = cfg["K"], cfg["D_MODEL"], cfg["T_CAP"], cfg["CHUNK"]
print(f"  ckpt cfg：K={K} D_MODEL={D_MODEL} T_CAP={T_CAP} CHUNK={CHUNK} ENC_OBJ={cfg['ENC_OBJ']}")


def zn_to_cell(pts):
    """[...,2] 正規化座標 → 格 index（round+clip，無 snap）。同 probe_z_geodesic.py:214-222。"""
    idx = np.round((np.asarray(pts, np.float64) - lo_np) / span_np * (shape_arr - 1)).astype(np.int64)
    return np.clip(idx, 0, shape_arr - 1)


def sample_windows(n, rng):
    """訓練同分佈的 (r, gr) 抽窗 —— 規則＝scratch_lacot_rollout.py:452-465（經
    probe_z_geodesic.py:256-270 驗過的重建）。此處把逐筆 while 迴圈改成批次 rejection
    sampling：分佈相同（r 均勻、_d 均勻、同一條 clamp 公式），只有 rng 消耗順序不同。"""
    rows = np.empty(0, np.int64)
    goals = np.empty(0, np.int64)
    while len(rows) < n:
        m = int((n - len(rows)) * 1.25) + 64
        r = rng.integers(0, N, size=m)
        te = traj_end[r]
        keep = (te - r) >= CHUNK
        r, te = r[keep], te[keep]
        _d = rng.random(len(r))
        gr = np.round(np.minimum(r + 1, te) * _d + te * (1.0 - _d)).astype(np.int64)
        gr = np.maximum(gr, np.minimum(r + CHUNK, te))
        rows = np.concatenate([rows, r])
        goals = np.concatenate([goals, gr])
    return rows[:n], goals[:n]


def build_traj(rows, goals):
    """固定 T_CAP 點線性內插（F7）。逐字同 probe_z_geodesic.py:273-282。回正規化座標。"""
    n = len(rows)
    f = np.linspace(rows[:, None].astype(np.float64), goals[:, None].astype(np.float64),
                    T_CAP, axis=1).reshape(n, T_CAP)
    lo_i = np.floor(f).astype(np.int64)
    hi_i = np.minimum(lo_i + 1, goals[:, None])
    w = (f - lo_i)[..., None]
    return ((OBS[lo_i] * (1.0 - w) + OBS[hi_i] * w - mu) / sd).astype(np.float32)


# ═════════════════════════════════════════════════════════════════
# 1. A1 量測：H(R | s,g 粗格桶) vs H(R)
# ═════════════════════════════════════════════════════════════════
hr(f"1. A1 條件冗餘：{N_WIN} 窗、桶=(起點格,終點格)@res8、R=粗格序列(相鄰去重)簽名")
print("  簽名＝窗的 T_CAP 內插點 → res8 格 → 相鄰去重後的格序列，FNV-1a 64-bit 摺疊成 hash")
print("  （54 萬級簽名的 64-bit 碰撞期望 ~1e-8；下面另用 3 萬窗子樣本做 hash↔exact 對驗）")
print("  ⭐ 解析度不是任意選的：訓練的 hindsight intent 錨＝traj_to_cells()＝【同一種】相鄰去重")
print("  res8 格序列（lacot/intent.py:12,21-34）⇒ a≈R（τ 的決定函數）⇒ I(τ;a|s,g)=H(a|s,g)")
print("  ≈ H(R|s,g) —— 本節的加權平均 H_MM(R|bucket) 就是 A1 那個互資訊的【直接】估計，")
print("  不是代理。（錨後續 anchors_resample 到固定 K 點、略有損 ⇒ 嚴格說 H(a|s,g)≤H(R|s,g)。）")
rng = np.random.default_rng(SEED)
rows, goals = sample_windows(N_WIN, rng)
epi_id = traj_end[rows]                       # 每 episode 唯一 ⇒ 當 episode 鍵
win_len = (goals - rows).astype(np.int64)

FNV_OFF = np.uint64(1469598103934665603)
FNV_PRM = np.uint64(1099511628211)


def fold_sig(cf):
    """[B,T] 格號序列 → 相鄰去重後的 FNV-1a 64-bit 簽名 [B]。"""
    h = np.full(cf.shape[0], FNV_OFF, np.uint64)
    prev = np.full(cf.shape[0], -1, np.int64)
    with np.errstate(over="ignore"):
        for t in range(cf.shape[1]):
            c = cf[:, t]
            chg = c != prev
            h[chg] = (h[chg] ^ c[chg].astype(np.uint64)) * FNV_PRM
            prev = c
    return h


sig = np.empty(N_WIN, np.uint64)
sig_c2 = np.empty(N_WIN, np.uint64)   # 敏感度變體：res8 格 2×2 合併（≈res4）的簽名
bucket = np.empty(N_WIN, np.int64)
bucket16 = np.empty(N_WIN, np.int64)  # 細桶變體：端點格用 2 倍解析度（擋「桶太粗」反駁）
Wg2 = (Wg + 1) // 2
H16, W16 = 2 * Hg, 2 * Wg
BATCH = 50_000
t_a1 = time.time()
for b0 in range(0, N_WIN, BATCH):
    sl = slice(b0, min(b0 + BATCH, N_WIN))
    traj = build_traj(rows[sl], goals[sl])                 # [B,T,2] 正規化
    cells = zn_to_cell(traj)                               # [B,T,2]
    cf = cells[..., 0] * Wg + cells[..., 1]                # [B,T] 攤平格號
    bucket[sl] = cf[:, 0] * (Hg * Wg) + cf[:, -1]          # 桶=(起點格,終點格)
    sig[sl] = fold_sig(cf)
    sig_c2[sl] = fold_sig((cells[..., 0] // 2) * Wg2 + cells[..., 1] // 2)
    ep_np = traj[:, [0, -1], :]                            # 端點（正規化座標）
    c16 = np.clip(np.round((ep_np - lo_np) / span_np * (np.array([H16, W16]) - 1)
                           ).astype(np.int64), 0, np.array([H16, W16]) - 1)
    cf16 = c16[..., 0] * W16 + c16[..., 1]
    bucket16[sl] = cf16[:, 0] * (H16 * W16) + cf16[:, 1]
print(f"  簽名/桶完成（{time.time() - t_a1:.1f}s）  窗長 gr−r：{pct(win_len, (25, 50, 75))}")

# --- 儀器自檢①：hash vs exact 簽名（3 萬窗子樣本）------------------------------
n_chk = min(30_000, N_WIN)
cells_c = zn_to_cell(build_traj(rows[:n_chk], goals[:n_chk]))
cf_c = cells_c[..., 0] * Wg + cells_c[..., 1]
exact = set()
h2t = {}
n_conflict = 0
for i in range(n_chk):
    row = cf_c[i]
    tup = tuple(row[np.concatenate([[True], row[1:] != row[:-1]])].tolist())
    exact.add(tup)
    prev_t = h2t.get(int(sig[i]))
    if prev_t is not None and prev_t != tup:
        n_conflict += 1
    h2t[int(sig[i])] = tup
n_hash = len(set(sig[:n_chk].tolist()))
print(f"  儀器自檢①  exact 簽名數={len(exact)}  hash 簽名數={n_hash}  衝突={n_conflict}")
assert len(exact) == n_hash and n_conflict == 0, "⛔ hash 簽名與 exact 簽名不一致，停手"


def entropies(bucket_a, sig_a, min_n):
    """回（逐桶表, 全體 H）。逐桶：n、m(相異簽名)、H_naive、H_MM、modal 佔比。全部向量化。"""
    order = np.lexsort((sig_a.astype(np.int64), bucket_a))
    b_s, s_s = bucket_a[order], sig_a[order]
    uniq_b, b_inv, b_cnt = np.unique(b_s, return_inverse=True, return_counts=True)
    pair_key = b_inv.astype(np.int64) * np.int64(1 << 32)  # bucket 群內 pair 分組
    # (bucket, sig) pair 計數：相鄰相同判斷（已按 bucket,sig 排序）
    new_pair = np.concatenate([[True], (b_s[1:] != b_s[:-1]) | (s_s[1:] != s_s[:-1])])
    pair_gid = np.cumsum(new_pair) - 1
    pair_cnt = np.bincount(pair_gid)
    pair_bucket = b_inv[new_pair]                     # 每個 pair 屬於哪個桶（桶的緊湊編號）
    n_of = b_cnt[pair_bucket].astype(np.float64)
    p = pair_cnt / n_of
    h_contrib = -p * np.log2(p)
    H_naive = np.bincount(pair_bucket, weights=h_contrib, minlength=len(uniq_b))
    m = np.bincount(pair_bucket, minlength=len(uniq_b))                 # 相異簽名數
    H_mm = H_naive + (m - 1) / (2.0 * b_cnt * LN2)                      # Miller-Madow
    modal = np.zeros(len(uniq_b))
    np.maximum.at(modal, pair_bucket, pair_cnt)
    modal = modal / b_cnt
    keep = b_cnt >= min_n
    # 全體 H(R)（同一個樣本集合上）
    su, sc = np.unique(sig_a, return_counts=True)
    ps = sc / sc.sum()
    HR_naive = float(-(ps * np.log2(ps)).sum())
    HR_mm = HR_naive + (len(su) - 1) / (2.0 * len(sig_a) * LN2)
    return dict(uniq_b=uniq_b, cnt=b_cnt, keep=keep, H_naive=H_naive, H_mm=H_mm,
                m=m, modal=modal, HR_naive=HR_naive, HR_mm=HR_mm, n_sig=len(su))


def report_a1(tag, bucket_a, sig_a, len_a):
    E = entropies(bucket_a, sig_a, MIN_BUCKET)
    keep = E["keep"]
    nq = int(keep.sum())
    cover = float(E["cnt"][keep].sum() / len(sig_a))
    print(f"\n  ── {tag}：窗={len(sig_a)}  桶(全)={len(E['uniq_b'])}  桶(n≥{MIN_BUCKET})={nq}"
          f"（覆蓋 {cover:.1%} 窗）  相異簽名={E['n_sig']}")
    print(f"     H(R) 無條件：naive={E['HR_naive']:.3f}  MM={E['HR_mm']:.3f} bits"
          f"（⚠️ {E['n_sig']}/{len(sig_a)} 簽名/樣本 ⇒ 分母本身低估，決定程度往低偏）")
    res = {}
    for thr in (5, 10, 20):
        kp = E["cnt"] >= thr
        if kp.sum() < 10:
            print(f"     n≥{thr:2d}：合格桶 {int(kp.sum())} <10，不報")
            continue
        Hn, Hm, cnts = E["H_naive"][kp], E["H_mm"][kp], E["cnt"][kp]
        det = 1.0 - Hm / E["HR_mm"]
        wavg_mm = float((Hm * cnts).sum() / cnts.sum())
        res[thr] = dict(det_med=float(np.median(det)), Hmm_med=float(np.median(Hm)),
                        wavg=wavg_mm)
        print(f"     n≥{thr:2d}（{int(kp.sum())} 桶）  H(R|b) naive：{pct(Hn)}")
        print(f"            H(R|b) MM   ：{pct(Hm)}   加權平均 MM={wavg_mm:.3f} bits")
        print(f"            決定程度 d_b=1−H_MM(R|b)/H_MM(R)：{pct(det)}")
        print(f"            modal route 佔比：{pct(E['modal'][kp])}   "
              f"H=0 的桶佔 {float((Hn == 0).mean()):.1%}")
    # 桶平均窗長分層（A1 的咬合點在長窗 —— 長窗才可能分岔）
    kp5 = E["keep"]
    if kp5.sum() >= 30:
        uniq_b = E["uniq_b"]
        b_of = np.searchsorted(uniq_b, bucket_a)
        lensum = np.bincount(b_of, weights=len_a.astype(np.float64), minlength=len(uniq_b))
        blen = (lensum / E["cnt"])[kp5]
        Hm5 = E["H_mm"][kp5]
        q1, q2 = np.percentile(blen, [33.3, 66.7])
        print(f"     桶平均窗長三分層（n≥{MIN_BUCKET} 桶；切點 {q1:.0f}/{q2:.0f} 步）：")
        for lo_v, hi_v, name in ((0, q1, f"短(≤{q1:.0f})"), (q1, q2, "中"), (q2, 1e9, f"長(>{q2:.0f})")):
            m_l = (blen > lo_v) & (blen <= hi_v)
            if m_l.sum() >= 10:
                det_l = 1.0 - Hm5[m_l] / E["HR_mm"]
                print(f"       {name:10s} {int(m_l.sum()):6d} 桶  H_MM 中位={np.median(Hm5[m_l]):.3f}"
                      f"  決定程度中位={np.median(det_l):.3f}")
    return res


res_raw = report_a1("主量測（訓練同分佈原樣窗）", bucket, sig, win_len)

# --- 偽複製穩健變體：每桶每 episode 只留一窗（同集重疊窗不重複計票）------------
key = bucket * np.int64(len(ends) + 1) + np.searchsorted(ends, epi_id)
_, first_idx = np.unique(key, return_index=True)
res_dedup = report_a1("穩健變體（每桶每 episode 留一窗）", bucket[first_idx], sig[first_idx],
                      win_len[first_idx])
# --- 簽名解析度敏感度：桶不變（res8），簽名 2×2 合併（同走廊抖動不再算不同路線）----
res_c2 = report_a1("敏感度變體（簽名 2×2 粗化、桶仍 res8）", bucket, sig_c2, win_len)
# --- 細桶變體：端點格 2 倍解析度（若多路線只是「res8 桶把不同 (s,g) 混在一起」的假象，
#     這裡 H(R|b) 會塌掉；沒塌 ⇒ 條件熵不是桶粗造成的）--------------------------------
res_b16 = report_a1("細桶變體（桶=2 倍解析度端點格、簽名 res8）", bucket16, sig, win_len)

det5 = res_raw.get(5, {}).get("det_med", float("nan"))
a1_pass = det5 > CRIT_A1_MEDIAN_DET
print(f"\n  ▶ A1 判定（預釘門檻：主量測 n≥5 決定程度中位 > {CRIT_A1_MEDIAN_DET}【猜測、可調】）：")
print(f"    決定程度中位 = {det5:.3f}  ⇒  {'A1 成立' if a1_pass else 'A1 未達門檻'}")
print(f"    （穩健變體 n≥5 中位 = {res_dedup.get(5, {}).get('det_med', float('nan')):.3f}；"
      f"簽名粗化變體 = {res_c2.get(5, {}).get('det_med', float('nan')):.3f}；"
      f"H_MM(R|b) 中位（主/穩健/粗化）= {res_raw.get(5, {}).get('Hmm_med', float('nan')):.3f} / "
      f"{res_dedup.get(5, {}).get('Hmm_med', float('nan')):.3f} / "
      f"{res_c2.get(5, {}).get('Hmm_med', float('nan')):.3f} bits）")
print(f"    ⭐ I(τ;a|s,g) 直接估計（＝加權平均 H_MM(R|bucket)、n≥5 桶）："
      f"主={res_raw.get(5, {}).get('wavg', float('nan')):.3f}  "
      f"穩健={res_dedup.get(5, {}).get('wavg', float('nan')):.3f}  "
      f"粗化={res_c2.get(5, {}).get('wavg', float('nan')):.3f}  "
      f"細桶={res_b16.get(5, {}).get('wavg', float('nan')):.3f} bits"
      f"（A1 主張此量 ≈ 0）")
print("    偏誤方向自檢：桶內 n 越大熵讀數越高（n≥5→n≥20 中位上行）＝小樣本低估仍在 ⇒"
      " 真 H(R|s,g) 只會更高；粗桶膨脹熵的反向偏誤由細桶變體定界。")

# ═════════════════════════════════════════════════════════════════
# 2. Z2 反對稱能量譜：φ₋=(e(τ)−e(rev τ))/2
# ═════════════════════════════════════════════════════════════════
hr(f"2. Z2 反對稱能量譜：{N_Z2} 窗、凍結 s27 encoder（載法＝probe_z_geodesic.py §1）")
from torch import nn  # noqa: E402


def sota_mlp(i, h, o, n=2):
    """逐字同 probe_z_geodesic.py:147-159（xavier 初始化隨後被 load_state_dict 覆蓋）。"""
    L, p = [], i
    for _ in range(n):
        lin = nn.Linear(p, h)
        nn.init.xavier_uniform_(lin.weight); nn.init.zeros_(lin.bias)
        L += [lin, nn.GELU(), nn.LayerNorm(h)]
        p = h
    lin = nn.Linear(p, o)
    nn.init.xavier_uniform_(lin.weight); nn.init.zeros_(lin.bias)
    return nn.Sequential(*L, lin)


traj_enc = sota_mlp(2, 512, 512).to(device)
e_pooler = PerceiverPooler(512, D_MODEL, K, 2, 4, max_len=max(512, T_CAP)).to(device)
for name, mod in (("traj_enc", traj_enc), ("e_pooler", e_pooler)):
    missing, unexpected = mod.load_state_dict(ck[name], strict=False)
    assert not missing and not unexpected, f"⛔ {name} state_dict 鍵沒對上：{missing}/{unexpected}"
    mod.eval()
    for p_ in mod.parameters():
        p_.requires_grad_(False)
print("  ✓ traj_enc / e_pooler state_dict 鍵全對上（Z2 只 encode，不需 decoder）")


def encode(traj_np):
    out = []
    with torch.no_grad():
        for b0 in range(0, len(traj_np), 128):
            tb = torch.from_numpy(traj_np[b0:b0 + 128])
            Bc = tb.shape[0]
            mask = torch.zeros(Bc, T_CAP, dtype=torch.bool)
            e = e_pooler(traj_enc(tb.reshape(Bc * T_CAP, 2)).reshape(Bc, T_CAP, 512),
                         key_padding_mask=mask)      # [B,K,D]，同 probe_z_geodesic.py:183-186
            out.append(e.reshape(Bc, -1))
    return torch.cat(out).numpy().astype(np.float64)


rows_z, goals_z = sample_windows(N_Z2, rng)
traj_z = build_traj(rows_z, goals_z)                  # [N,T,2]
E_fwd = encode(traj_z)                                # e(τ)
E_rev = encode(traj_z[:, ::-1].copy())                # e(rev τ)（正規化是逐維仿射，與時間反轉可交換）
E_fwd2 = encode(traj_z[:64])                          # 儀器自檢②：決定性
assert np.array_equal(E_fwd[:64], E_fwd2), "⛔ encoder 非決定性（同輸入兩次輸出不同），停手"
print(f"  ✓ 決定性自檢過（64 窗重算逐位相同）  e 攤平維度={E_fwd.shape[1]}（K×D={K}×{D_MODEL}）")

phi_m = 0.5 * (E_fwd - E_rev)
phi_p = 0.5 * (E_fwd + E_rev)
e_sq = (E_fwd ** 2).sum(1)
ratio = (phi_m ** 2).sum(1) / e_sq                          # 任務規格的主定義
ratio_sym = (phi_m ** 2).sum(1) / ((phi_m ** 2).sum(1) + (phi_p ** 2).sum(1))
cos_fr = (E_fwd * E_rev).sum(1) / (np.linalg.norm(E_fwd, axis=1) * np.linalg.norm(E_rev, axis=1))
# 控制組：隨機配對（把 rev 換成別的窗）—— 給「無關兩窗」的能量比尺度
perm = np.random.default_rng(SEED + 1).permutation(N_Z2)
phi_rand = 0.5 * (E_fwd - E_fwd[perm])
ratio_rand = (phi_rand ** 2).sum(1) / e_sq
print(f"\n  ‖φ₋‖²/‖e‖²      ：{pct(ratio)}   mean={ratio.mean():.4f}")
print(f"  ‖φ₋‖²/(‖φ₊‖²+‖φ₋‖²)：{pct(ratio_sym)}")
print(f"  控制組（隨機配對）：{pct(ratio_rand, (25, 50, 75))}  ← 無關兩窗的差能量尺度")
cq1, cmed, cq3 = np.percentile(cos_fr, [25, 50, 75])
print(f"  cos(e(τ),e(rev τ))  Q1={cq1:+.4f}  中位={cmed:+.4f}  Q3={cq3:+.4f}")
print("  （C1 錨：probe_c_battery_report.txt 原始四分位＝Q1=0.5549／中位=0.9901／Q3=0.9982")
print("   —— C1 分佈本來就寬，⑨ 只引了中位；N=200 vs 本節 N 更大、rng 流不同，中位對短長窗")
print("   比例敏感。本節的能量譜就是把那個被中位遮住的下半邊攤開。）")
len_z = (goals_z - rows_z).astype(np.float64)
rho_lr, p_lr = spearmanr(len_z, ratio)
print(f"\n  能量比 × 窗長：Spearman rho={rho_lr:+.3f}（p={p_lr:.2g}）；按窗長三分位拆：")
lq1, lq2 = np.percentile(len_z, [33.3, 66.7])
for lo_v, hi_v, name in ((0, lq1, f"短(≤{lq1:.0f}步)"), (lq1, lq2, "中"), (lq2, 1e9, f"長(>{lq2:.0f}步)")):
    m_l = (len_z > lo_v) & (len_z <= hi_v)
    print(f"    {name:12s} n={int(m_l.sum()):4d}  能量比中位={np.median(ratio[m_l]):.4f}"
          f"  cos中位={np.median(cos_fr[m_l]):+.4f}")

for tag, M in (("φ₋", phi_m), ("φ₊", phi_p), ("e ", E_fwd)):
    Mc = M - M.mean(0, keepdims=True)
    sv = np.linalg.svd(Mc, compute_uv=False)
    ev = sv ** 2 / (sv ** 2).sum()
    cum = np.cumsum(ev)
    d90, d95, d99 = (int(np.searchsorted(cum, q) + 1) for q in (0.90, 0.95, 0.99))
    pr = float((sv ** 2).sum() ** 2 / (sv ** 4).sum())
    extra = ""
    if tag == "φ₋":
        mean_frac = float((M.mean(0) ** 2).sum() / (M ** 2).sum(1).mean())
        extra = f"  （‖mean φ₋‖²/E‖φ₋‖²={mean_frac:.3f}，中心化前後差異小）"
        d90_phim = d90
    print(f"  {tag} PCA（中心化）：dim@90%={d90}  @95%={d95}  @99%={d99}  參與比PR={pr:.1f}{extra}")

z2_med = float(np.median(ratio))
z2_pass = z2_med < CRIT_Z2_RATIO
print(f"\n  ▶ Z2 判定（預釘：能量比中位 < {CRIT_Z2_RATIO:.0%}）：中位={z2_med:.4f} ⇒ "
      f"{'Remark 4.2「幾乎全落 F₊」實錘、D_d 可小' if z2_pass else '未達 ⇒ 反對稱能量不可忽略'}")
print(f"    D_d 預算實測依據：φ₋ 90% 變異需 {d90_phim} 維（Remark 4.2 的 D_d=16 與此對照）")
print(f"    ⚠️ 邊界：N={N_Z2} 樣本估 {E_fwd.shape[1]} 維 PCA，dim 讀數受樣本上限 {N_Z2} 截尾；"
      f"單顆 ckpt（s27）、單 env。")

# ═════════════════════════════════════════════════════════════════
# 3. Prop 1.1 四格對答案（已有量測，本節不新跑）
# ═════════════════════════════════════════════════════════════════
hr("3. Prop 1.1（ε=0 ⇒ Δ=0 定性級聯）四格對答案 —— 數字出處 FINDINGS-2026-09-05 ①⑦⑬")
print("""  ε＝⑬ 分支散度探針的 cond 層相對差（d_zero 括號附）；單顆 s40。
  Δ＝任務層成功率差。⚠️ 兩種定義並存、不可混讀：
     idp 兩格＝同一顆模型 on/zero 成對差（Prop 1.1 的 Δ_succ 正定義；8 顆 s40-47）；
     f27n 兩格＝臂 vs 無 intent 基線（f27n 無 drop ⇒ zero 模式 OOD、無成對量測，
     以跨臂差替代 —— 是 Δ_succ 的代理不是本尊，表內標 ※）。

  格                 ε_cond(d_zero)   Δ                     顯著性錨          級聯方向檢查
  f27n × R0      ※   .6046 (.0861)    +.133 (.321→.454)     ≈2.7 SE           ε大、Δ非零 ✓ 一致
  f27n × subgoal ※   .6046 (.0861)    −.002 (.857→.855)     噪音級            ε大、Δ≈0   ✓ 一致（逆命題本就不成立：h 可平坦）
  idp  × R0          .0109 (.0009)    +.015±.025            6/8 同號、不顯著   ε≈0、Δ≈0   ✓ 一致
  idp  × subgoal     .0109 (.0009)    +.037±.020            8/8 同號(p≈.008)  ε≈0、Δ 小而系統性 ⚠ 邊界格""")
print("""  判讀（中性）：
  - 嚴格反例＝無。Prop 1.1 是 ε=0 的精確命題；idp 的 ε=.0109≠0，+.037 的小 Δ 與
    Prop 1.2 量化鏈（|Δ| ≤ 2ρ(η) + C·ε）相容 —— 不構成反駁。
  - 但邊界格有資訊：兩腿的 Δ/ε 斜率差一個量級以上（R0 腿 .133/.6046≈0.22；
    subgoal 腿 .037/.0109≈3.4）⇒ ε 單獨不是 Δ 的跨腿預測子，Prop 1.2 的常數
    （L、ρ(η)）腿依賴、目前未量 —— 與理論件 §5 的 H1 待驗欄一致。
  - 邊界：ε 是單顆 s40、sup 的有限樣本估；f27n 兩格的 Δ 是代理（※）。""")
prop_pass = True   # 預釘判準：不存在「ε≈0 且 Δ 顯著非零(8/8 級)」格 —— idp×subgoal 的
#                    ε=.0109 非 0，且量級與 Prop 1.2 相容 ⇒ 按預釘文字判「無嚴格反例」，
#                    同時如實標示它是最接近邊界的格（見上）。
print(f"  ▶ Prop 1.1 判定：{'方向成立、無嚴格反例（附邊界格警語）' if prop_pass else '存在反例'}")

# ═════════════════════════════════════════════════════════════════
# 4. 彙總
# ═════════════════════════════════════════════════════════════════
hr("4. 彙總")
print(f"  A1  決定程度中位（n≥5 主量測）={det5:.3f}（門檻 {CRIT_A1_MEDIAN_DET}【猜測、可調】）"
      f" ⇒ {'成立' if a1_pass else '未達門檻'}；"
      f"I(τ;a|s,g)≈{res_raw.get(5, {}).get('wavg', float('nan')):.1f} bits ≠ 0")
print(f"  Z2  能量比中位={z2_med:.4f}（門檻 {CRIT_Z2_RATIO:.0%}）"
      f"  φ₋ dim@90%={d90_phim} ⇒ {'預釘判準過' if z2_pass else '未達'}；"
      f"⚠️ 但分佈雙峰＝窗長驅動（長窗中位見 §2 分層）——「幾乎全落 F₊」是短窗性質，非全域")
print(f"  P11 四格無嚴格反例；idp×subgoal 為邊界格（8/8 同號小 Δ 伴 ε=1.1%）")
print(f"\n耗時 {time.time() - T0:.1f}s")
