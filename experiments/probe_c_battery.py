"""路線一（距離幾何）第二探針：C-battery —— 量三個「loss 設計案」的前置檢查＋一個 baseline 欄，
數字用來裁一個 loss 設計案（rank/ordering 類正則）的生死。CPU、standalone、唯讀（不訓練、
不改任何既有檔案）。以 experiments/probe_z_geodesic.py 為底座 —— 凍結 ckpt 的
traj_enc/e_pooler/u_dec 載入方式、資料切窗與正規化、佔據圖與 BFS、legal_fraction 合法率計算，
全部逐字沿用該腳本，每個決定附 file:line 出處（出處一律指向 probe_z_geodesic.py；它自己對
scratch_lacot_rollout.py 的出處請見該檔）。引不到出處的地方明講是「本探針自己的選擇」。

量四件事（判準預釘、照抄，不加自己的建議）：
  C1  rev 對現況：N=200 個真窗，encode τ 與 rev(τ)（時間倒放、同一套正規化）。報
      cos(e(τ),e(rev τ)) 中位/四分位；‖e(τ)−e(rev τ)‖ 在隨機 pair 距離分佈的百分位（中位）。
      判準：中位 cos<0 ⇒「L_rev 買不到新東西」；rev 對距離落最低十分位 ⇒「rank 池必須排除 rev」。
  C2  d_time 效度：同軌跡窗對，Δt 分 4 檔（每檔 ≥200 對）＝d_time；BFS(兩窗起點)＝d_bfs。報
      整體 Spearman＋分箱 Spearman（找死檔）；再抽 500 個三元組看「時間排序 vs BFS 排序」反轉率。
      判準：整體 rho<.4 或存在死檔 ⇒「只用短 Δt」；反向三元組 >15% ⇒「錯拉底線成立」。
  C5  密度洞 vs 排序：同 probe_z_geodesic.py 的 100 對，弦中點(t=.5)／matched-Gaussian／
      held-out 真 e 三組對「全體真窗 e 雲」的 k-NN(k=5) 距離分佈。
      判準：中點 k-NN 顯著大於 Gaussian（中位比>1.5 或分佈明顯右移）⇒「density 洞實錘、
      ordering 藥治不到」。
  C4-baseline  座標空間 lerp：同 100 對，各自 decode 成 waypoints 後直接座標空間 lerp(t=.5)，
      量合法率——天真平均樓地板，⛔ 不是 falsifier，報告照這樣標。

跑法：
    cd ~/Projects/lacot
    OGBENCH_DATA_DIR=/home/cymaxwelllee/data/ogbench MUJOCO_GL=osmesa \
    /home/cymaxwelllee/venvs/lacot-rocm/bin/python experiments/probe_c_battery.py \
        2>&1 | tee experiments/probe_c_battery_report.txt

（MUJOCO_GL 跟 probe_z_geodesic.py 一樣其實用不到 —— 這支也只讀 npz＋凍結權重，不 step
環境、不 render；設進去只是跟任務描述的共用環境對齊，無害。）
"""
import os
import sys
import time

import numpy as np
import torch
from torch import nn
from scipy.stats import spearmanr, percentileofscore
from scipy.spatial.distance import pdist, cdist

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
# ⛔ 唯讀重用：只 import 純類別定義的模組，不 import 主檔（同 probe_z_geodesic.py:46-52）。
from lacot.e_target import PerceiverPooler      # noqa: E402
from lacot.traj_decoder import TrajDecoder      # noqa: E402
from lacot.refine_grad import GeoEnergy         # noqa: E402
from lacot.subgoal import grid_bfs              # noqa: E402

T0 = time.time()

# ─────────────────────────────────────────────────────────────────
# 設定 —— ckpt/資料/GEO_RES 沿用 probe_z_geodesic.py:59-70（同一顆 ckpt、同一份資料）
# ─────────────────────────────────────────────────────────────────
ENV_NAME = os.environ.get("LACOT_ENV", "pointmaze-large-stitch-v0")
OGB_DATA = os.environ.get("OGBENCH_DATA_DIR", os.path.expanduser("~/data/ogbench"))
CKPT_NAME = ("ckpt_large-stitch_self_K8_c256_ch4_st8000_T128_ep2_gu_eorecon_ictr_tch0.5"
             "_emw0.999_wu500_dssoft_norf_cd0.1_bci_s27.pt")
CKPT_PATH = os.environ.get("LACOT_CKPT", os.path.join(REPO_ROOT, "results", CKPT_NAME))
GEO_RES = 8
device = "cpu"
KNN_K = 5

# C-battery 專用常數。PAIR_N/PAIR_SEED 刻意固定＝probe_z_geodesic.py:64-65 的預設值
# （N_PAIRS=100, SEED=0），不走 env override，確保「同 100 對」重跑仍是同一批。
# 其餘每段各自獨立 rng 流／可 env override，數字是本探針自訂，非沿用 probe_z_geodesic.py 的值。
PAIR_N, PAIR_SEED = 100, 0
C1_N = int(os.environ.get("LACOT_C1_N", 200))
C1_SEED = int(os.environ.get("LACOT_C1_SEED", 1101))
C2_PER_BIN = int(os.environ.get("LACOT_C2_PER_BIN", 200))
C2_N_TRIPLETS = int(os.environ.get("LACOT_C2_N_TRIPLETS", 500))
C2_BIN_SEED = int(os.environ.get("LACOT_C2_BIN_SEED", 1201))
C2_TRIP_SEED = int(os.environ.get("LACOT_C2_TRIP_SEED", 1202))
C5_CLOUD_M = int(os.environ.get("LACOT_C5_CLOUD_M", 2000))
C5_HOLD_N = int(os.environ.get("LACOT_C5_HOLD_N", 100))
C5_CLOUD_SEED = int(os.environ.get("LACOT_C5_CLOUD_SEED", 1301))
C5_HOLD_SEED = int(os.environ.get("LACOT_C5_HOLD_SEED", 1302))
C5_GAUSS_SEED = int(os.environ.get("LACOT_C5_GAUSS_SEED", 1303))


def hr(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# ─────────────────────────────────────────────────────────────────
# 0. 資料載入 + 正規化（逐字對齊 probe_z_geodesic.py:87-102）
# ─────────────────────────────────────────────────────────────────
hr("0. 資料載入（同 probe_z_geodesic.py:87-102）")
print(f"  env={ENV_NAME}  data_dir={OGB_DATA}")
_npz = np.load(f"{OGB_DATA}/{ENV_NAME}.npz")
OBS = np.asarray(_npz["observations"], np.float32)
TERM = np.asarray(_npz["terminals"], bool)
N = OBS.shape[0]
ends = np.flatnonzero(TERM)
starts = np.concatenate([[0], ends[:-1] + 1])
traj_end = np.empty(N, np.int64)
for s0, e0 in zip(starts, ends):
    traj_end[s0:e0 + 1] = e0
assert ends[-1] == N - 1, "⛔ 資料集最後一筆不是 terminal（同 rollout.py:35 的自檢）"
mu, sd = OBS.mean(0), OBS.std(0) + 1e-6   # 全資料集算，出處=scratch_lacot_rollout.py:37-38
n_eps = len(starts)
ep_lens = ends - starts + 1
print(f"  OBS shape={OBS.shape}  episodes={n_eps}  每集長度 min/median/max="
      f"{ep_lens.min()}/{int(np.median(ep_lens))}/{ep_lens.max()}")
print(f"  mu={mu.tolist()}  sd={sd.tolist()}")

# ─────────────────────────────────────────────────────────────────
# 1. 讀 ckpt cfg，重建 traj_enc / e_pooler / u_dec（逐字對齊 probe_z_geodesic.py:106-180）
# ─────────────────────────────────────────────────────────────────
hr("1. 重建凍結模型（同 probe_z_geodesic.py:106-180）")
print(f"  ckpt={CKPT_PATH}")
ck = torch.load(CKPT_PATH, map_location=device, weights_only=False)
cfg = ck.get("cfg", {})
print(f"  ckpt['cfg']={cfg}")
K, COND, D_MODEL, T_CAP = cfg["K"], cfg["COND"], cfg["D_MODEL"], cfg["T_CAP"]
ENC_OBJ, CHUNK = cfg["ENC_OBJ"], cfg["CHUNK"]

# ⛔ 卡點檢查——跟 probe_z_geodesic.py:117-138 同一套範圍限制，不硬湊。
problems = []
if not ENC_OBJ.startswith("recon"):
    problems.append(f"ENC_OBJ={ENC_OBJ!r} 不是 recon*")
if "u_dec" not in ck:
    problems.append("ckpt 沒有 'u_dec' 鍵")
if "s_embed" in ck:
    problems.append("ckpt 有 's_embed' 鍵（DEC_START=hard，本腳本沒實作）")
if "vq" in ck:
    problems.append("ckpt 有 'vq' 鍵（需要 VQ snap，本腳本沒實作）")
if "intent_ad" in ck:
    problems.append("ckpt 有 'intent_ad' 鍵（需要 intent adapter，本腳本沒實作）")
if problems:
    print("\n⛔⛔⛔ 卡點：ckpt 的訓練設定超出本腳本驗證過的範圍 ⛔⛔⛔")
    for p in problems:
        print("  - " + p)
    print("不硬湊、不塞假資料。停在這裡。")
    sys.exit(1)
print("  ✓ ENC_OBJ=recon_ictr / 無 s_embed / 無 vq / 無 intent_ad ⇒ decode = 純 u_dec(u)")


def sota_mlp(i, h, o, n=2):
    """逐字抄自 probe_z_geodesic.py:147-159。"""
    L, p = [], i
    for _ in range(n):
        lin = nn.Linear(p, h)
        nn.init.xavier_uniform_(lin.weight)
        nn.init.zeros_(lin.bias)
        L += [lin, nn.GELU(), nn.LayerNorm(h)]
        p = h
    lin = nn.Linear(p, o)
    nn.init.xavier_uniform_(lin.weight)
    nn.init.zeros_(lin.bias)
    return nn.Sequential(*L, lin)


traj_enc = sota_mlp(2, 512, 512).to(device)
e_pooler = PerceiverPooler(512, D_MODEL, K, 2, 4, max_len=max(512, T_CAP)).to(device)
u_dec = TrajDecoder(D_MODEL, T_CAP).to(device)

for name, mod in (("traj_enc", traj_enc), ("e_pooler", e_pooler), ("u_dec", u_dec)):
    missing, unexpected = mod.load_state_dict(ck[name], strict=False)
    status = "✓ 全部對上" if (not missing and not unexpected) else "⛔ 鍵不對！"
    print(f"  {name:10s} load_state_dict  missing={missing}  unexpected={unexpected}  {status}")
    assert not missing and not unexpected, f"⛔ {name} 的 state_dict 鍵沒有完全對上，停手。"

for m in (traj_enc, e_pooler, u_dec):
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)


def etarget(traj, mask):
    """逐字對齊 probe_z_geodesic.py:183-186。"""
    Bc, Tc, _ = traj.shape
    return e_pooler(traj_enc(traj.reshape(Bc * Tc, 2)).reshape(Bc, Tc, 512), key_padding_mask=mask)


def decode(u):
    """對齊 probe_z_geodesic.py:189-192（DEC_START!='hard' 分支，純 u_dec(u)）。"""
    return u_dec(u)


def encode_windows(traj_np, chunk=500):
    """跟 etarget() 語意相同，但分批餵（chunk=500）——本探針的工程選擇（非 file:line 出處），
    避免 C5 的 M=2000 雲一次整批塞進 MLP。traj_enc/e_pooler 逐樣本獨立（LayerNorm 非 BatchNorm、
    cross-attention 用 per-sample key_padding_mask），分批跟整批數值上應一致。"""
    outs = []
    n = traj_np.shape[0]
    with torch.no_grad():
        for i in range(0, n, chunk):
            sub = torch.from_numpy(traj_np[i:i + chunk])
            m = torch.zeros(sub.shape[0], T_CAP, dtype=torch.bool)   # 全 False，同 :295
            outs.append(etarget(sub, m))
    return torch.cat(outs, 0)


def flat(e):
    """[B,K,D_MODEL] → [B,K*D_MODEL]。攤平慣例沿用 probe_z_geodesic.py:369-372。"""
    Bc = e.shape[0]
    return e.reshape(Bc, -1)


# ─────────────────────────────────────────────────────────────────
# 2. 佔據圖 + BFS + 切窗工具（逐字對齊 probe_z_geodesic.py:198-282）
# ─────────────────────────────────────────────────────────────────
hr("2. 佔據圖 + BFS + 切窗工具（同 probe_z_geodesic.py:198-282）")
geo = GeoEnergy(OBS, mu, sd, res=GEO_RES, device="cpu")
occ = (geo.dist[0, 0].numpy() == 0.0)
shape_arr = np.asarray(geo.shape, np.int64)
lo_np, span_np = geo.lo, (geo.hi - geo.lo)
print(f"  grid shape={tuple(int(s) for s in geo.shape)}  自由格覆蓋率={geo.coverage:.1%}"
      f"（free={int(occ.sum())}/{int(occ.size)}）")
gh = geo.health()
print(f"  GeoEnergy.health()：ok={gh['ok']}  mapping_err={gh['mapping_err']:.2e}"
      f"  wall_median_random={gh['wall_median_random']:.4f}")


def zn_to_cell_batch(pts):
    """逐字對齊 probe_z_geodesic.py:214-222（不 snap，只 round+clip）。"""
    idx = np.round((np.asarray(pts, np.float64) - lo_np) / span_np * (shape_arr - 1)).astype(np.int64)
    return np.clip(idx, 0, shape_arr - 1)


def legal_fraction(pts):
    idx = zn_to_cell_batch(pts)
    ok = occ[idx[..., 0], idx[..., 1]]
    return float(ok.mean()), int(ok.size)


def zn_to_cell_one(z, allow_snap=False):
    """逐字對齊 probe_z_geodesic.py:232-241。"""
    idx = zn_to_cell_batch(z[None])[0]
    c = tuple(int(v) for v in idx)
    if occ[c] or not allow_snap:
        return c, (not occ[c])
    free_cells = np.argwhere(occ)
    nn_idx = free_cells[np.abs(free_cells - idx).sum(1).argmin()]
    return tuple(int(v) for v in nn_idx), True


def start_cell(r):
    """單點起點 cell，供 C2 用。z=(OBS[r]-mu)/sd 全程 float32 算術（不先轉 float64）——
    跟 build_traj() 在 t=0 端點的計算路徑一致（t=0 時 lo_i=r、w=0，等價於直接算這條式子），
    位元上等同 probe_z_geodesic.py:383-384 對 trajA[i,0] 呼叫 zn_to_cell_one 那條路徑。"""
    z = (OBS[r] - mu) / sd
    return zn_to_cell_one(z, allow_snap=True)


def sample_rows_goals(n, rng):
    """逐字對齊 probe_z_geodesic.py:256-270，改為外部傳入 rng（本探針要開多條獨立抽樣流，
    不像 probe_z_geodesic.py 只用一條 module-level rng；抽樣邏輯本身逐字相同）。"""
    rows, goals = [], []
    n_retry = 0
    while len(rows) < n:
        r = int(rng.integers(0, N))
        te = int(traj_end[r])
        if te - r < CHUNK:
            n_retry += 1
            continue
        _d = rng.random()
        gr = int(round(min(r + 1, te) * _d + te * (1 - _d)))
        gr = max(gr, min(r + CHUNK, te))
        rows.append(r)
        goals.append(gr)
    return np.array(rows), np.array(goals), n_retry


def build_traj(rows, goals):
    """逐字對齊 probe_z_geodesic.py:273-282。"""
    n = len(rows)
    f = np.linspace(rows[:, None].astype(np.float64), goals[:, None].astype(np.float64),
                     T_CAP, axis=1).reshape(n, T_CAP)
    lo_i = np.floor(f).astype(np.int64)
    hi_i = np.minimum(lo_i + 1, goals[:, None])
    w = (f - lo_i)[..., None]
    traj = ((OBS[lo_i] * (1.0 - w) + OBS[hi_i] * w - mu) / sd).astype(np.float32)
    return traj


# ─────────────────────────────────────────────────────────────────
# 3. C1【rev 對現況】
# ─────────────────────────────────────────────────────────────────
hr(f"3. C1【rev 對現況】N={C1_N}")
print("  設計選擇：rev(τ) = build_traj() 產出的正規化窗沿 T_CAP 軸整段反轉（[:, ::-1, :]），")
print("  不是拿 (goal,row) 互換端點重跑 build_traj —— build_traj 的 floor/clamp 不對稱")
print("  （lo_i=floor(f)、hi_i=min(lo_i+1,goal)，見上面 build_traj 定義，")
print("  對齊 probe_z_geodesic.py:278-280），互換端點不是真正的鏡像倒放。")
print("  本探針自己的選擇（任務規格文字『時間倒放』直接對應，非 file:line 出處）。")

rng_c1 = np.random.default_rng(C1_SEED)
rows_c1, goals_c1, retry_c1 = sample_rows_goals(C1_N, rng_c1)
traj_c1 = build_traj(rows_c1, goals_c1)
rev_c1 = traj_c1[:, ::-1, :].copy()
e_fwd = flat(encode_windows(traj_c1)).numpy()
e_rev = flat(encode_windows(rev_c1)).numpy()
print(f"  抽到 {C1_N} 個真窗（重試 {retry_c1} 次）")

num = (e_fwd * e_rev).sum(1)
den = np.linalg.norm(e_fwd, axis=1) * np.linalg.norm(e_rev, axis=1)
cos = num / den
cos_q1, cos_med, cos_q3 = np.percentile(cos, [25, 50, 75])
print(f"  cos(e(τ),e(rev τ))   Q1={cos_q1:.4f}   中位={cos_med:.4f}   Q3={cos_q3:.4f}")

rev_dist = np.linalg.norm(e_fwd - e_rev, axis=1)
# 隨機 pair 距離背景分佈＝這 C1_N 個真窗 e(τ) 兩兩距離（i<j）—— 本探針的選擇，非 file:line 出處。
bg = pdist(e_fwd)
print(f"  隨機 pair 距離背景分佈：n={len(bg)}（{C1_N} 個真窗兩兩距離）  中位={np.median(bg):.3f}")
percentiles = percentileofscore(bg, rev_dist, kind="mean")
med_pct = float(np.median(percentiles))
print(f"  ‖e(τ)−e(rev τ)‖ 中位={np.median(rev_dist):.3f}；")
print(f"  落在隨機 pair 距離分佈的百分位（每個窗各自算，取中位）={med_pct:.1f}")

c1a = "成立" if cos_med < 0 else "不成立"
c1b = "成立" if med_pct <= 10 else "不成立"
print(f"\n  判準 C1a：中位 cos<0 ⇒「L_rev 買不到新東西」。中位 cos={cos_med:.4f} ⇒ {c1a}")
print(f"  判準 C1b：rev 對距離落在最低十分位 ⇒「rank 池必須排除 rev」。"
      f"中位百分位={med_pct:.1f} ⇒ {c1b}")


# ─────────────────────────────────────────────────────────────────
# 4. C2【d_time 效度】
# ─────────────────────────────────────────────────────────────────
hr("4. C2【d_time 效度】")
DT_MAX = int(ep_lens.min()) - 1 - CHUNK
edges = np.linspace(1, DT_MAX + 1, 5).astype(int)
bins = [(int(edges[i]), int(edges[i + 1]) - 1) for i in range(4)]
print(f"  Δt 全距＝1..{DT_MAX}（來自單集最短長度 {int(ep_lens.min())}、CHUNK={CHUNK}；")
print("  合法窗判準 te-r>=CHUNK 出處＝probe_z_geodesic.py:262）；均切 4 箱："
      f"{bins}（本探針操作型定義，非 file:line 出處）")

rng_bin = np.random.default_rng(C2_BIN_SEED)


def sample_pair_for_bin(lo, hi, rng):
    n_try = 0
    while True:
        n_try += 1
        k = int(rng.integers(0, n_eps))
        s0, e0 = int(starts[k]), int(ends[k])
        max_r = e0 - CHUNK
        dt = int(rng.integers(lo, hi + 1))
        if max_r - dt < s0:
            continue
        r_a = int(rng.integers(s0, max_r - dt + 1))
        r_b = r_a + dt
        return r_a, r_b, dt, n_try


all_dt, all_dbfs = [], []
bin_stats = []
for (lo, hi) in bins:
    dts, dbfss = [], []
    n_unreach = n_snap = tries = 0
    for _ in range(C2_PER_BIN):
        r_a, r_b, dt, n_try = sample_pair_for_bin(lo, hi, rng_bin)
        tries += n_try
        cA, snapA = start_cell(r_a)
        cB, snapB = start_cell(r_b)
        n_snap += int(snapA) + int(snapB)
        dmap = grid_bfs(occ, cA)
        d = dmap.get(cB)
        dts.append(dt)
        dbfss.append(np.nan if d is None else d)
        if d is None:
            n_unreach += 1
    dts = np.array(dts, float)
    dbfss = np.array(dbfss, float)
    valid = ~np.isnan(dbfss)
    if valid.sum() >= 3:
        rho, p = spearmanr(dts[valid], dbfss[valid])
    else:
        rho, p = float("nan"), float("nan")
    is_dead = (not (p < 0.05)) or (not (rho > 0))
    bin_stats.append(dict(lo=lo, hi=hi, n=int(valid.sum()), rho=rho, p=p, dead=is_dead))
    all_dt.append(dts[valid])
    all_dbfs.append(dbfss[valid])
    print(f"  Δt∈[{lo:3d},{hi:3d}]  n_valid={int(valid.sum())}/{C2_PER_BIN}"
          f"（不連通排除{n_unreach}，取樣重試{tries - C2_PER_BIN}次，起點snap{n_snap}次）"
          f"  Spearman rho={rho:.3f} p={p:.4g}  {'死檔' if is_dead else '活'}")

all_dt = np.concatenate(all_dt)
all_dbfs = np.concatenate(all_dbfs)
rho_all, p_all = spearmanr(all_dt, all_dbfs)
n_dead = sum(1 for b in bin_stats if b["dead"])
print(f"\n  整體（4 箱合併）n={len(all_dt)}  Spearman rho={rho_all:.3f}  p={p_all:.4g}")
print(f"  死檔數（本探針操作型定義：該箱 p≥0.05 或 rho≤0 ⇒ 死）＝{n_dead}/4")

c2a = "成立" if (rho_all < 0.4 or n_dead > 0) else "不成立"
print(f"\n  判準 C2a：整體 rho<.4 或存在死檔 ⇒「只用短 Δt」。"
      f"整體 rho={rho_all:.3f}（{'<' if rho_all < 0.4 else '≥'}.4），死檔={n_dead} ⇒ {c2a}")

# 三元組：anchor=A（三點抽樣順序中的第一個，角色對稱、隨機下不偏），比較 B/C 何者「時間較近」
# 是否等於「BFS 較近」。這是本探針對「d_time 排序 vs d_bfs 排序相反」的操作化，非 file:line 出處。
rng_trip = np.random.default_rng(C2_TRIP_SEED)
n_reversed = n_tie = n_unreach_t = n_snap_t = 0
for _ in range(C2_N_TRIPLETS):
    while True:
        k = int(rng_trip.integers(0, n_eps))
        s0, e0 = int(starts[k]), int(ends[k])
        max_r = e0 - CHUNK
        width = max_r - s0 + 1
        if width >= 3:
            break
    offs = rng_trip.choice(width, size=3, replace=False)
    rA, rB, rC = int(s0 + offs[0]), int(s0 + offs[1]), int(s0 + offs[2])
    dt_AB, dt_AC = abs(rB - rA), abs(rC - rA)
    cA, snA = start_cell(rA)
    cB, snB = start_cell(rB)
    cC, snC = start_cell(rC)
    n_snap_t += int(snA) + int(snB) + int(snC)
    dmap = grid_bfs(occ, cA)
    dAB, dAC = dmap.get(cB), dmap.get(cC)
    if dAB is None or dAC is None:
        n_unreach_t += 1
        continue
    if dt_AB == dt_AC or dAB == dAC:
        n_tie += 1
        continue
    if (dt_AB < dt_AC) != (dAB < dAC):
        n_reversed += 1
n_valid_t = C2_N_TRIPLETS - n_tie - n_unreach_t
prop_rev = n_reversed / n_valid_t if n_valid_t else float("nan")
print(f"\n  三元組 n={C2_N_TRIPLETS}（排除不連通{n_unreach_t}、平手{n_tie}，有效{n_valid_t}，"
      f"起點snap{n_snap_t}次）  反向比例={prop_rev:.1%}（{n_reversed}/{n_valid_t}）")
c2b = "成立" if prop_rev > 0.15 else "不成立"
print(f"  判準 C2b：反向三元組 >15% ⇒「錯拉底線成立」。反向比例={prop_rev:.1%} ⇒ {c2b}")


# ─────────────────────────────────────────────────────────────────
# 5. 共用 100 對（同 probe_z_geodesic.py 預設 N_PAIRS=100, SEED=0）
# ─────────────────────────────────────────────────────────────────
hr(f"5. 共用 {PAIR_N} 對（同 probe_z_geodesic.py 預設 N_PAIRS={PAIR_N}, SEED={PAIR_SEED}）")
rng_pair = np.random.default_rng(PAIR_SEED)
rows_A, goals_A, retryA = sample_rows_goals(PAIR_N, rng_pair)
rows_B, goals_B, retryB = sample_rows_goals(PAIR_N, rng_pair)
trajA = build_traj(rows_A, goals_A)
trajB = build_traj(rows_B, goals_B)
etA = encode_windows(trajA)
etB = encode_windows(trajB)
with torch.no_grad():
    pts_realA = decode(etA).numpy()
    pts_realB = decode(etB).numpy()
print(f"  抽到 {PAIR_N} 對（重試 A={retryA} B={retryB}）—— 抽法/呼叫順序逐字對齊"
      " probe_z_geodesic.py:285-301，rng 種子/呼叫序相同 ⇒ 這是同一批 100 對。")


# ─────────────────────────────────────────────────────────────────
# 6. C5【密度洞 vs 排序】
# ─────────────────────────────────────────────────────────────────
hr(f"6. C5【密度洞 vs 排序】k={KNN_K}")
print(f"  『全體真窗 e 雲』操作化為 M={C5_CLOUD_M} 個獨立抽樣真窗（本探針對『全體』的有限")
print(f"  樣本化，非任務規格具體數字）；held-out={C5_HOLD_N} 個另外獨立抽樣，")
print("  跟雲、跟上面的 100 對都用不同 rng 流（無重疊，下面實測驗證）。")
rng_cloud = np.random.default_rng(C5_CLOUD_SEED)
rows_R, goals_R, retryR = sample_rows_goals(C5_CLOUD_M, rng_cloud)
traj_R = build_traj(rows_R, goals_R)
rng_hold = np.random.default_rng(C5_HOLD_SEED)
rows_H, goals_H, retryH = sample_rows_goals(C5_HOLD_N, rng_hold)
traj_H = build_traj(rows_H, goals_H)
eR = flat(encode_windows(traj_R)).numpy()
eH = flat(encode_windows(traj_H)).numpy()

# 汙染檢查（一個看得見的控制：驗證三組真窗抽樣真的互不重疊，不是靜默假設）。
set_pair = set(zip(rows_A.tolist(), goals_A.tolist())) | set(zip(rows_B.tolist(), goals_B.tolist()))
set_cloud = set(zip(rows_R.tolist(), goals_R.tolist()))
set_hold = set(zip(rows_H.tolist(), goals_H.tolist()))
overlap_ch = len(set_cloud & set_hold)
overlap_cp = len(set_cloud & set_pair)
overlap_hp = len(set_hold & set_pair)
print(f"  重疊檢查（(row,goal) 完全相同才算）：cloud∩held-out={overlap_ch}"
      f"  cloud∩100對={overlap_cp}  held-out∩100對={overlap_hp}（理論上應全為 0）")

e_mid = flat(0.5 * (etA + etB)).numpy()
et_pool = torch.cat([etA, etB], 0)
U_MEAN, U_STD = et_pool.mean(0, keepdim=True), et_pool.std(0, keepdim=True)
g_rand = torch.Generator().manual_seed(C5_GAUSS_SEED)
et_rand = U_MEAN + U_STD * torch.randn(PAIR_N, K, D_MODEL, generator=g_rand)
e_gauss = flat(et_rand).numpy()
print("  matched-Gaussian 構造沿用 probe_z_geodesic.py:322-329（逐維 mean/std 匹配，"
      "precedent＝probe_u.py:92,110-111），"
      f"seed 為本探針自訂（{C5_GAUSS_SEED}，跟 probe_z_geodesic.py 的 20260905 無關）。")


def knn_dist(query, cloud, k):
    d = cdist(query, cloud)
    d.sort(axis=1)
    return d[:, k - 1]


knn_mid = knn_dist(e_mid, eR, KNN_K)
knn_gauss = knn_dist(e_gauss, eR, KNN_K)
knn_hold = knn_dist(eH, eR, KNN_K)

for gname, arr in (("弦中點(t=.5)", knn_mid), ("matched-Gaussian", knn_gauss), ("held-out 真e", knn_hold)):
    lo_q, mid_q, hi_q = np.percentile(arr, [25, 50, 75])
    print(f"  {gname:18s} n={len(arr):4d}  Q1={lo_q:.4f}  中位={mid_q:.4f}  Q3={hi_q:.4f}")

med_mid = float(np.median(knn_mid))
med_gauss = float(np.median(knn_gauss))
q1_mid = float(np.percentile(knn_mid, 25))
q3_gauss = float(np.percentile(knn_gauss, 75))
ratio = med_mid / med_gauss
shifted = q1_mid > q3_gauss
c5 = "成立" if (ratio > 1.5 or shifted) else "不成立"
print("\n  判準 C5：中點 k-NN 顯著大於 Gaussian（中位比>1.5 或分佈明顯右移）"
      "⇒「density 洞實錘、ordering 藥治不到」。")
print(f"  中位比={ratio:.3f}（{'>' if ratio > 1.5 else '≤'}1.5）；"
      f"IQR 不重疊（中點Q1={q1_mid:.4f} vs GaussianQ3={q3_gauss:.4f}）"
      f"＝{'是' if shifted else '否'} ⇒ {c5}")


# ─────────────────────────────────────────────────────────────────
# 7. C4-baseline【座標空間 lerp】—— 天真平均樓地板，⛔ 不是 falsifier
# ─────────────────────────────────────────────────────────────────
hr("7. C4-baseline【座標空間 lerp】")
print(f"  同 {PAIR_N} 對，兩條各自 decode 成 waypoints（沿用 5. 算好的 pts_realA/pts_realB，")
print("  即 roundtrip 還原點，precedent＝probe_z_geodesic.py:300-301,704-722）後直接")
print("  逐 waypoint 座標空間 lerp(t=.5)。⛔ 這是天真平均樓地板，不是 falsifier。")
pts_lerp = (0.5 * (pts_realA + pts_realB)).reshape(-1, 2)
legal_lerp, n_lerp = legal_fraction(pts_lerp)
print(f"  n_waypoints={n_lerp}  合法率={legal_lerp:.1%}")


# ─────────────────────────────────────────────────────────────────
# 彙總
# ─────────────────────────────────────────────────────────────────
hr("彙總")
print(f"  C1a（cos 中位<0 ⇒ L_rev 買不到新東西）：{c1a}（中位 cos={cos_med:.4f}）")
print(f"  C1b（rev 距離落最低十分位 ⇒ rank 池排除 rev）：{c1b}（中位百分位={med_pct:.1f}）")
print(f"  C2a（整體 rho<.4 或死檔 ⇒ 只用短 Δt）：{c2a}（rho={rho_all:.3f}, 死檔={n_dead}/4）")
print(f"  C2b（反向三元組>15% ⇒ 錯拉底線成立）：{c2b}（反向比例={prop_rev:.1%}）")
print(f"  C5（中點k-NN>>Gaussian ⇒ density洞實錘）：{c5}（中位比={ratio:.3f}）")
print(f"  C4-baseline（座標空間lerp合法率，⛔ 非 falsifier）：{legal_lerp:.1%}")
print(f"\n耗時 {time.time() - T0:.1f}s")
