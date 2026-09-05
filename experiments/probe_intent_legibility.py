"""intent 編碼鏈可讀性探針：軌跡 → hindsight 錨 anc → IntentAdapter.cond_global → ix（64 維）。
量「ix 這個 64 維向量，還讀不讀得回路線的幾何特徵」——用 ridge 線性 probe 測，不碰模型本身、
不訓練、不改任何既有檔案。

═══ 任務背景 ═══════════════════════════════════════════════════════════════
intent 編碼鏈（見 lacot/intent.py 開頭 docstring）：
  訓練 hindsight：軌跡 xy → traj_to_cells()（相鄰去重）→ cells_to_anchors() → anchors_resample()
                  → 錨點 anc [T_A=32, 2]（正規化座標，float32）
  接法 (i) embed（lacot/intent_embed.py）：anc flatten 過兩層 MLP（cond_global）→ ix [64]，
                  併進 cond_head 尾巴（軟條件，不強迫對齊）。
本探針只做「ix 還讀不讀得回幾何」這一件事：如果讀得回，問題出在下游怎麼用 ix；
如果讀不回，問題在編碼本身就是糊的。

═══ 材料與出處 ══════════════════════════════════════════════════════════════
- ckpt：results/ckpt_..._ite_..._s40.pt（cfg 存在 ckpt 裡，K=8/COND=256/CHUNK=4/T_CAP=128/
  ENC_OBJ=recon_ictr；檔名 "ite" = intent_embed.py 的 TAG，"emw0.999"=EMA_W=0.999）。
  用 ck["ema"]["intent_ad"]（跟 eval 一致，出處：scratch_lacot_rollout.py:1244-1252 LOAD_EMA
  分支——eval 端 LACOT_LOAD_EMA=1 就是切到這份影子權重）。T_A 從權重形狀反推
  （mlp.0.weight.shape[1]//2），不猜 32——ckpt 存檔的 cfg dict（:2433-2437）沒有存 INTENT_TA，
  猜錯會在 assert 階段被抓到、不會靜默吃錯形狀。
- 資料切窗／正規化／占據圖：逐字對齊 experiments/probe_z_geodesic.py（已驗證，見該檔第 0/2/3 節）
  ，其本身對齊 experiments/scratch_lacot_rollout.py 的 make_batch/GeoEnergy 用法。
- hindsight 錨的重建：逐字對齊 scratch_lacot_rollout.py:325-368（INTENT 區塊的
  _i_zn_to_cell/_i_cell_to_zn/intent_anchors_of，INTENT_SRC 預設="hindsight"、INTENT_TA 預設=32）
  ——⛔ 不 import 該檔（頂層執行 2438 行訓練/eval 主流程，副作用太大），在這裡重建。
- ridge probe：venv 沒裝 sklearn（已查證），手寫閉式解（標準化 X＋中心化 Xy 的嶺回歸），
  5-fold CV 在【訓練集內】選 alpha，最後在留出的 20% test 上報 R²（uniform-average，跟
  sklearn.metrics.r2_score 多輸出維度的預設一致）。

═══ 判準（任務預釘，照抄）═══════════════════════════════════════════════════
  ix 的「終點 cell」與「淨方向」兩個 test R²：
    > 0.5 ⇒「編碼層看得懂」成立、問題在使用端
    < 0.2 ⇒「編碼是糊的」成立
    介於中間 ⇒ 部分可讀

跑法：
    cd ~/Projects/lacot
    OGBENCH_DATA_DIR=/home/cymaxwelllee/data/ogbench MUJOCO_GL=osmesa \
    /home/cymaxwelllee/venvs/lacot-rocm/bin/python experiments/probe_intent_legibility.py \
        2>&1 | tee experiments/probe_intent_legibility_report.txt
"""
import os
import sys
import time

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
# ⛔ 唯讀重用：只 import 純類別/函式定義的模組（無頂層執行副作用），不 import 主檔。
from lacot.refine_grad import GeoEnergy      # noqa: E402  (占據圖，lacot/refine_grad.py:32-63)
from lacot.intent import hindsight_intent    # noqa: E402  (hindsight 機制本身，lacot/intent.py:77-82)
from lacot.intent_embed import IntentAdapter # noqa: E402  (接法 (i) embed，lacot/intent_embed.py:23-41)

T0 = time.time()

# ─────────────────────────────────────────────────────────────────
# 設定
# ─────────────────────────────────────────────────────────────────
ENV_NAME = os.environ.get("LACOT_ENV", "pointmaze-large-stitch-v0")
OGB_DATA = os.environ.get("OGBENCH_DATA_DIR", os.path.expanduser("~/data/ogbench"))
CKPT_NAME = ("ckpt_large-stitch_self_K8_c256_ch4_st8000_T128_ep2_gu_eorecon_ictr_tch0.5"
             "_btf27n_emw0.999_wu500_s1from_ite_dssoft_norf_cd0.1_bci_s40.pt")
CKPT_PATH = os.environ.get("LACOT_CKPT", os.path.join(REPO_ROOT, "results", CKPT_NAME))
N = int(os.environ.get("LACOT_N", 2000))
SEED = int(os.environ.get("LACOT_SEED", 0))              # 抽樣窗用的主 rng seed
SPLIT_SEED = SEED + 1000                                  # train/test 切分（跟資料抽樣、CV 分開，避免糾纏）
CV_SEED = SEED + 2000                                     # ridge alpha 的 5-fold CV 切分
SHUF_SEED = SEED + 3000                                   # shuffle 對照組的目標置換
GEO_RES = 8    # 占據圖解析度 —— 全 repo 唯一慣例，precedent＝probe_z_geodesic.py:67 / scratch_lacot_rollout.py:328
TEST_FRAC = 0.2
CV_K = 5
ALPHA_GRID = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0]
device = "cpu"


def hr(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# ─────────────────────────────────────────────────────────────────
# 0. 資料載入 + 正規化（逐字對齊 experiments/probe_z_geodesic.py:82-102）
# ─────────────────────────────────────────────────────────────────
hr("0. 資料載入與正規化")
print(f"  env={ENV_NAME}  data_dir={OGB_DATA}  N={N}  seed={SEED}")
_npz = np.load(f"{OGB_DATA}/{ENV_NAME}.npz")
OBS = np.asarray(_npz["observations"], np.float32)
TERM = np.asarray(_npz["terminals"], bool)
N_OBS = OBS.shape[0]
ends = np.flatnonzero(TERM)
starts = np.concatenate([[0], ends[:-1] + 1])
traj_end = np.empty(N_OBS, np.int64)
for s0, e0 in zip(starts, ends):
    traj_end[s0:e0 + 1] = e0
assert ends[-1] == N_OBS - 1, "⛔ 資料集最後一筆不是 terminal ⇒ traj_end 尾巴未初始化"
mu, sd = OBS.mean(0), OBS.std(0) + 1e-6
print(f"  OBS shape={OBS.shape}  episodes={len(ends)}  obs_dim={OBS.shape[1]}")
assert OBS.shape[1] == 2, f"⛔ 本探針假設 2D 座標（pointmaze），拿到 obs_dim={OBS.shape[1]}"


# ─────────────────────────────────────────────────────────────────
# 1. 載入 ckpt 的 ema intent_ad 權重（file:line 出處見檔頭）
# ─────────────────────────────────────────────────────────────────
hr("1. 載入 ckpt 的 ema intent_ad 權重")
print(f"  ckpt={CKPT_PATH}")
ck = torch.load(CKPT_PATH, map_location=device, weights_only=False)
cfg = ck.get("cfg", {})
print(f"  ckpt['cfg']={cfg}")
CHUNK, T_CAP = cfg["CHUNK"], cfg["T_CAP"]

problems = []
if "intent_ad" not in ck:
    problems.append("ckpt 沒有 'intent_ad' 鍵 ⇒ 這顆是【沒開 intent】訓的，探針做不下去")
if "ema" not in ck or "intent_ad" not in ck.get("ema", {}):
    problems.append("ckpt 沒有 'ema'['intent_ad'] 段 ⇒ 沒有 EMA 影子權重可用（訓練時 LACOT_EMA_W 沒開）")
if problems:
    print("\n⛔⛔⛔ 卡點：ckpt 不含本探針需要的段 ⛔⛔⛔")
    for p in problems:
        print("  - " + p)
    print("不硬湊、不塞假資料。停在這裡。")
    sys.exit(1)

ema_sd = ck["ema"]["intent_ad"]
T_A = ema_sd["mlp.0.weight"].shape[1] // 2   # 反推 t_anchor，⛔ 不猜 32（ckpt cfg 沒存 INTENT_TA）
print(f"  ema intent_ad 權重鍵：{list(ema_sd.keys())}")
print(f"  反推 T_A（錨點數）= mlp.0.weight.shape[1]//2 = {T_A}")

intent_ad = IntentAdapter(t_anchor=T_A).to(device)
missing, unexpected = intent_ad.load_state_dict(ema_sd, strict=False)
status = "✓ 全部對上" if (not missing and not unexpected) else "⛔ 鍵不對！"
print(f"  intent_ad.load_state_dict(ema)  missing={missing}  unexpected={unexpected}  {status}")
assert not missing and not unexpected, (
    f"⛔ 卡點：intent_ad 的 state_dict 鍵沒有完全對上（missing={missing}, unexpected={unexpected}）"
    f" ⇒ 重建的 IntentAdapter(t_anchor={T_A}) 跟 ckpt 存的不是同一個形狀，停手。")
intent_ad.eval()
for p in intent_ad.parameters():
    p.requires_grad_(False)
print(f"  出處：EMA 載入分支＝scratch_lacot_rollout.py:1244-1252（LACOT_LOAD_EMA=1 時取 ema 段，"
      f"eval 用的就是這份）；state_dict load 慣例對齊 probe_z_geodesic.py:168-173。")


# ─────────────────────────────────────────────────────────────────
# 2. 占據圖 + cell↔正規化座標轉換（訓練同款，逐字對齊 scratch_lacot_rollout.py:325-347）
# ─────────────────────────────────────────────────────────────────
hr("2. 占據圖 + cell↔正規化座標轉換")
geo = GeoEnergy(OBS, mu, sd, res=GEO_RES, device="cpu")
occ = (geo.dist[0, 0].numpy() == 0.0)
free_cells = np.argwhere(occ)
lo_np = np.asarray(geo.lo, np.float64)
span_np = np.asarray(geo.hi - geo.lo, np.float64)
shape_np = np.asarray(geo.shape, np.int64)
print(f"  grid shape={tuple(int(s) for s in geo.shape)}  自由格覆蓋率={geo.coverage:.1%}"
      f"（free={int(occ.sum())}/{int(occ.size)}）")
_n_snap = [0]


def zn_to_cell(z):
    """【正規化】座標 → E 細格；落牆格 snap 到最近自由格。逐字對齊
    scratch_lacot_rollout.py:335-343 的 _i_zn_to_cell（訓練用來算 hindsight 錨的同一個函式）。"""
    idx = np.clip(np.round((np.asarray(z, np.float64)[:2] - lo_np) / span_np * (shape_np - 1)).astype(int),
                  0, shape_np - 1)
    c = tuple(idx)
    if occ[c]:
        return c
    _n_snap[0] += 1
    return tuple(free_cells[int(np.abs(free_cells - idx).sum(1).argmin())])


def cell_to_zn(c):
    """E 細格 → 格心的【正規化】座標。逐字對齊 scratch_lacot_rollout.py:345-347 的 _i_cell_to_zn。"""
    return lo_np + np.asarray(c, np.float64) / (shape_np - 1) * span_np


# ─────────────────────────────────────────────────────────────────
# 3. 抽 N 個真軌跡窗 + 算 hindsight 錨（切窗對齊 probe_z_geodesic.py:256-282／
#    scratch_lacot_rollout.py:452-465,484-489；錨對齊 scratch_lacot_rollout.py:349-368）
# ─────────────────────────────────────────────────────────────────
hr(f"3. 抽 {N} 個真軌跡窗 + 算 hindsight 錨（T_CAP={T_CAP} CHUNK={CHUNK} T_A={T_A}）")
rng = np.random.default_rng(SEED)


def sample_rows_goals(n):
    rows, goals = [], []
    n_retry = 0
    while len(rows) < n:
        r = int(rng.integers(0, N_OBS))
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
    """scratch_lacot_rollout.py:484-489 逐字對齊：固定 T_CAP 點的線性內插重採樣。"""
    n = len(rows)
    f = np.linspace(rows[:, None].astype(np.float64), goals[:, None].astype(np.float64),
                     T_CAP, axis=1).reshape(n, T_CAP)
    lo_i = np.floor(f).astype(np.int64)
    hi_i = np.minimum(lo_i + 1, goals[:, None])
    w = (f - lo_i)[..., None]
    traj = ((OBS[lo_i] * (1.0 - w) + OBS[hi_i] * w - mu) / sd).astype(np.float32)
    return traj


rows, goals, n_retry = sample_rows_goals(N)
traj = build_traj(rows, goals)   # [N, T_CAP, 2] 正規化
print(f"  抽到 {N} 個窗（CHUNK 篩掉重試次數={n_retry}）")
print(f"  片段長度（原始索引步數，goal-row）p50={np.median(goals - rows):.0f}"
      f" p90={np.percentile(goals - rows, 90):.0f} max={(goals-rows).max()}")

anc = np.empty((N, T_A, 2), np.float32)
n_cells_arr = np.empty(N, np.int64)
jit_arr = np.empty(N, np.float64)
for i in range(N):
    a, n_cells, jit = hindsight_intent(traj[i], zn_to_cell, cell_to_zn, T_A)
    anc[i] = a
    n_cells_arr[i] = n_cells
    jit_arr[i] = jit
print(f"  hindsight cell 路線長度（去重後）：p50={np.median(n_cells_arr):.0f}"
      f" p90={np.percentile(n_cells_arr, 90):.0f} max={n_cells_arr.max()}"
      f"（=1 的窗數={int((n_cells_arr==1).sum())}，起訖同格、退化為原地）")
print(f"  jitter_rate（A→B→A 抖動比例）均值={jit_arr.mean():.4f}")
print(f"  zn_to_cell snap 次數（落牆格 fallback，理論上應接近 0）：{_n_snap[0]}"
      f" / {N * T_CAP} 次呼叫")


# ─────────────────────────────────────────────────────────────────
# 4. ix = intent_ad.cond_global(anc)；準備三個目標
# ─────────────────────────────────────────────────────────────────
hr("4. 算 ix，準備三個目標（終點 cell／路徑長／淨方向）")
with torch.no_grad():
    ix = intent_ad.cond_global(torch.from_numpy(anc)).numpy().astype(np.float64)   # [N,64]
anc_flat = anc.reshape(N, -1).astype(np.float64)                                    # [N,64]，攤平上界特徵
print(f"  ix shape={ix.shape}  anc_flat shape={anc_flat.shape}")

end_cell = np.array([zn_to_cell(traj[i, -1]) for i in range(N)], np.float64)   # (a) 終點 cell [N,2]
diffs = np.diff(traj.astype(np.float64), axis=1)                              # [N,T_CAP-1,2]
path_len = np.linalg.norm(diffs, axis=2).sum(axis=1)                          # (b) 路徑弧長 [N]
start_pt = traj[:, 0, :].astype(np.float64)
end_pt = traj[:, -1, :].astype(np.float64)
net_vec = end_pt - start_pt
net_norm = np.linalg.norm(net_vec, axis=1)
degenerate = net_norm < 1e-8
net_dir = np.zeros_like(net_vec)                                              # (c) 淨方向單位向量 [N,2]
net_dir[~degenerate] = net_vec[~degenerate] / net_norm[~degenerate, None]
print(f"  終點 cell：唯一格數={len(set(map(tuple, end_cell.astype(int))))}")
print(f"  路徑弧長：mean={path_len.mean():.3f} std={path_len.std():.3f}")
print(f"  淨位移退化（起訖重合）窗數={int(degenerate.sum())}/{N}"
      f"（這些窗排除在『淨方向』probe 外——單位向量在該點無定義，不能餵零向量硬湊）")


# ─────────────────────────────────────────────────────────────────
# 5. ridge probe 工具（numpy 手寫閉式解，venv 無 sklearn；5-fold CV 選 alpha）
# ─────────────────────────────────────────────────────────────────
hr("5. ridge probe 工具（設計選擇）")
print("  venv 已查證無 sklearn ⇒ 手寫嶺回歸閉式解：標準化 X（用 train 統計量）、中心化 X/y")
print("  （等價於不懲罰截距項，跟 sklearn Ridge(fit_intercept=True) 同款代數）；")
print("  alpha 由訓練集內 5-fold CV 網格搜尋選出（grid={:s}），⛔ 不碰 test 集。".format(str(ALPHA_GRID)))
print("  多維目標（終點 cell／淨方向皆 2 維）的 R² 採 uniform-average（各維 R² 平均），")
print("  跟 sklearn.metrics.r2_score 的 multioutput='uniform_average' 預設一致（本探針自己的選擇，")
print("  非 file:line 出處——task 規格沒指定，這是最貼近『沒裝 sklearn 就用 numpy 手寫』字面意思的還原）。")


def _to2d(y):
    return y.reshape(-1, 1) if y.ndim == 1 else y


def _r2_uniform_avg(y_true, y_pred):
    yt, yp = _to2d(y_true), _to2d(y_pred)
    ss_res = ((yt - yp) ** 2).sum(axis=0)
    ss_tot = ((yt - yt.mean(axis=0)) ** 2).sum(axis=0)
    r2_dim = 1.0 - ss_res / np.maximum(ss_tot, 1e-12)
    return float(r2_dim.mean()), r2_dim


def _ridge_fit(Xtr, ytr, alpha):
    Xm, ym = Xtr.mean(0), ytr.mean(0)
    Xc, yc = Xtr - Xm, ytr - ym
    d = Xc.shape[1]
    beta = np.linalg.solve(Xc.T @ Xc + alpha * np.eye(d), Xc.T @ yc)
    return beta, Xm, ym


def _ridge_predict(beta, Xm, ym, X):
    return (X - Xm) @ beta + ym


def _cv_select_alpha(X, y, k=CV_K, seed=CV_SEED):
    y2 = _to2d(y)
    n = X.shape[0]
    perm = np.random.default_rng(seed).permutation(n)
    folds = np.array_split(perm, k)
    best_alpha, best_score = ALPHA_GRID[0], -np.inf
    for alpha in ALPHA_GRID:
        scores = []
        for i in range(k):
            val_idx = folds[i]
            tr_idx = np.concatenate([folds[j] for j in range(k) if j != i])
            beta, Xm, ym = _ridge_fit(X[tr_idx], y2[tr_idx], alpha)
            pred = _ridge_predict(beta, Xm, ym, X[val_idx])
            r2, _ = _r2_uniform_avg(y2[val_idx], pred)
            scores.append(r2)
        mscore = float(np.mean(scores))
        if mscore > best_score:
            best_score, best_alpha = mscore, alpha
    return best_alpha, best_score


def ridge_probe(X, y, test_frac=TEST_FRAC, split_seed=SPLIT_SEED):
    """80:20 train/test；alpha 由 train 內 CV 選；回 dict(alpha, cv_r2, test_r2, n_train, n_test)。"""
    n = X.shape[0]
    idx = np.random.default_rng(split_seed).permutation(n)
    n_test = int(round(n * test_frac))
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    Xtr, Xte = X[train_idx], X[test_idx]
    ytr, yte = _to2d(y)[train_idx], _to2d(y)[test_idx]
    Xmean, Xstd = Xtr.mean(0), Xtr.std(0)
    Xstd = np.where(Xstd < 1e-8, 1.0, Xstd)
    Xtr_s, Xte_s = (Xtr - Xmean) / Xstd, (Xte - Xmean) / Xstd
    alpha, cv_r2 = _cv_select_alpha(Xtr_s, ytr)
    beta, Xm, ym = _ridge_fit(Xtr_s, ytr, alpha)
    pred = _ridge_predict(beta, Xm, ym, Xte_s)
    test_r2, r2_dim = _r2_uniform_avg(yte, pred)
    return dict(alpha=alpha, cv_r2=cv_r2, test_r2=test_r2, r2_per_dim=r2_dim.tolist(),
                n_train=len(train_idx), n_test=len(test_idx))


# ─────────────────────────────────────────────────────────────────
# 6. 三目標 × [ix probe / anc 上界 / shuffle 對照]
# ─────────────────────────────────────────────────────────────────
hr("6. 三目標 × [ix probe / anc 上界 / shuffle 對照]")
TARGETS = [
    ("endpoint_cell(2d)", end_cell, False),
    ("path_length(1d)", path_len, False),
    ("net_direction(2d)", net_dir, True),
]
results = {}
for tname, yvals, needs_mask in TARGETS:
    if needs_mask:
        keep = ~degenerate
        y_use, ix_use, ancf_use = yvals[keep], ix[keep], anc_flat[keep]
    else:
        y_use, ix_use, ancf_use = yvals, ix, anc_flat

    res_ix = ridge_probe(ix_use, y_use)
    res_anc = ridge_probe(ancf_use, y_use)
    perm = np.random.default_rng(SHUF_SEED).permutation(len(y_use))
    y_shuf = y_use[perm]
    # 設計選擇（非 file:line 出處）：shuffle 對照跑在 ix 特徵上——它要驗的是「ix probe 這整條
    # pipeline（含 CV 選 alpha）會不會在無關係時也生出虛高 R²」，不是重驗 anc 上界那條 pipeline。
    res_shuf = ridge_probe(ix_use, y_shuf)

    results[tname] = dict(ix=res_ix, anc=res_anc, shuf=res_shuf, n=len(y_use))
    print(f"\n  [{tname}]  n={len(y_use)}")
    for label, r in (("ix probe   ", res_ix), ("anc 上界   ", res_anc), ("shuffle對照", res_shuf)):
        print(f"    {label}  alpha={r['alpha']:<8g} cv_r2={r['cv_r2']:+.3f}  "
              f"test_r2={r['test_r2']:+.3f}  (n_train={r['n_train']} n_test={r['n_test']}"
              f"  per_dim={['%.3f' % v for v in r['r2_per_dim']]})")


# ─────────────────────────────────────────────────────────────────
# 彙總數字表 + 判準判定
# ─────────────────────────────────────────────────────────────────
hr("彙總數字表")
print(f"  {'目標':22s}{'n':>6s}{'ix probe R²':>14s}{'anc 上界 R²':>14s}{'shuffle R²':>14s}")
for tname, _, _ in TARGETS:
    r = results[tname]
    print(f"  {tname:22s}{r['n']:>6d}{r['ix']['test_r2']:>14.3f}"
          f"{r['anc']['test_r2']:>14.3f}{r['shuf']['test_r2']:>14.3f}")

hr("判準判定（任務預釘門檻）")


def verdict(r2):
    if r2 > 0.5:
        return "R²>0.5 ⇒「編碼層看得懂」成立、問題在使用端"
    if r2 < 0.2:
        return "R²<0.2 ⇒「編碼是糊的」成立"
    return "介於中間 ⇒ 部分可讀"


end_ix_r2 = results["endpoint_cell(2d)"]["ix"]["test_r2"]
dir_ix_r2 = results["net_direction(2d)"]["ix"]["test_r2"]
print(f"  終點 cell   ix probe test R² = {end_ix_r2:+.3f}  →  {verdict(end_ix_r2)}")
print(f"  淨方向      ix probe test R² = {dir_ix_r2:+.3f}  →  {verdict(dir_ix_r2)}")

print(f"\n耗時 {time.time() - T0:.1f}s")
