"""A0 可行性探針：antmaze-large-stitch-v0 移植前測（唯讀，2026-09-04 主人交辦）。

問題：pointmaze 上「資料重建佔據圖＋BFS」這條上層路線（GEO 佔據圖 → grid_bfs 找路）
站得住；搬到 antmaze-large-stitch-v0 上，同一套幾何工具還適不適用？這支腳本只量數字，
不碰任何訓練、不碰 GPU、不送 Slurm job。

量五件事（章節化印出，都在 stdout）：
    1. 資料下載＋train/val 基本形狀（shape／episode 數／步長分佈／dtype）
    2. xy 範圍與尺度 ＋ 29 維逐維 std（qpos 15 維 vs qvel 14 維量級對比，給 e_target normalize 用）
    3. 佔據圖 gate：res∈{6,8,10} 各建一版，量自由格覆蓋率、最大連通分量佔比、
       100 對隨機 (s,g) 的 BFS 連通率、端點 snap 率
    4. 那 100 對的 BFS 細格步數分佈（p50/p90/max，給 T_A 選型用）
    5. 動作空間（shape／dtype／min/max，確認 8 維 [-1,1]）

⛔ 唯讀探針，不改任何既有檔案：
    - experiments/scratch_lacot_rollout.py、lacot/ 底下的既有模組全部不動。
    - 佔據圖沿用 lacot.refine_grad.GeoEnergy 的既有形狀 —— res＝每格切幾份，網格解析度
      綁在資料實際跨度上（GeoEnergy 內部邏輯，這支腳本原樣重用、不重寫）。
      GeoEnergy docstring 本來就寫 obs_xy:[N,2]，pointmaze 那邊能整個 OBS 塞進去是因為
      pointmaze 的 obs 本身就只有 2 維；antmaze obs 是 29 維，所以這裡老實地先切
      obs[:,:2] 再餵進去 —— 這不是另一套 cell 定義，是照 docstring 原本的用法用。
    - BFS 沿用 lacot.subgoal.grid_bfs（單一來源，這支腳本直接 import 用，不複製一份）。
    - snap／連通率／最大連通分量的量法，照 experiments/scratch_lacot_rollout.py 裡
      _EOCC 那段（約 1624-1683 行）的既有慣例抄寫法（不是抄檔案本身，是抄同一套邏輯），
      這樣才跟 pointmaze 8/30 那次 gate 探針的數字可比。

跑法：
    cd ~/Projects/lacot
    OGBENCH_DATA_DIR=/home/cymaxwelllee/data/ogbench \
    /home/cymaxwelllee/venvs/lacot-rocm/bin/python experiments/probe_antmaze_a0.py \
        2>&1 | tee experiments/probe_antmaze_a0_report.txt
"""
import os
import shutil
import sys
import time

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
# ⛔ 唯讀重用：只 import，不改這兩個模組任何一行。
from lacot.refine_grad import GeoEnergy   # noqa: E402
from lacot.subgoal import grid_bfs        # noqa: E402

T0 = time.time()

ENV_NAME = os.environ.get("LACOT_ENV", "antmaze-large-stitch-v0")
OGB_DATA = os.environ.get("OGBENCH_DATA_DIR", "/home/cymaxwelllee/data/ogbench")
os.environ.setdefault("OGBENCH_DATA_DIR", OGB_DATA)
MIN_FREE_GB = 20.0
RES_LIST = [6, 8, 10]
N_PAIRS = 100
SEED = 0


def hr(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def pct(arr, q):
    return float(np.percentile(arr, q)) if len(arr) else float("nan")


def episode_stats(terminals):
    """terminals[i]==1 標一個 episode 的最後一步。回 (episode 數, 每集長度陣列)。
    ⚠️ 查不到就標註，不裝沒事：terminal 型態或收尾不對會印警告，不會靜默吞掉。"""
    terminals = np.asarray(terminals).astype(bool)
    n = len(terminals)
    ends = np.flatnonzero(terminals)
    if len(ends) == 0:
        print("  ⚠️ 找不到任何 terminal=1 的步 ⇒ 無法切 episode，episode 數字不可信")
        return 0, np.array([])
    if ends[-1] != n - 1:
        print(f"  ⚠️ 最後一筆 idx={n - 1} 不是 terminal（最後一個 terminal 在 idx={ends[-1]}）"
              f" ⇒ 資料尾巴可能有未結束的殘集，episode 數字打折看")
    starts = np.concatenate([[0], ends[:-1] + 1])
    lens = ends - starts + 1
    return len(ends), lens


# ─────────────────────────────────────────────────────────────────
# 0. 磁碟空間檢查（下載前）
# ─────────────────────────────────────────────────────────────────
hr("0. 磁碟空間檢查（下載前）")
total_b, used_b, free_b = shutil.disk_usage("/")
free_gb = free_b / (1024 ** 3)
print(f"  df -h / 可用空間：{free_gb:.1f} GB（門檻 {MIN_FREE_GB:g} GB）")
os.makedirs(OGB_DATA, exist_ok=True)
same_dev = os.stat("/").st_dev == os.stat(OGB_DATA).st_dev
print(f"  OGBENCH_DATA_DIR={OGB_DATA}  與 / 同一個檔案系統：{same_dev}"
      f"{'' if same_dev else '  ⚠️ 不同 filesystem，上面那格空間數字不代表這裡的餘量！'}")
if free_gb < MIN_FREE_GB:
    print(f"  ⛔ 空間不足 {MIN_FREE_GB:g} GB，停手不下載。")
    sys.exit(1)
print("  ✓ 空間足夠，繼續。")


# ─────────────────────────────────────────────────────────────────
# 1. 資料下載＋基本形狀
# ─────────────────────────────────────────────────────────────────
hr(f"1. 資料下載＋基本形狀（{ENV_NAME}）")
print("  下載中斷可安全重跑：ogbench 的 download_datasets 先寫 .tmp 檔，成功才 os.rename")
print("  成正式檔名；中斷只留半成品 .tmp，正式檔名仍不存在 ⇒ 重跑這支腳本會自動整份重抓。")
import ogbench  # noqa: E402  （刻意晚 import：確保上面磁碟檢查一定先跑）

try:
    train_ds, val_ds = ogbench.make_env_and_datasets(
        ENV_NAME, dataset_dir=OGB_DATA, dataset_only=True)
except Exception as e:
    print(f"  ⛔ 下載/載入失敗：{type(e).__name__}: {e}")
    print("  ⇒ 重跑這支腳本本身就是重試（見上面兩行說明），不用手動清理。")
    raise

for split_name, ds in (("train", train_ds), ("val", val_ds)):
    obs = np.asarray(ds["observations"])
    act = np.asarray(ds["actions"])
    n_ep, lens = episode_stats(ds["terminals"])
    print(f"  [{split_name}] observations {obs.shape} dtype={obs.dtype}"
          f"   actions {act.shape} dtype={act.dtype}")
    if len(lens):
        print(f"  [{split_name}] episode 數={n_ep}   步長 p50={pct(lens, 50):.0f}"
              f" p90={pct(lens, 90):.0f} max={int(lens.max())} min={int(lens.min())}")
    else:
        print(f"  [{split_name}] episode 數={n_ep}   步長：n/a")
    extra_keys = [k for k in ds.keys()
                  if k not in ("observations", "actions", "terminals", "next_observations")]
    if extra_keys:
        print(f"  [{split_name}] 其他欄位：{extra_keys}")

OBS = np.asarray(train_ds["observations"], np.float64)
ACT = np.asarray(train_ds["actions"], np.float64)
OBS_DIM = OBS.shape[1]
print(f"\n  train obs 維度={OBS_DIM}（預期 29：{'match' if OBS_DIM == 29 else '⚠️ 不符，見第 2 節降級處理'}）")


# ─────────────────────────────────────────────────────────────────
# 2. xy 範圍與尺度 ＋ 29 維逐維 std（qpos vs qvel）
# ─────────────────────────────────────────────────────────────────
hr("2. xy 範圍與尺度 ＋ 逐維 std（qpos vs qvel 量級對比，給 e_target normalize 決策用）")
xy = OBS[:, :2]
xy_min, xy_max = xy.min(0), xy.max(0)
print(f"  x: [{xy_min[0]:.3f}, {xy_max[0]:.3f}]  span={xy_max[0] - xy_min[0]:.3f}")
print(f"  y: [{xy_min[1]:.3f}, {xy_max[1]:.3f}]  span={xy_max[1] - xy_min[1]:.3f}")

std_all = OBS.std(0)
print(f"\n  逐維 std（共 {OBS_DIM} 維，dim0/1＝xy）：")
print("  " + "  ".join(f"{s:.3f}" for s in std_all))

split = 15
if OBS_DIM != 29:
    print(f"  ⚠️ obs 維度={OBS_DIM}≠預期的 29 ⇒ 下面 qpos/qvel 15/{OBS_DIM - 15} 切法是硬猜的，"
          f"不保證對齊真正的 qpos/qvel 邊界，僅供參考")
qpos_std, qvel_std = std_all[:split], std_all[split:]
print(f"\n  qpos(前{split}維) vs qvel(後{OBS_DIM - split}維) std 量級對比：")
print(f"  {'':10s}{'min':>10s}{'median':>10s}{'max':>10s}{'mean':>10s}")
for name, arr in (("qpos", qpos_std), ("qvel", qvel_std)):
    if len(arr) == 0:
        print(f"  {name:10s}  (空)")
        continue
    print(f"  {name:10s}{arr.min():10.4f}{np.median(arr):10.4f}{arr.max():10.4f}{arr.mean():10.4f}")
if len(qpos_std) and len(qvel_std):
    ratio = qvel_std.mean() / max(qpos_std.mean(), 1e-9)
    print(f"  qvel/qpos 平均 std 比值 ≈ {ratio:.2f}"
          f"（差很多 ⇒ e_target 若不逐維 normalize，qvel 或 qpos 其中一邊會被稀釋）")


# ─────────────────────────────────────────────────────────────────
# 3+4. 佔據圖 gate（res 掃 6/8/10）＋ BFS 路線長度分佈
# ─────────────────────────────────────────────────────────────────
hr("3+4. 佔據圖 gate（res∈{6,8,10}，沿用 GeoEnergy 既有 cell 定義）＋ BFS 路線長度分佈")
print("  ⛔ 只 import lacot.refine_grad.GeoEnergy／lacot.subgoal.grid_bfs 直接用，")
print("     不另寫第二套 cell 定義；xy 先切 obs[:,:2] 才餵進 GeoEnergy（docstring 本來的用法）。")

mu_xy = xy.mean(0)
sd_xy = xy.std(0) + 1e-6
print(f"\n  xy normalize：mu=({mu_xy[0]:.3f}, {mu_xy[1]:.3f})  sd=({sd_xy[0]:.3f}, {sd_xy[1]:.3f})")

rng = np.random.default_rng(SEED)
pair_idx = rng.integers(0, len(xy), size=(N_PAIRS, 2))
print(f"  100 對 (s,g)：seed={SEED}，從 train 的 {len(xy)} 筆 transition 均勻抽 index pair"
      f"（同一組 index 三個 res 共用，可互相比較；不分同集/跨集）")

gate_rows = []
for res in RES_LIST:
    print(f"\n  --- res={res} ---")
    geo = GeoEnergy(xy, mu_xy, sd_xy, res=res, device="cpu")
    occ = (geo.dist[0, 0].cpu().numpy() == 0.0)
    shape = tuple(int(s) for s in geo.shape)
    free = int(occ.sum())
    total = int(occ.size)
    coverage = free / total
    print(f"  grid shape={shape}  free={free}/{total}  coverage={coverage:.1%}"
          f"（GEO.coverage 自報={geo.coverage:.1%}，"
          f"{'一致' if abs(geo.coverage - coverage) < 1e-9 else '⚠️ 不一致，數字對不上'}）")

    gh = geo.health()
    print(f"  GEO.health()：ok={gh['ok']}  mapping_err={gh['mapping_err']:.2e}"
          f"  盒內隨機點穿牆中位={gh['wall_median_random']:.4f}"
          + ("" if gh["ok"] else "\n    ⚠️ " + "；".join(gh["reasons"])
             + "（pointmaze 用的門檻，antmaze 幾何不同不一定適用，僅供參考不當硬性擋門）"))

    free_cells = np.argwhere(occ)
    shp_arr = np.asarray(geo.shape, np.int64)

    def xy_to_cell(pt, _occ=occ, _free_cells=free_cells, _shp=shp_arr, _geo=geo):
        """原始 xy → E 細格；落牆格就 snap 到最近自由格。跟 scratch_lacot_rollout.py
        _EOCC 段 _e_xy_to_cell 同一套算法（唯讀參考、這裡重寫一份，不改原檔）。"""
        z = (np.asarray(pt[:2], np.float64) - mu_xy) / sd_xy
        idx = np.clip(np.round((z - _geo.lo) / (_geo.hi - _geo.lo) * (_shp - 1)).astype(int),
                      0, _shp - 1)
        c = tuple(idx)
        if _occ[c]:
            return c, False
        nn = _free_cells[int(np.abs(_free_cells - idx).sum(1).argmin())]
        return tuple(nn), True

    # 最大連通分量佔比：BFS 從資料首格（照 spec 指定的量法）
    src0, snap0 = xy_to_cell(xy[0])
    reach0 = grid_bfs(occ, src0)
    ratio_bfs_first = len(reach0) / free
    print(f"  BFS 從資料首格（cell={src0}，snap={snap0}）：reach={len(reach0)}/{free}"
          f" = {ratio_bfs_first:.1%}")

    # 交叉驗證：scipy.ndimage.label 抓真正的最大連通分量（同一份 occ、同一套 cell，
    # 只是換一個演算法驗證「資料首格」是否真的代表最大塊 —— 不是另一套 cell 定義）。
    from scipy.ndimage import label
    struct4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    lbl, ncc = label(occ, structure=struct4)
    sizes = np.bincount(lbl.ravel())
    sizes[0] = 0
    largest_id = int(sizes.argmax())
    largest_size = int(sizes[largest_id])
    ratio_label = largest_size / free
    in_largest = bool(lbl[src0] == largest_id)
    print(f"  scipy.label 交叉驗證：{ncc} 個連通分量，最大分量={largest_size}/{free}"
          f"={ratio_label:.1%}（資料首格{'✓ 在' if in_largest else '⚠️ 不在'}最大分量內）")

    n_snap = 0
    n_connect = 0
    path_lens = []
    for a, b in pair_idx:
        ca, sa = xy_to_cell(xy[a])
        cb, sb = xy_to_cell(xy[b])
        n_snap += int(sa) + int(sb)
        dist_map = grid_bfs(occ, ca)
        if cb in dist_map:
            n_connect += 1
            path_lens.append(dist_map[cb])
    snap_rate = n_snap / (2 * N_PAIRS)
    connect_rate = n_connect / N_PAIRS
    path_lens = np.array(path_lens)
    print(f"  100 對 (s,g)：BFS 連通率={connect_rate:.1%}（{n_connect}/{N_PAIRS}）"
          f"   端點 snap 率={snap_rate:.1%}（{n_snap}/{2 * N_PAIRS} 端點落牆格）")
    if len(path_lens):
        print(f"  BFS 細格步數（連通的 {len(path_lens)} 對）："
              f"p50={pct(path_lens, 50):.1f} p90={pct(path_lens, 90):.1f} max={int(path_lens.max())}")
    else:
        print("  ⚠️ 沒有任何一對連通，無法算路線長度分佈")

    gate_rows.append(dict(
        res=res, shape=shape, coverage=coverage, ratio_bfs_first=ratio_bfs_first,
        ratio_label=ratio_label, health_ok=gh["ok"], connect_rate=connect_rate,
        snap_rate=snap_rate, path_p50=pct(path_lens, 50) if len(path_lens) else float("nan"),
        path_p90=pct(path_lens, 90) if len(path_lens) else float("nan"),
        path_max=int(path_lens.max()) if len(path_lens) else -1))

hr("3+4 彙總表")
print(f"  {'res':>4s}{'shape':>14s}{'coverage':>10s}{'最大CC':>9s}{'連通率':>8s}{'snap率':>8s}"
      f"{'p50':>7s}{'p90':>7s}{'max':>6s}")
for row in gate_rows:
    print(f"  {row['res']:>4d}{str(row['shape']):>14s}{row['coverage']:>9.1%} "
          f"{row['ratio_label']:>8.1%} {row['connect_rate']:>7.1%} {row['snap_rate']:>7.1%}"
          f"{row['path_p50']:>7.1f}{row['path_p90']:>7.1f}{row['path_max']:>6d}")


# ─────────────────────────────────────────────────────────────────
# 5. 動作空間
# ─────────────────────────────────────────────────────────────────
hr("5. 動作空間")
print(f"  actions shape={ACT.shape} dtype={train_ds['actions'].dtype}")
dim_ok = ACT.shape[1] == 8
range_ok = (-1.0001 <= ACT.min()) and (ACT.max() <= 1.0001)
print(f"  全域 min={ACT.min():.4f}  max={ACT.max():.4f}"
      f"  （預期 8 維、[-1,1]：維度{'match' if dim_ok else '⚠️ 不符'}、"
      f"範圍{'match' if range_ok else '⚠️ 超出 [-1,1]'}）")
print("  逐維 min/max：")
for d in range(ACT.shape[1]):
    print(f"    dim{d}: [{ACT[:, d].min():.4f}, {ACT[:, d].max():.4f}]")

hr(f"探針完畢，耗時 {time.time() - T0:.1f}s")
