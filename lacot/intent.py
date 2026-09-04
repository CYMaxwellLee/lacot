"""兩層架構的上層底座 — E 格路線（intent）的提取、選路、翻譯。

主人 2026-09-04 裁示：intent 來源先 A（現成 E 格路線）再 B（學字典）；
接法 (i) embed／(ii) per-token 錨／(iii) residual 三臂同日對比。

⛔ 本模組只有幾何，不碰模型、不建圖 —— 佔據圖與 cell↔xy 轉換由呼叫端注入
（rollout 主檔已有 _EOCC／_e_xy_to_cell／_e_cell_to_xy，⛔ 不在此另建第二份）。
⛔ BFS 一律用 lacot.subgoal 的單一來源（grid_bfs／grid_shortest_path），不另寫。

資料流：
  訓練 hindsight：軌跡 xy → traj_to_cells()（相鄰去重）→ cells_to_anchors() [K,2]
  推論選路：    route_cells()（grid_shortest_path 包裝）→ cells_to_anchors()
  統一格式：    anchors_resample() → [T_A, 2] 固定長度 —— 三個接法都只吃這個。
"""
import numpy as np

from .subgoal import grid_shortest_path


def traj_to_cells(traj_xy, xy_to_cell):
    """hindsight 提取：軌跡 xy 序列 → 依訪問順序的 cell 序列（僅去「連續重複」）。

    ⚠️ 邊界抖動（A→B→A）第一版【保留】—— 它是軌跡真實走過的形狀；
       抖動率由 smoke 量（jitter_rate），高到影響再議壓法，⛔ 不預先「治」沒量過的病。
    traj_xy [T,2] array-like；xy_to_cell: xy → (i,j)（呼叫端注入，含 snap 保底）。
    回 list[(i,j)]，len ≥ 1。
    """
    cells = []
    for xy in np.asarray(traj_xy, np.float64):
        c = tuple(xy_to_cell(xy))
        if not cells or c != cells[-1]:
            cells.append(c)
    return cells


def jitter_rate(cells):
    """A→B→A 型抖動佔比（診斷用）：cells[k]==cells[k-2] 的 k 數 / len。"""
    if len(cells) < 3:
        return 0.0
    hits = sum(1 for k in range(2, len(cells)) if cells[k] == cells[k - 2])
    return hits / (len(cells) - 2)


def route_cells(occ, s_cell, g_cell):
    """推論選路：佔據圖上 s→g 的最短 cell 路線（含兩端）。到不了回 None。
    ⭐ 就是 grid_shortest_path —— 包一層是為了讓 intent 的呼叫端不用知道它住哪。"""
    return grid_shortest_path(occ, s_cell, g_cell)


def cells_to_anchors(cells, cell_to_xy):
    """cell 序列 → 錨點 xy 序列 [K,2] np.float64（cell 中心，呼叫端注入轉換）。"""
    return np.asarray([cell_to_xy(c) for c in cells], np.float64).reshape(len(cells), 2)


def anchors_resample(anchors, T):
    """沿弧長線性插值到固定 T 點 [T,2] —— 三接法統一輸入格式。

    K=1（s、g 同格）⇒ tile 成 T 份同一點（路線退化為「原地」，合法）。
    ⛔ 不能按索引均攤 —— 同 arc_subgoal 的理由：錨點間距不均（斜走 vs 直走）。
    """
    A = np.asarray(anchors, np.float64).reshape(-1, 2)
    if len(A) == 1:
        return np.tile(A, (T, 1))
    seg = np.linalg.norm(np.diff(A, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = max(cum[-1], 1e-9)
    t_src = cum / total
    t_dst = np.linspace(0.0, 1.0, T)
    out = np.empty((T, 2), np.float64)
    for k in (0, 1):
        out[:, k] = np.interp(t_dst, t_src, A[:, k])
    return out


def hindsight_intent(traj_xy, xy_to_cell, cell_to_xy, T):
    """訓練用一站式：軌跡段（呼叫端已切好 s..g）→ 重採樣錨點 [T,2]。
    回 (anchors_T, n_cells, jit)：後兩個給診斷欄。"""
    cells = traj_to_cells(traj_xy, xy_to_cell)
    A = cells_to_anchors(cells, cell_to_xy)
    return anchors_resample(A, T), len(cells), jitter_rate(cells)


def route_intent(occ, s_xy, g_xy, xy_to_cell, cell_to_xy, T):
    """推論用一站式：s,g 座標 → E 圖最短路 → 重採樣錨點 [T,2]。
    路不通（理論上 gate 過就不會）⇒ 回 None，呼叫端自己保底。"""
    cells = route_cells(occ, tuple(xy_to_cell(s_xy)), tuple(xy_to_cell(g_xy)))
    if cells is None:
        return None
    return anchors_resample(cells_to_anchors(cells, cell_to_xy), T)
