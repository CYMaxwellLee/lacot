"""walk_verify 共用模組：P0 字典 v1（條件化還原機）＋ 真螞蟻量尺包 ＋ Pillow 畫圖。

對應 docs/PLAN-2026-09-08-walk-first.md（P0/P1/P2）與
docs/DESIGN-2026-09-08-b2-gait-dictionary.md §⑤（decoder 吃「字＋當下狀態」）。

⛔ 不改動任何既有檔案；本檔只被同目錄的 p0_*/p1_* script import。
⛔ 這台 venv 沒裝 matplotlib（已實測），圖一律 Pillow 手畫，不裝套件。

名詞：
  段 (segment) = 連續 seg_len 步的 action（同一 episode 內、不跨邊界、stride=seg_len）。
  obs (29) = concat(qpos(15), qvel(14))  ← 已用真資料驗證逐位元相等。
  軀幹 z = qpos[:, 2]。
"""
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# 沿用 M0 的驗證過色盤（dataviz skill references/palette.md）
PALETTE_CATEGORICAL = [
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100",
    "#e87ba4", "#008300", "#4a3aa7", "#e34948",
]
SEQUENTIAL_BLUE = "#2a78d6"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"

ACT_DIM = 8
OBS_DIM = 29
EP_LEN = 201
GOAL_TOL = 0.5  # ogbench locomaze/maze.py:86 —— ant 的 _goal_tol，非自訂值


def hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


# ------------------------------------------------------------------
# 資料
# ------------------------------------------------------------------
def load_npz(data_dir, name, split="train"):
    fn = f"{name}.npz" if split == "train" else f"{name}-val.npz"
    path = os.path.join(data_dir, fn)
    if not os.path.exists(path):
        raise FileNotFoundError(f"⛔ 找不到資料檔：{path}")
    d = np.load(path)
    out = dict(
        observations=np.asarray(d["observations"], dtype=np.float32),
        actions=np.asarray(d["actions"], dtype=np.float32),
        terminals=np.asarray(d["terminals"], dtype=bool),
        qpos=np.asarray(d["qpos"], dtype=np.float64),
        qvel=np.asarray(d["qvel"], dtype=np.float64),
        _path=path,
    )
    assert out["observations"].shape[1] == OBS_DIM
    assert out["actions"].shape[1] == ACT_DIM
    return out


def episode_bounds(terminals):
    """回傳 (starts, ends)：每個 episode 的首/末 global index（含末）。"""
    ends = np.flatnonzero(terminals)
    starts = np.concatenate([[0], ends[:-1] + 1])
    return starts, ends


def cut_segments(raw, seg_len):
    """切非重疊動作段。回傳 seg_starts（每段第一步的 global index）。

    段的最後一步是 seg_start+seg_len-1，執行完之後的狀態在 seg_start+seg_len，
    所以段不能跨 episode 邊界：要求 seg_start+seg_len <= ep_end+1。
    """
    starts, ends = episode_bounds(raw["terminals"])
    seg_starts = []
    for s0, e0 in zip(starts, ends):
        T = int(e0 - s0 + 1)
        n_chunks = T // seg_len
        for c in range(n_chunks):
            idx0 = int(s0 + c * seg_len)
            if idx0 + seg_len <= e0 + 1:
                seg_starts.append(idx0)
    return np.array(seg_starts, dtype=np.int64)


def build_seg_tensors(raw, seg_starts, seg_len):
    """段的 (動作段展平, 段起點 obs)。"""
    act = raw["actions"]
    obs = raw["observations"]
    idx = seg_starts[:, None] + np.arange(seg_len)[None, :]
    segs = act[idx].reshape(len(seg_starts), seg_len * ACT_DIM).astype(np.float32)
    cond = obs[seg_starts].astype(np.float32)
    return segs, cond


def split_segments(n, split_seed=42, val_frac=0.1):
    """M0 同協定：對「段」做 9:1 隨機切分（split_seed=42），保持與 .212 可比。"""
    rng = np.random.default_rng(split_seed)
    perm = rng.permutation(n)
    n_val = int(n * val_frac)
    return perm[n_val:], perm[:n_val]


# ------------------------------------------------------------------
# 模型：encoder/VQ 照 M0；decoder 改條件化（字 embedding + 正規化 obs）
# ------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class VQEMA(nn.Module):
    """照 M0 的 VQ-EMA（EMA 更新 + dead-code restart），一行未改。"""

    def __init__(self, K, D, decay=0.99, eps=1e-5, dead_steps=500):
        super().__init__()
        self.K, self.D, self.decay, self.eps, self.dead_steps = K, D, decay, eps, dead_steps
        embed = torch.randn(K, D) * 0.1
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(K))
        self.register_buffer("embed_avg", embed.clone())
        self.register_buffer("unused_steps", torch.zeros(K, dtype=torch.long))
        self.register_buffer("restarts_total", torch.zeros(1, dtype=torch.long))

    def quantize_idx(self, z_e):
        dist = (z_e.pow(2).sum(1, keepdim=True) - 2 * z_e @ self.embed.t()
                + self.embed.pow(2).sum(1))
        return dist.argmin(1)

    def forward(self, z_e, training):
        idx = self.quantize_idx(z_e)
        onehot = F.one_hot(idx, self.K).type(z_e.dtype)
        z_q = self.embed[idx]
        if training:
            with torch.no_grad():
                counts = onehot.sum(0)
                self.cluster_size.mul_(self.decay).add_(counts, alpha=1 - self.decay)
                self.embed_avg.mul_(self.decay).add_(onehot.t() @ z_e, alpha=1 - self.decay)
                n = self.cluster_size.sum()
                cs = (self.cluster_size + self.eps) / (n + self.K * self.eps) * n
                self.embed.copy_(self.embed_avg / cs.unsqueeze(1))
                used = counts > 0
                self.unused_steps[used] = 0
                self.unused_steps[~used] += 1
                dead = self.unused_steps >= self.dead_steps
                n_dead = int(dead.sum().item())
                if n_dead > 0:
                    pool = z_e.detach()
                    pick = pool[torch.randint(0, pool.shape[0], (n_dead,))]
                    self.embed[dead] = pick
                    self.embed_avg[dead] = pick
                    self.cluster_size[dead] = 1.0
                    self.unused_steps[dead] = 0
                    self.restarts_total += n_dead
        z_q_st = z_e + (z_q - z_e).detach()
        commit = F.mse_loss(z_e, z_q.detach())
        return z_q_st, idx, commit


class CondVQVAE(nn.Module):
    """字典 v1：encoder(動作段)->z->VQ->字；decoder(字 embedding ‖ 正規化 obs)->動作段。

    obs_mean/obs_std 存在 buffer 裡（P1/P2 一定要用同一份統計，不能重算）。
    """

    def __init__(self, seg_dim, latent_dim, K, obs_dim=OBS_DIM, hidden=512,
                 decay=0.99, dead_steps=500):
        super().__init__()
        self.seg_dim, self.latent_dim, self.K, self.obs_dim = seg_dim, latent_dim, K, obs_dim
        self.encoder = MLP(seg_dim, latent_dim, hidden)
        self.decoder = MLP(latent_dim + obs_dim, seg_dim, hidden)
        self.vq = VQEMA(K, latent_dim, decay=decay, dead_steps=dead_steps)
        self.register_buffer("obs_mean", torch.zeros(obs_dim))
        self.register_buffer("obs_std", torch.ones(obs_dim))

    def set_obs_stats(self, mean, std):
        self.obs_mean.copy_(torch.as_tensor(mean, dtype=torch.float32))
        self.obs_std.copy_(torch.as_tensor(np.maximum(std, 1e-6), dtype=torch.float32))

    def norm_obs(self, obs):
        return (obs - self.obs_mean) / self.obs_std

    def decode_from_idx(self, idx, obs):
        """字 index (B,) ＋ 原始 obs (B,29) → 動作段 (B, seg_dim)。P1/P2 執行端用這支。"""
        z_q = self.vq.embed[idx]
        return self.decoder(torch.cat([z_q, self.norm_obs(obs)], dim=1))

    def decode_all_codes(self, obs):
        """同一個 obs 展開全部 K 個字 → (K, seg_dim)。P2 前瞻枚舉用。"""
        z_q = self.vq.embed  # (K, D)
        o = self.norm_obs(obs).expand(self.K, -1)
        return self.decoder(torch.cat([z_q, o], dim=1))

    def encode_idx(self, x):
        return self.vq.quantize_idx(self.encoder(x))

    def forward(self, x, obs, training):
        z_e = self.encoder(x)
        z_q_st, idx, commit = self.vq(z_e, training=training)
        recon = self.decoder(torch.cat([z_q_st, self.norm_obs(obs)], dim=1))
        return recon, idx, commit


def load_dict_v1(ckpt_path, map_location="cpu"):
    ck = torch.load(ckpt_path, map_location=map_location, weights_only=False)
    cfg = ck["config"]
    model = CondVQVAE(cfg["seg_dim"], cfg["latent_dim"], cfg["k"], obs_dim=cfg["obs_dim"],
                      hidden=cfg["hidden"], decay=cfg["decay"], dead_steps=cfg["dead_steps"])
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, cfg


# ------------------------------------------------------------------
# 真螞蟻量尺包
# ------------------------------------------------------------------
def build_ruler(raw, dist_grid=None, quantiles=(1, 5, 10, 25, 50, 75, 90, 95, 99),
                max_horizon=200, sub=1, seed=0):
    """從資料量出四把尺。

    (a) distance -> steps 查表：對每個起點，第一次「離起點的直線距離 >= d」所花的步數
        （first-passage time）。對每個 d 取分位數。⚠️ 沒在 episode 內達到 d 的起點
        算 censored，另外回報比例（不填預設值）。
    (b) 步速：每步 xy 位移 |xy_{t+1}-xy_t|
    (c) 軀幹 z 高度：qpos[:,2]；翻倒線 = p1
    (d) 相鄰步動作差：|a_t - a_{t-1}|（L2 over 8 dims），不跨 episode
    """
    if dist_grid is None:
        dist_grid = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]
    qpos = raw["qpos"]
    act = raw["actions"]
    starts, ends = episode_bounds(raw["terminals"])
    xy = qpos[:, :2]

    # (b) 步速（同 episode 內）
    step_disp = []
    for s0, e0 in zip(starts, ends):
        seg = xy[s0:e0 + 1]
        step_disp.append(np.linalg.norm(np.diff(seg, axis=0), axis=1))
    step_disp = np.concatenate(step_disp)

    # (c) 軀幹 z
    z = qpos[:, 2]

    # (d) 相鄰動作差（同 episode 內）
    act_diff = []
    for s0, e0 in zip(starts, ends):
        a = act[s0:e0 + 1]
        act_diff.append(np.linalg.norm(np.diff(a, axis=0), axis=1))
    act_diff = np.concatenate(act_diff)

    # (a) distance -> first-passage steps（向量化：每個 episode 算一次 (T,T) 距離矩陣）
    rng = np.random.default_rng(seed)
    n_ep = len(starts)
    ep_sel = np.arange(n_ep) if sub <= 1 else np.sort(
        rng.choice(n_ep, size=max(1, n_ep // sub), replace=False))
    hits_by_d = {f"{d:g}": [] for d in dist_grid}
    n_total_by_d = {f"{d:g}": 0 for d in dist_grid}
    n_cens_by_d = {f"{d:g}": 0 for d in dist_grid}
    for ei in ep_sel:
        s0, e0 = int(starts[ei]), int(ends[ei])
        seg = xy[s0:e0 + 1]
        T = len(seg)
        D = np.linalg.norm(seg[None, :, :] - seg[:, None, :], axis=2)  # (T,T)
        ii = np.arange(T)
        # 只看 i < j <= i+max_horizon
        gap = ii[None, :] - ii[:, None]
        window = (gap >= 1) & (gap <= max_horizon)
        for d in dist_grid:
            key = f"{d:g}"
            hit = (D >= d) & window
            any_hit = hit.any(1)
            first_j = hit.argmax(1)
            rows = np.flatnonzero(any_hit)
            hits_by_d[key].append((first_j[rows] - ii[rows]).astype(np.int64))
            n_total_by_d[key] += T - 1  # 起點 i=0..T-2 才有前進的機會
            n_cens_by_d[key] += int((~any_hit[: T - 1]).sum())
    table = {}
    for d in dist_grid:
        key = f"{d:g}"
        hits = np.concatenate(hits_by_d[key]) if hits_by_d[key] else np.array([], dtype=np.int64)
        n_total, n_cens = n_total_by_d[key], n_cens_by_d[key]
        table[key] = dict(
            d=float(d),
            n_reached=int(len(hits)),
            n_total=int(n_total),
            censored_frac=float(n_cens / max(n_total, 1)),
            quantiles={str(q): (float(np.percentile(hits, q)) if len(hits) else None)
                       for q in quantiles},
            mean=float(hits.mean()) if len(hits) else None,
        )

    def qdict(a):
        return {str(q): float(np.percentile(a, q)) for q in quantiles}

    return dict(
        dataset=raw["_path"],
        n_episodes=int(len(starts)),
        distance_to_steps=dict(grid=[float(x) for x in dist_grid], table=table,
                               max_horizon=int(max_horizon), episode_subsample=int(sub)),
        step_speed=dict(n=int(len(step_disp)), mean=float(step_disp.mean()),
                        std=float(step_disp.std()), quantiles=qdict(step_disp)),
        torso_z=dict(n=int(len(z)), mean=float(z.mean()), std=float(z.std()),
                     quantiles=qdict(z), fall_line_p1=float(np.percentile(z, 1))),
        action_smoothness=dict(n=int(len(act_diff)), mean=float(act_diff.mean()),
                               std=float(act_diff.std()), quantiles=qdict(act_diff)),
        goal_tol=GOAL_TOL,
    )


def ruler_steps_for_distance(ruler, d, q="50"):
    """從查表線性內插出「走 d 距離的第 q 分位步數」。超出格點就外插最近端（會標記）。"""
    grid = np.asarray(ruler["distance_to_steps"]["grid"], dtype=float)
    tbl = ruler["distance_to_steps"]["table"]
    ys = np.array([tbl[f"{g:g}"]["quantiles"][q] for g in grid], dtype=float)
    ok = np.isfinite(ys)
    grid, ys = grid[ok], ys[ok]
    if d <= grid[0]:
        return float(ys[0] * max(d, 1e-6) / grid[0]), "below_grid"
    if d >= grid[-1]:
        return float(ys[-1] * d / grid[-1]), "above_grid"
    return float(np.interp(d, grid, ys)), "in_grid"


# ------------------------------------------------------------------
# Pillow 畫圖（這台沒 matplotlib）
# ------------------------------------------------------------------
def _font():
    from PIL import ImageFont
    return ImageFont.load_default()


def draw_loss_curve(curves, out_path, title, ylog=True, ylabel="loss"):
    """curves: list of dict(label=..., xs=[...], ys=[...])"""
    from PIL import Image, ImageDraw
    W, H = 960, 480
    ml, mr, mt, mb = 80, 200, 46, 56
    pw, ph = W - ml - mr, H - mt - mb
    img = Image.new("RGB", (W, H), hex_to_rgb(SURFACE))
    dr = ImageDraw.Draw(img)
    ft = _font()
    allx = np.concatenate([np.asarray(c["xs"], dtype=float) for c in curves])
    ally = np.concatenate([np.asarray(c["ys"], dtype=float) for c in curves])
    ally = ally[np.isfinite(ally) & (ally > 0 if ylog else np.isfinite(ally))]
    x0, x1 = float(allx.min()), float(allx.max())
    y0, y1 = float(ally.min()), float(ally.max())
    if ylog:
        y0, y1 = np.log10(max(y0, 1e-12)), np.log10(max(y1, 1e-11))
    pad = (y1 - y0) * 0.08 or 1.0
    y0, y1 = y0 - pad, y1 + pad

    def px(x):
        return ml + pw * (x - x0) / max(x1 - x0, 1e-9)

    def py(y):
        v = np.log10(max(y, 1e-12)) if ylog else y
        return mt + ph * (1 - (v - y0) / max(y1 - y0, 1e-9))

    for f in np.linspace(0, 1, 6):
        yy = mt + ph * (1 - f)
        dr.line([(ml, yy), (ml + pw, yy)], fill=hex_to_rgb(GRIDLINE))
        val = y0 + f * (y1 - y0)
        lab = f"{10**val:.4g}" if ylog else f"{val:.4g}"
        dr.text((6, yy - 6), lab, fill=hex_to_rgb(INK_MUTED), font=ft)
    for f in np.linspace(0, 1, 6):
        xx = ml + pw * f
        dr.line([(xx, mt), (xx, mt + ph)], fill=hex_to_rgb(GRIDLINE))
        dr.text((xx - 18, mt + ph + 6), f"{x0 + f*(x1-x0):.0f}", fill=hex_to_rgb(INK_MUTED), font=ft)
    for i, c in enumerate(curves):
        col = hex_to_rgb(PALETTE_CATEGORICAL[i % 8])
        pts = [(px(x), py(y)) for x, y in zip(c["xs"], c["ys"]) if np.isfinite(y) and y > 0]
        if len(pts) > 1:
            dr.line(pts, fill=col, width=2)
        ly = mt + 8 + i * 18
        dr.line([(ml + pw + 14, ly), (ml + pw + 34, ly)], fill=col, width=3)
        dr.text((ml + pw + 38, ly - 6), c["label"], fill=hex_to_rgb(INK_SECONDARY), font=ft)
    dr.line([(ml, mt), (ml, mt + ph)], fill=hex_to_rgb(AXIS), width=2)
    dr.line([(ml, mt + ph), (ml + pw, mt + ph)], fill=hex_to_rgb(AXIS), width=2)
    dr.text((ml, 14), title, fill=hex_to_rgb(INK_PRIMARY), font=ft)
    dr.text((ml, H - 18), "train step", fill=hex_to_rgb(INK_MUTED), font=ft)
    dr.text((6, 14), ylabel + (" (log)" if ylog else ""), fill=hex_to_rgb(INK_MUTED), font=ft)
    img.save(out_path)


def draw_hist_overlay(series, out_path, title, xlabel, bins=60, xrange=None, vlines=None):
    """疊圖：多條分佈的機率密度折線（同一組 bin），Pillow 手畫。

    series: list of dict(label=..., values=np.array)
    vlines: list of dict(x=..., label=..., color_idx=...)
    """
    from PIL import Image, ImageDraw
    W, H = 960, 480
    ml, mr, mt, mb = 74, 210, 46, 56
    pw, ph = W - ml - mr, H - mt - mb
    img = Image.new("RGB", (W, H), hex_to_rgb(SURFACE))
    dr = ImageDraw.Draw(img)
    ft = _font()
    allv = np.concatenate([np.asarray(s["values"], dtype=float) for s in series])
    allv = allv[np.isfinite(allv)]
    if xrange is None:
        lo, hi = float(np.percentile(allv, 0.2)), float(np.percentile(allv, 99.8))
    else:
        lo, hi = xrange
    if hi <= lo:
        hi = lo + 1e-6
    edges = np.linspace(lo, hi, bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    dens = []
    for s in series:
        v = np.asarray(s["values"], dtype=float)
        v = v[np.isfinite(v)]
        h, _ = np.histogram(np.clip(v, lo, hi), bins=edges)
        dens.append(h / max(h.sum(), 1))
    ymax = max(float(d.max()) for d in dens) * 1.15 or 1.0

    def px(x):
        return ml + pw * (x - lo) / (hi - lo)

    def py(y):
        return mt + ph * (1 - y / ymax)

    for f in np.linspace(0, 1, 6):
        yy = mt + ph * (1 - f)
        dr.line([(ml, yy), (ml + pw, yy)], fill=hex_to_rgb(GRIDLINE))
        dr.text((6, yy - 6), f"{f*ymax*100:.2f}%", fill=hex_to_rgb(INK_MUTED), font=ft)
    for f in np.linspace(0, 1, 7):
        xx = ml + pw * f
        dr.line([(xx, mt), (xx, mt + ph)], fill=hex_to_rgb(GRIDLINE))
        dr.text((xx - 16, mt + ph + 6), f"{lo + f*(hi-lo):.3g}", fill=hex_to_rgb(INK_MUTED), font=ft)
    for i, (s, d) in enumerate(zip(series, dens)):
        col = hex_to_rgb(PALETTE_CATEGORICAL[i % 8])
        pts = [(px(c), py(v)) for c, v in zip(centers, d)]
        dr.line(pts, fill=col, width=2)
        ly = mt + 8 + i * 18
        dr.line([(ml + pw + 14, ly), (ml + pw + 34, ly)], fill=col, width=3)
        dr.text((ml + pw + 38, ly - 6), s["label"], fill=hex_to_rgb(INK_SECONDARY), font=ft)
    for j, vl in enumerate(vlines or []):
        if not (lo <= vl["x"] <= hi):
            continue
        col = hex_to_rgb(PALETTE_CATEGORICAL[vl.get("color_idx", 7) % 8])
        xx = px(vl["x"])
        for yy in range(int(mt), int(mt + ph), 8):
            dr.line([(xx, yy), (xx, yy + 4)], fill=col, width=2)
        dr.text((xx + 3, mt + 4 + 14 * j), vl["label"], fill=col, font=ft)
    dr.line([(ml, mt), (ml, mt + ph)], fill=hex_to_rgb(AXIS), width=2)
    dr.line([(ml, mt + ph), (ml + pw, mt + ph)], fill=hex_to_rgb(AXIS), width=2)
    dr.text((ml, 14), title, fill=hex_to_rgb(INK_PRIMARY), font=ft)
    dr.text((ml, H - 18), xlabel, fill=hex_to_rgb(INK_MUTED), font=ft)
    img.save(out_path)


def draw_usage_histogram(usage_frac, out_path, title):
    from PIL import Image, ImageDraw
    K = len(usage_frac)
    W, H = max(560, 40 * K + 160), 420
    ml, mr, mt, mb = 70, 30, 50, 60
    pw, ph = W - ml - mr, H - mt - mb
    img = Image.new("RGB", (W, H), hex_to_rgb(SURFACE))
    dr = ImageDraw.Draw(img)
    ft = _font()
    top = (max(usage_frac) or 1.0) * 1.18
    for g in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = mt + ph - ph * g
        dr.line([(ml, y), (ml + pw, y)], fill=hex_to_rgb(GRIDLINE))
        dr.text((8, y - 6), f"{g*top*100:.1f}%", fill=hex_to_rgb(INK_MUTED), font=ft)
    bw = pw / K
    for i, v in enumerate(usage_frac):
        x0 = ml + i * bw + bw * 0.15
        x1 = ml + (i + 1) * bw - bw * 0.15
        y1 = mt + ph
        dr.rectangle([x0, y1 - ph * (v / top), x1, y1], fill=hex_to_rgb(SEQUENTIAL_BLUE),
                     outline=hex_to_rgb(INK_PRIMARY))
        if K <= 34:
            dr.text(((x0 + x1) / 2 - 8, y1 + 4), str(i), fill=hex_to_rgb(INK_SECONDARY), font=ft)
    dr.line([(ml, mt), (ml, mt + ph)], fill=hex_to_rgb(AXIS), width=2)
    dr.line([(ml, mt + ph), (ml + pw, mt + ph)], fill=hex_to_rgb(AXIS), width=2)
    dr.text((ml, 14), title, fill=hex_to_rgb(INK_PRIMARY), font=ft)
    dr.text((ml, H - 18), "code index", fill=hex_to_rgb(INK_MUTED), font=ft)
    img.save(out_path)


def save_json(obj, path):
    def _default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_default)


# ------------------------------------------------------------------
# MuJoCo 執行工具（P1/P2 共用）
# ------------------------------------------------------------------
def make_env(dataset_dir, name="antmaze-medium-stitch-v0", render=False):
    """建立 antmaze env（env_only）。render=True 前必須已設 MUJOCO_GL。"""
    import ogbench
    env = ogbench.make_env_and_datasets(name, dataset_dir=dataset_dir, env_only=True)
    env.reset(seed=0)
    return env
