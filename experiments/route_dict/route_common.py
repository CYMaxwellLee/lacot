"""路線字典（M0R）共用模組（見 docs/PLAN-2026-09-08-walk-first.md 第一波 P3）。

任務 B 的獨立實作，⛔ 不 import m0_gait_dict/ 或 gk_scan/（各自自成一份，允許
重複少量常數/樣式以保持自我完整，見兩份任務說明的「新檔只放各自目錄」規定）。

職責：
  1. 從 antmaze-medium-stitch-v0 的 qpos xy 軌跡切「固定弧長段」（弧長 7.5，
     不跨 episode），每段重取樣成 16 個 xy 點，並做「起點為原點、段方向歸一化」
     的座標轉換（純幾何，零主模型、零 obs 條件化）。
  2. 單一 codebook VQ-VAE（K ∈ {8,16,32}）+ VQ-EMA（同 A 的實作）。
  3. 訓練與量測同任務 A（重建 MSE、baseline、perplexity/存活數、plateau 判準）。
  4. Pillow 手畫：K 個路線字疊在同一張平面圖 + 3 個字各疊 5 條真實段。

⛔ CPU-only。
"""
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

EP_LEN = 201
DATASET_NAME = "antmaze-medium-stitch-v0"
ARC_LENGTH = 7.5
N_RESAMPLE = 16

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"


def hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))


def distinct_colors(n):
    """HSV colorwheel 均分出 n 個視覺可分色（stdlib colorsys，不裝套件）。"""
    import colorsys

    cols = []
    for i in range(n):
        h = i / max(n, 1)
        r, g, b = colorsys.hsv_to_rgb(h, 0.72, 0.82)
        cols.append((int(r * 255), int(g * 255), int(b * 255)))
    return cols


# ------------------------------------------------------------------
# 資料：固定弧長切段 + 重取樣 + 起點/方向歸一化
# ------------------------------------------------------------------
def _episode_bounds(terminals):
    ends = np.flatnonzero(terminals)
    starts = np.concatenate([[0], ends[:-1] + 1])
    return starts, ends


def extract_route_segments(qpos_xy, terminals, arc_length=ARC_LENGTH, n_resample=N_RESAMPLE):
    starts, ends = _episode_bounds(terminals)
    n_ep = len(starts)
    ep_len = int(ends[0] - starts[0] + 1)
    grid_ok = ep_len == EP_LEN and np.array_equal(starts, np.arange(0, n_ep * ep_len, ep_len)) and (
        n_ep * ep_len == len(terminals)
    )
    if not grid_ok:
        # 通用 fallback：不假設格子，仍用同一套逐 episode 邏輯（下方迴圈本身就不依賴格子假設）。
        pass

    segs = []
    n_degenerate_rotation = 0
    n_segments_per_ep = []
    for s0, e0 in zip(starts, ends):
        traj = qpos_xy[s0 : e0 + 1].astype(np.float64)  # (T, 2)
        deltas = np.diff(traj, axis=0)
        step_len = np.linalg.norm(deltas, axis=1)
        cum = np.concatenate([[0.0], np.cumsum(step_len)])
        total = cum[-1]
        n_seg = int(total // arc_length)
        n_segments_per_ep.append(n_seg)
        for si in range(n_seg):
            a0 = si * arc_length
            a1 = a0 + arc_length
            targets = np.linspace(a0, a1, n_resample)
            px = np.interp(targets, cum, traj[:, 0])
            py = np.interp(targets, cum, traj[:, 1])
            pts = np.stack([px, py], axis=1)  # (16, 2)

            origin = pts[0].copy()
            pts_c = pts - origin
            end_vec = pts_c[-1]
            norm = np.linalg.norm(end_vec)
            if norm < 1e-3:
                theta = 0.0
                n_degenerate_rotation += 1
            else:
                theta = np.arctan2(end_vec[1], end_vec[0])
            c, s = np.cos(-theta), np.sin(-theta)
            R = np.array([[c, -s], [s, c]])
            pts_r = pts_c @ R.T
            segs.append(pts_r.reshape(-1))

    segs = np.stack(segs).astype(np.float32) if segs else np.zeros((0, n_resample * 2), dtype=np.float32)
    return dict(
        segs=segs,
        n_episodes=n_ep,
        ep_len=ep_len,
        grid_ok=bool(grid_ok),
        n_degenerate_rotation=int(n_degenerate_rotation),
        n_segments_per_episode_mean=float(np.mean(n_segments_per_ep)) if n_segments_per_ep else 0.0,
        n_segments_per_episode_min=int(np.min(n_segments_per_ep)) if n_segments_per_ep else 0,
        n_segments_per_episode_max=int(np.max(n_segments_per_ep)) if n_segments_per_ep else 0,
    )


def load_route_data(data_path, split_seed=42, val_frac=0.1, arc_length=ARC_LENGTH, n_resample=N_RESAMPLE):
    d = np.load(data_path)
    qpos = np.asarray(d["qpos"], dtype=np.float64)
    terminals = np.asarray(d["terminals"], dtype=bool)
    qpos_xy = qpos[:, :2]

    ext = extract_route_segments(qpos_xy, terminals, arc_length=arc_length, n_resample=n_resample)
    segs = ext["segs"]
    n = len(segs)

    rng = np.random.default_rng(split_seed)
    perm = rng.permutation(n)
    n_val = int(n * val_frac)
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    return dict(
        segs=segs,
        train_idx=train_idx,
        val_idx=val_idx,
        seg_dim=segs.shape[1],
        n_resample=n_resample,
        arc_length=arc_length,
        extraction=ext,
    )


# ------------------------------------------------------------------
# 模型：單一 codebook VQ-VAE（無條件 decoder，路線是幾何不需狀態條件）
# ------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class VQEMA(nn.Module):
    def __init__(self, K, D, decay=0.99, eps=1e-5, dead_steps=500):
        super().__init__()
        self.K = K
        self.D = D
        self.decay = decay
        self.eps = eps
        self.dead_steps = dead_steps
        embed = torch.randn(K, D) * 0.1
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(K))
        self.register_buffer("embed_avg", embed.clone())
        self.register_buffer("unused_steps", torch.zeros(K, dtype=torch.long))
        self.register_buffer("restarts_total", torch.zeros(1, dtype=torch.long))

    def forward(self, z_e, training):
        dist = (
            z_e.pow(2).sum(1, keepdim=True)
            - 2 * z_e @ self.embed.t()
            + self.embed.pow(2).sum(1)
        )
        idx = dist.argmin(1)
        onehot = F.one_hot(idx, self.K).type(z_e.dtype)
        z_q = self.embed[idx]

        if training:
            with torch.no_grad():
                counts = onehot.sum(0)
                self.cluster_size.mul_(self.decay).add_(counts, alpha=1 - self.decay)
                embed_sum = onehot.t() @ z_e
                self.embed_avg.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
                n = self.cluster_size.sum()
                cluster_size = (
                    (self.cluster_size + self.eps) / (n + self.K * self.eps) * n
                )
                self.embed.copy_(self.embed_avg / cluster_size.unsqueeze(1))

                used = counts > 0
                self.unused_steps[used] = 0
                self.unused_steps[~used] += 1
                dead = self.unused_steps >= self.dead_steps
                n_dead = int(dead.sum().item())
                if n_dead > 0:
                    pool = z_e.detach()
                    pick_i = torch.randint(0, pool.shape[0], (n_dead,))
                    pick = pool[pick_i]
                    self.embed[dead] = pick
                    self.embed_avg[dead] = pick
                    self.cluster_size[dead] = 1.0
                    self.unused_steps[dead] = 0
                    self.restarts_total += n_dead

        z_q_st = z_e + (z_q - z_e).detach()
        commitment_loss = F.mse_loss(z_e, z_q.detach())
        return z_q_st, idx, commitment_loss


class RouteVQVAE(nn.Module):
    def __init__(self, seg_dim, latent_dim, K, hidden=512, decay=0.99, dead_steps=500):
        super().__init__()
        self.encoder = MLP(seg_dim, latent_dim, hidden)
        self.decoder = MLP(latent_dim, seg_dim, hidden)
        self.vq = VQEMA(K, latent_dim, decay=decay, dead_steps=dead_steps)

    def forward(self, x, training):
        z_e = self.encoder(x)
        z_q_st, idx, commit_loss = self.vq(z_e, training=training)
        recon = self.decoder(z_q_st)
        return recon, idx, commit_loss

    def decode_code(self, code_idx_1d):
        z_q = self.vq.embed[code_idx_1d]
        return self.decoder(z_q)


# ------------------------------------------------------------------
# 訓練與量測（同 A 的 plateau 判準與門檻）
# ------------------------------------------------------------------
PLATEAU_WINDOW = 5
PLATEAU_REL_THRESH = 0.03
EXTEND_TO_STEPS = 50000


def train_and_eval(
    data, K, latent_dim=16, steps=20000, batch=512, lr=3e-4, beta=0.25,
    hidden=512, decay=0.99, dead_steps=500, seed=0, collapse_top1_thresh=0.5,
):
    import time

    torch.manual_seed(seed)
    device = torch.device("cpu")

    segs = data["segs"]
    train_idx = data["train_idx"]
    val_idx = data["val_idx"]
    seg_dim = segs.shape[1]

    model = RouteVQVAE(seg_dim, latent_dim, K, hidden=hidden, decay=decay, dead_steps=dead_steps).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    train_t = torch.from_numpy(segs[train_idx]).to(device)
    val_t = torch.from_numpy(segs[val_idx]).to(device)
    n_train = len(train_idx)

    batch_rng = np.random.default_rng(seed + 1)
    loss_curve = []
    t0 = time.time()

    def run_steps(n_steps, start_step):
        final_commit = None
        for i in range(n_steps):
            step = start_step + i
            bidx = batch_rng.integers(0, n_train, size=batch)
            x = train_t[bidx]
            recon, _idx, commit_loss = model(x, training=True)
            recon_loss = F.mse_loss(recon, x)
            loss = recon_loss + beta * commit_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            if step % 1000 == 0 or i == n_steps - 1:
                loss_curve.append([step, float(recon_loss.item()), float(commit_loss.item())])
            final_commit = float(commit_loss.item())
        return final_commit

    final_commit = run_steps(steps, 0)

    extended = False
    if len(loss_curve) >= PLATEAU_WINDOW:
        window = [p[1] for p in loss_curve[-PLATEAU_WINDOW:]]
        rel_change = (window[0] - window[-1]) / max(window[0], 1e-8)
    else:
        rel_change = None
    plateau_info = dict(
        rel_change_at_check=rel_change, threshold=PLATEAU_REL_THRESH, window=PLATEAU_WINDOW,
        flat=(rel_change is not None and rel_change <= PLATEAU_REL_THRESH),
    )
    if rel_change is not None and rel_change > PLATEAU_REL_THRESH and steps < EXTEND_TO_STEPS:
        extra = EXTEND_TO_STEPS - steps
        final_commit = run_steps(extra, steps)
        extended = True
        window2 = [p[1] for p in loss_curve[-PLATEAU_WINDOW:]]
        rel_change2 = (window2[0] - window2[-1]) / max(window2[0], 1e-8)
        plateau_info["rel_change_after_extend"] = rel_change2
        plateau_info["extended_to_steps"] = EXTEND_TO_STEPS
    plateau_info["extended"] = extended
    total_steps_used = EXTEND_TO_STEPS if extended else steps

    train_seconds = time.time() - t0

    model.eval()
    with torch.no_grad():
        train_recon, train_codes, _ = model(train_t, training=False)
        recon_mse_train = float(F.mse_loss(train_recon, train_t).item())
        val_recon, val_codes, _ = model(val_t, training=False)
        recon_mse_val = float(F.mse_loss(val_recon, val_t).item())
        all_codes = torch.cat([train_codes, val_codes]).numpy()

    baseline_vec = segs[train_idx].mean(0, keepdims=True)
    baseline_mse_train = float(np.mean((segs[train_idx] - baseline_vec) ** 2))
    baseline_mse_val = float(np.mean((segs[val_idx] - baseline_vec) ** 2))

    counts = np.bincount(all_codes, minlength=K).astype(np.int64)
    total = counts.sum()
    frac = counts / max(total, 1)
    nz = frac[frac > 0]
    perplexity = float(np.exp(-np.sum(nz * np.log(nz)))) if len(nz) > 0 else 0.0
    order = np.argsort(-frac)
    top1 = float(frac[order[0]]) if K >= 1 else 0.0
    top3 = float(frac[order[: min(3, K)]].sum())
    n_active = int((counts > 0).sum())
    collapsed = (n_active < K) or (top1 > collapse_top1_thresh)

    results = dict(
        metrics=dict(
            recon_mse_train=recon_mse_train,
            recon_mse_val=recon_mse_val,
            baseline_mse_train=baseline_mse_train,
            baseline_mse_val=baseline_mse_val,
            val_improvement_over_baseline_pct=100.0 * (1.0 - recon_mse_val / baseline_mse_val),
            perplexity=perplexity,
            perplexity_max=int(K),
            n_active_codes=n_active,
            usage_counts=counts.tolist(),
            usage_frac=frac.tolist(),
            top1_usage_frac=top1,
            top3_usage_frac=top3,
            commit_loss_final=final_commit,
            restarts_total=int(model.vq.restarts_total.item()),
            collapsed=bool(collapsed),
            collapse_top1_thresh=collapse_top1_thresh,
            total_steps_used=total_steps_used,
        ),
        plateau=plateau_info,
        loss_curve=loss_curve,
        timing=dict(train_seconds=train_seconds),
    )
    return model, results, all_codes


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


# ------------------------------------------------------------------
# Pillow：K 個路線字疊圖 + 3 個字的真實段疊圖
# ------------------------------------------------------------------
def _load_font():
    from PIL import ImageFont

    return ImageFont.load_default()


def _fit_scale(all_pts_list, W, H, margin=60):
    xs = np.concatenate([p[:, 0] for p in all_pts_list])
    ys = np.concatenate([p[:, 1] for p in all_pts_list])
    max_ext = max(np.abs(xs).max(), np.abs(ys).max(), 1e-6)
    scale = (min(W, H) / 2 - margin) / max_ext
    return scale


def draw_route_words(word_polylines, out_path, title):
    """word_polylines: list of (16,2) arrays（每個字 decode 出的原型形狀）。全部畫在同一張圖、共原點。"""
    from PIL import Image, ImageDraw

    K = len(word_polylines)
    W, H = 760, 760
    img = Image.new("RGB", (W, H), hex_to_rgb(SURFACE))
    draw = ImageDraw.Draw(img)
    font = _load_font()
    cx, cy = W / 2, H / 2

    scale = _fit_scale(word_polylines, W, H)
    colors = distinct_colors(K)

    # axes
    draw.line([(cx, 40), (cx, H - 40)], fill=hex_to_rgb(GRIDLINE))
    draw.line([(40, cy), (W - 40, cy)], fill=hex_to_rgb(GRIDLINE))
    draw.ellipse([cx - 4, cy - 4, cx + 4, cy + 4], fill=hex_to_rgb(INK_PRIMARY))

    for k, pts in enumerate(word_polylines):
        px = cx + pts[:, 0] * scale
        py = cy - pts[:, 1] * scale
        coords = list(zip(px.tolist(), py.tolist()))
        draw.line(coords, fill=colors[k], width=3)
        draw.ellipse([coords[-1][0] - 3, coords[-1][1] - 3, coords[-1][0] + 3, coords[-1][1] + 3], fill=colors[k])
        draw.text((coords[-1][0] + 5, coords[-1][1] - 5), str(k), fill=colors[k], font=font)

    draw.text((10, 10), title, fill=hex_to_rgb(INK_PRIMARY), font=font)
    draw.text((10, H - 20), "origin = segment start, +x = segment net direction", fill=hex_to_rgb(INK_MUTED), font=font)
    img.save(out_path)


def draw_code_examples(code_ids, examples_per_code, decoded_per_code, out_path, title):
    """code_ids: list of 3 codes；examples_per_code: dict code -> list of (16,2) 真實段；
    decoded_per_code: dict code -> (16,2) 該字的 decode 原型（灰色參考線）。"""
    from PIL import Image, ImageDraw

    n = len(code_ids)
    panel_w, panel_h = 340, 380
    W, H = panel_w * n + 20 * (n + 1), panel_h + 60
    img = Image.new("RGB", (W, H), hex_to_rgb(SURFACE))
    draw = ImageDraw.Draw(img)
    font = _load_font()

    all_pts = []
    for c in code_ids:
        all_pts.extend(examples_per_code[c])
        all_pts.append(decoded_per_code[c])
    scale = _fit_scale(all_pts, panel_w, panel_h - 40, margin=40)

    ex_color = hex_to_rgb("#2a78d6")
    proto_color = hex_to_rgb("#e34948")

    for i, c in enumerate(code_ids):
        x0 = 20 + i * (panel_w + 20)
        cx, cy = x0 + panel_w / 2, 40 + (panel_h - 40) / 2
        draw.rectangle([x0, 30, x0 + panel_w, 30 + panel_h - 20], outline=hex_to_rgb(AXIS))
        draw.line([(cx, 30), (cx, 30 + panel_h - 20)], fill=hex_to_rgb(GRIDLINE))
        for pts in examples_per_code[c]:
            px = cx + pts[:, 0] * scale
            py = cy - pts[:, 1] * scale
            draw.line(list(zip(px.tolist(), py.tolist())), fill=ex_color, width=2)
        proto = decoded_per_code[c]
        px = cx + proto[:, 0] * scale
        py = cy - proto[:, 1] * scale
        draw.line(list(zip(px.tolist(), py.tolist())), fill=proto_color, width=3)
        draw.text((x0 + 4, 10), f"code {c} (n_examples={len(examples_per_code[c])})", fill=hex_to_rgb(INK_PRIMARY), font=font)

    draw.text((10, H - 20), title + "  [blue=real segments assigned to code, red=decoded prototype]", fill=hex_to_rgb(INK_MUTED), font=font)
    img.save(out_path)
