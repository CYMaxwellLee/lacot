"""M0 偵察共用模組：B2 步法字典（見 docs/DESIGN-2026-09-08-b2-gait-dictionary.md §三 M0）。

職責：
  1. 從 OGBench antmaze-medium-stitch-v0 訓練集切「動作段」（連續 seg_len 步的 action，
     同一軌跡內、不跨 episode 邊界）。
  2. 小 MLP encoder/decoder + VQ-EMA codebook（EMA 衰減、dead-code restart）。
  3. 訓練與量測（重建 MSE、baseline MSE、code 使用率、perplexity）。
  4. 用 Pillow 手畫長條圖 / 曲線圖（這台 venv 沒裝 matplotlib，不裝套件，改用 Pillow）。
  5. MuJoCo（osmesa 軟體 render）把動作段回放成短 gif，供人眼判讀步法。

⛔ 不改動任何既有檔案；本檔只被同目錄的 run_config.py import。
"""
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------------------------------------------
# 調色盤：來自 dataviz skill references/palette.md（已驗證 CVD 安全的預設盤）。
# 這台沒裝 matplotlib，圖表用 Pillow 手畫，顏色沿用同一份驗證過的盤，不即興配色。
# ------------------------------------------------------------------
PALETTE_CATEGORICAL = [
    "#2a78d6",  # 1 blue
    "#eb6834",  # 2 orange
    "#1baf7a",  # 3 aqua
    "#eda100",  # 4 yellow
    "#e87ba4",  # 5 magenta
    "#008300",  # 6 green
    "#4a3aa7",  # 7 violet
    "#e34948",  # 8 red
]
SEQUENTIAL_BLUE = "#2a78d6"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"

ACT_DIM = 8
EP_LEN = 201  # antmaze-medium-stitch-v0：5000 episodes × 201 steps（已實測驗證）


def hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))


# ------------------------------------------------------------------
# 資料：切動作段
# ------------------------------------------------------------------
def load_segments(data_path, seg_len, split_seed=42, val_frac=0.1):
    """從 antmaze-medium-stitch-v0.npz 切連續 seg_len 步的 action 段（不跨 episode）。

    回傳 dict：segs (float32 [N, seg_len*act_dim])、seg_starts（每段第一步在原始
    陣列裡的 global index，供之後 qpos/qvel 回放用）、train_idx / val_idx（9:1 切分）。
    """
    d = np.load(data_path)
    actions = np.asarray(d["actions"], dtype=np.float32)
    terminals = np.asarray(d["terminals"], dtype=bool)
    act_dim = actions.shape[1]
    assert act_dim == ACT_DIM, f"預期 act_dim={ACT_DIM}，實際 {act_dim}"

    ends = np.flatnonzero(terminals)
    starts = np.concatenate([[0], ends[:-1] + 1])
    n_ep = len(starts)
    ep_len = int(ends[0] - starts[0] + 1)

    # 快路徑假設：episode 是固定長度、連續排列的格子（已用真資料驗證成立）。
    # 驗不過就走通用 fallback（不假設格子），寧可慢也不要切錯。
    grid_ok = ep_len == EP_LEN and np.array_equal(starts, np.arange(0, n_ep * ep_len, ep_len)) and (
        n_ep * ep_len == len(terminals)
    )

    if grid_ok:
        n_chunks = ep_len // seg_len
        usable = n_chunks * seg_len
        act_grid = actions[: n_ep * ep_len].reshape(n_ep, ep_len, act_dim)[:, :usable]
        act_grid = act_grid.reshape(n_ep, n_chunks, seg_len, act_dim)
        segs = act_grid.reshape(n_ep * n_chunks, seg_len * act_dim).astype(np.float32)
        seg_starts = (
            np.arange(n_ep)[:, None] * ep_len + np.arange(n_chunks)[None, :] * seg_len
        ).reshape(-1).astype(np.int64)
        fallback_used = False
    else:
        # 通用路徑：逐 episode 切，不假設固定格子。
        segs_list, starts_list = [], []
        for s0, e0 in zip(starts, ends):
            T = int(e0 - s0 + 1)
            n_chunks = T // seg_len
            for c in range(n_chunks):
                idx0 = int(s0 + c * seg_len)
                segs_list.append(actions[idx0 : idx0 + seg_len].reshape(-1))
                starts_list.append(idx0)
        segs = np.stack(segs_list).astype(np.float32)
        seg_starts = np.array(starts_list, dtype=np.int64)
        fallback_used = True

    rng = np.random.default_rng(split_seed)
    perm = rng.permutation(len(seg_starts))
    n_val = int(len(perm) * val_frac)
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    return dict(
        segs=segs,
        seg_starts=seg_starts,
        train_idx=train_idx,
        val_idx=val_idx,
        act_dim=act_dim,
        ep_len=ep_len,
        n_episodes=n_ep,
        fallback_segmentation_used=fallback_used,
    )


# ------------------------------------------------------------------
# 模型：小 MLP encoder/decoder + VQ-EMA codebook
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
    """VQ-VAE codebook，EMA 更新（Zhipeng et al. / van den Oord EMA 版）＋ dead-code restart。"""

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
        # z_e: (B, D)
        dist = (
            z_e.pow(2).sum(1, keepdim=True)
            - 2 * z_e @ self.embed.t()
            + self.embed.pow(2).sum(1)
        )
        idx = dist.argmin(1)  # (B,)
        onehot = F.one_hot(idx, self.K).type(z_e.dtype)  # (B, K)
        z_q = self.embed[idx]

        if training:
            with torch.no_grad():
                counts = onehot.sum(0)  # (K,)
                self.cluster_size.mul_(self.decay).add_(counts, alpha=1 - self.decay)
                embed_sum = onehot.t() @ z_e  # (K, D)
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


class VQVAE(nn.Module):
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


# ------------------------------------------------------------------
# 訓練與量測
# ------------------------------------------------------------------
def train_and_eval(
    data,
    K,
    steps=20000,
    batch=512,
    lr=3e-4,
    beta=0.25,
    latent_dim=16,
    hidden=512,
    decay=0.99,
    dead_steps=500,
    seed=0,
):
    torch.manual_seed(seed)
    device = torch.device("cpu")

    segs = data["segs"]
    train_idx = data["train_idx"]
    val_idx = data["val_idx"]
    seg_dim = segs.shape[1]

    model = VQVAE(seg_dim, latent_dim, K, hidden=hidden, decay=decay, dead_steps=dead_steps).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    train_t = torch.from_numpy(segs[train_idx]).to(device)
    val_t = torch.from_numpy(segs[val_idx]).to(device)
    n_train = len(train_idx)

    batch_rng = np.random.default_rng(seed + 1)
    loss_curve = []
    t0 = time.time()
    final_commit = None
    for step in range(steps):
        bidx = batch_rng.integers(0, n_train, size=batch)
        x = train_t[bidx]
        recon, _idx, commit_loss = model(x, training=True)
        recon_loss = F.mse_loss(recon, x)
        loss = recon_loss + beta * commit_loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 1000 == 0 or step == steps - 1:
            loss_curve.append([step, float(recon_loss.item()), float(commit_loss.item())])
        final_commit = float(commit_loss.item())
    train_seconds = time.time() - t0

    model.eval()
    with torch.no_grad():
        train_recon, train_codes, _ = model(train_t, training=False)
        recon_mse_train = float(F.mse_loss(train_recon, train_t).item())
        val_recon, val_codes, _ = model(val_t, training=False)
        recon_mse_val = float(F.mse_loss(val_recon, val_t).item())

        all_codes = torch.cat([train_codes, val_codes]).numpy()

    # baseline：猜「資料集平均動作段」（用 train 段算平均，train/val 各自量 MSE）
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

    results = dict(
        metrics=dict(
            recon_mse_train=recon_mse_train,
            recon_mse_val=recon_mse_val,
            baseline_mse_train=baseline_mse_train,
            baseline_mse_val=baseline_mse_val,
            perplexity=perplexity,
            perplexity_max=int(K),
            n_active_codes=n_active,
            usage_counts=counts.tolist(),
            usage_frac=frac.tolist(),
            top1_usage_frac=top1,
            top3_usage_frac=top3,
            commit_loss_final=final_commit,
            restarts_total=int(model.vq.restarts_total.item()),
        ),
        loss_curve=loss_curve,
        timing=dict(train_seconds=train_seconds),
    )
    return model, results, all_codes


# ------------------------------------------------------------------
# Pillow 手畫圖（這台沒裝 matplotlib）
# ------------------------------------------------------------------
def _load_font():
    from PIL import ImageFont

    return ImageFont.load_default()


def draw_usage_histogram(usage_frac, out_path, title):
    from PIL import Image, ImageDraw

    K = len(usage_frac)
    W, H = max(560, 40 * K + 160), 420
    margin_l, margin_r, margin_t, margin_b = 70, 30, 50, 60
    plot_w = W - margin_l - margin_r
    plot_h = H - margin_t - margin_b

    img = Image.new("RGB", (W, H), hex_to_rgb(SURFACE))
    draw = ImageDraw.Draw(img)
    font = _load_font()

    maxv = max(usage_frac) if len(usage_frac) else 1.0
    maxv = maxv if maxv > 0 else 1.0
    # 一點頭部空間放數值標籤
    top = maxv * 1.18

    # gridlines (0%, 25%, 50%, 75%, 100% of `top`)
    for frac_g in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = margin_t + plot_h - plot_h * frac_g
        draw.line([(margin_l, y), (margin_l + plot_w, y)], fill=hex_to_rgb(GRIDLINE), width=1)
        draw.text((8, y - 6), f"{frac_g*top*100:.1f}%", fill=hex_to_rgb(INK_MUTED), font=font)

    bar_w = plot_w / K
    bar_rgb = hex_to_rgb(SEQUENTIAL_BLUE)
    for i, v in enumerate(usage_frac):
        x0 = margin_l + i * bar_w + bar_w * 0.15
        x1 = margin_l + (i + 1) * bar_w - bar_w * 0.15
        h = plot_h * (v / top)
        y1 = margin_t + plot_h
        y0 = y1 - h
        draw.rectangle([x0, y0, x1, y1], fill=bar_rgb, outline=hex_to_rgb(INK_PRIMARY))
        if K <= 34:
            draw.text(((x0 + x1) / 2 - 8, y1 + 4), str(i), fill=hex_to_rgb(INK_SECONDARY), font=font)
        if v > 0 and K <= 20:
            draw.text(((x0 + x1) / 2 - 12, y0 - 12), f"{v*100:.1f}%", fill=hex_to_rgb(INK_PRIMARY), font=font)

    draw.line([(margin_l, margin_t), (margin_l, margin_t + plot_h)], fill=hex_to_rgb(AXIS), width=2)
    draw.line([(margin_l, margin_t + plot_h), (margin_l + plot_w, margin_t + plot_h)], fill=hex_to_rgb(AXIS), width=2)
    draw.text((margin_l, 14), title, fill=hex_to_rgb(INK_PRIMARY), font=font)
    draw.text((margin_l, H - 18), "code index", fill=hex_to_rgb(INK_MUTED), font=font)
    img.save(out_path)


def draw_code_fallback_plot(examples, out_path, code_idx, seg_len, act_dim):
    """MuJoCo render 失敗時的退路：xy 位移向量 ＋ 8 維 action 曲線疊圖。

    examples: list of dict(disp_xy=(dx,dy), actions=np.array[seg_len,act_dim])
    """
    from PIL import Image, ImageDraw

    W, H = 900, 420
    img = Image.new("RGB", (W, H), hex_to_rgb(SURFACE))
    draw = ImageDraw.Draw(img)
    font = _load_font()
    ex_colors = [hex_to_rgb(c) for c in PALETTE_CATEGORICAL[:3]]

    # --- 左半：xy 位移向量 ---
    lx0, ly0, lx1, ly1 = 20, 60, 420, 400
    cx, cy = (lx0 + lx1) / 2, (ly0 + ly1) / 2
    draw.rectangle([lx0, ly0, lx1, ly1], outline=hex_to_rgb(AXIS))
    draw.line([(cx, ly0), (cx, ly1)], fill=hex_to_rgb(GRIDLINE))
    draw.line([(lx0, cy), (lx1, cy)], fill=hex_to_rgb(GRIDLINE))
    maxd = max(0.05, max((abs(e["disp_xy"][0]) for e in examples), default=0.05),
               max((abs(e["disp_xy"][1]) for e in examples), default=0.05))
    scale = (min(lx1 - lx0, ly1 - ly0) / 2 - 20) / (maxd * 1.2)
    for i, e in enumerate(examples):
        dx, dy = e["disp_xy"]
        ex, ey = cx + dx * scale, cy - dy * scale
        draw.line([(cx, cy), (ex, ey)], fill=ex_colors[i % 3], width=3)
        draw.ellipse([ex - 4, ey - 4, ex + 4, ey + 4], fill=ex_colors[i % 3])
        draw.text((lx0, ly1 + 6 + 14 * i), f"ex{i}: dx={dx:+.3f} dy={dy:+.3f}", fill=ex_colors[i % 3], font=font)
    draw.text((lx0, ly0 - 20), f"code {code_idx}: xy displacement (arrow = start->end)", fill=hex_to_rgb(INK_PRIMARY), font=font)

    # --- 右半：8 維 action 曲線疊圖（每個 example 一欄小圖）---
    rx0 = 460
    col_w = (W - rx0 - 20) / max(1, len(examples))
    dim_colors = [hex_to_rgb(c) for c in PALETTE_CATEGORICAL]
    for i, e in enumerate(examples):
        px0 = rx0 + i * col_w + 10
        px1 = rx0 + (i + 1) * col_w - 10
        py0, py1 = 60, 340
        draw.rectangle([px0, py0, px1, py1], outline=hex_to_rgb(AXIS))
        mid_y = (py0 + py1) / 2
        draw.line([(px0, mid_y), (px1, mid_y)], fill=hex_to_rgb(GRIDLINE))
        acts = e["actions"]  # (seg_len, act_dim), range ~[-1,1]
        T = acts.shape[0]
        for d in range(act_dim):
            pts = []
            for t in range(T):
                x = px0 + (px1 - px0) * (t / max(1, T - 1))
                v = float(np.clip(acts[t, d], -1, 1))
                y = mid_y - v * (py1 - py0) / 2 * 0.95
                pts.append((x, y))
            draw.line(pts, fill=dim_colors[d], width=2)
        draw.text((px0, py1 + 4), f"ex{i} action curves (8 dims)", fill=hex_to_rgb(INK_SECONDARY), font=font)

    # legend for action dims
    leg_y = 400
    for d in range(act_dim):
        lx = rx0 + d * 55
        draw.line([(lx, leg_y), (lx + 14, leg_y)], fill=dim_colors[d], width=3)
        draw.text((lx + 16, leg_y - 6), f"a{d}", fill=hex_to_rgb(INK_SECONDARY), font=font)

    img.save(out_path)


# ------------------------------------------------------------------
# MuJoCo render（osmesa 軟體算圖，不碰 GPU）
# ------------------------------------------------------------------
def render_segment_gif(env_state, seg_start, seg_len, out_path, qpos, qvel):
    """把一段 (seg_start .. seg_start+seg_len) 的 qpos/qvel 用 MuJoCo 回放並存成 gif。

    env_state: dict(u=env.unwrapped, renderer=mujoco.Renderer)
    """
    import mujoco
    import imageio.v2 as imageio

    u = env_state["u"]
    renderer = env_state["renderer"]

    xy0 = qpos[seg_start][:2]
    xy1 = qpos[seg_start + seg_len][:2]
    mid = (xy0 + xy1) / 2.0

    cam = mujoco.MjvCamera()
    cam.lookat[0] = float(mid[0])
    cam.lookat[1] = float(mid[1])
    cam.lookat[2] = 0.3
    cam.distance = 3.0
    cam.elevation = -35
    cam.azimuth = 135

    frames = []
    for t in range(seg_start, seg_start + seg_len + 1):
        u.set_state(qpos[t].copy(), qvel[t].copy())
        renderer.update_scene(u.data, camera=cam)
        frame = renderer.render()
        frames.append(frame.copy())
    imageio.mimsave(out_path, frames, duration=0.2, loop=0)
    return len(frames)


def make_render_env(dataset_dir):
    """建立 antmaze-medium-stitch-v0 env 供 render 用（env_only，不重複載入 dataset）。
    呼叫前必須已設 MUJOCO_GL=osmesa（在 process 環境變數，import mujoco 前）。
    """
    import mujoco
    import ogbench

    env = ogbench.make_env_and_datasets(
        "antmaze-medium-stitch-v0", dataset_dir=dataset_dir, env_only=True
    )
    u = env.unwrapped
    env.reset()
    # 觸發 lazy renderer 初始化
    u.set_state(u.data.qpos.copy(), u.data.qvel.copy())
    _ = u.render()
    renderer = u.custom_renderer
    return dict(env=env, u=u, renderer=renderer)


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
