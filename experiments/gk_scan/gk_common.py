"""G×K 因子化 VQ 掃描共用模組（見 docs/PLAN-2026-09-08-walk-first.md 階段乙 P2.5）。

任務 A 的獨立實作，⛔ 不 import 同倉庫其他實驗目錄（m0_gait_dict/ 註明「只被同目錄
run_config.py import」；本檔自成一份，允許重複少量常數/樣式以保持自我完整）。

職責：
  1. 從 antmaze-medium-stitch-v0 切「動作段」（固定 seg_len=4，不跨 episode），
     同時取每段起點的完整 29 維 obs（供 decoder 條件化用）。
  2. Dreamer 式因子化 VQ-VAE：encoder 輸出切 G 份、各自過自己的 VQ-EMA codebook
     （K、D=8），拼接後＋段起點 obs（normalize）→ decoder → 4 步動作段。
  3. 訓練（20k steps，loss 未走平延到 50k，判準見 train_and_eval 內 PLATEAU_* 常數）
     與量測（重建 MSE、baseline、每組 perplexity/存活數、因子有效性）。
  4. Pillow 手畫膝點曲線（這台沒裝 matplotlib，不裝套件）。
  5. MuJoCo osmesa rollout + 並排 gif（給推薦格的 A/B 對比視覺化）。

⛔ CPU-only：本檔與呼叫它的 run_cell.py 一律不碰 GPU。
"""
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------------------------------------------
# 常數（沿用 m0_gait_dict 已驗證過的資料集事實與 dataviz 色盤，數值一致但本檔獨立持有）
# ------------------------------------------------------------------
ACT_DIM = 8
OBS_DIM = 29
EP_LEN = 201  # antmaze-medium-stitch-v0：5000 episodes × 201 steps（M0 已驗證）
DATASET_NAME = "antmaze-medium-stitch-v0"
GROUP_DIM = 8  # 每組 embedding 維度：任務指定的預設假定值，非本檔發明

PALETTE_CATEGORICAL = [
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100",
    "#e87ba4", "#008300", "#4a3aa7", "#e34948",
]
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"


def hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))


# ------------------------------------------------------------------
# 資料：切動作段 + 段起點 obs
# ------------------------------------------------------------------
def load_segments_with_obs(data_path, seg_len=4, split_seed=42, val_frac=0.1):
    d = np.load(data_path)
    actions = np.asarray(d["actions"], dtype=np.float32)
    observations = np.asarray(d["observations"], dtype=np.float32)
    terminals = np.asarray(d["terminals"], dtype=bool)
    act_dim = actions.shape[1]
    obs_dim = observations.shape[1]
    assert act_dim == ACT_DIM, f"預期 act_dim={ACT_DIM}，實際 {act_dim}"
    assert obs_dim == OBS_DIM, f"預期 obs_dim={OBS_DIM}，實際 {obs_dim}"

    ends = np.flatnonzero(terminals)
    starts = np.concatenate([[0], ends[:-1] + 1])
    n_ep = len(starts)
    ep_len = int(ends[0] - starts[0] + 1)

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

    obs_start = observations[seg_starts].astype(np.float32)  # (N, obs_dim)

    rng = np.random.default_rng(split_seed)
    perm = rng.permutation(len(seg_starts))
    n_val = int(len(perm) * val_frac)
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    return dict(
        segs=segs,
        obs_start=obs_start,
        seg_starts=seg_starts,
        train_idx=train_idx,
        val_idx=val_idx,
        act_dim=act_dim,
        obs_dim=obs_dim,
        ep_len=ep_len,
        n_episodes=n_ep,
        fallback_segmentation_used=fallback_used,
    )


def compute_obs_norm(obs_start, train_idx, eps=1e-3):
    """obs normalize 統計量只用 train 段算（避免用到 val 資訊），套用到 train/val 皆同。"""
    mu = obs_start[train_idx].mean(0, keepdims=True)
    sd = obs_start[train_idx].std(0, keepdims=True)
    sd = np.maximum(sd, eps)  # 避免除以接近 0 的標準差（部分 obs 維度變異極小）
    return mu.astype(np.float32), sd.astype(np.float32)


# ------------------------------------------------------------------
# 模型：MLP + 每組獨立 VQ-EMA + 條件化 decoder
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
    """單一 codebook，VQ-EMA + dead-code restart（沿用 M0 驗證過的實作邏輯）。"""

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


class MultiVQEMA(nn.Module):
    """G 個獨立 VQEMA codebook，各管 z_e 的一段（每段 D 維）。"""

    def __init__(self, G, K, D, decay=0.99, dead_steps=500):
        super().__init__()
        self.G = G
        self.K = K
        self.D = D
        self.groups = nn.ModuleList([VQEMA(K, D, decay=decay, dead_steps=dead_steps) for _ in range(G)])

    def forward(self, z_e_full, training):
        # z_e_full: (B, G*D)
        chunks = z_e_full.split(self.D, dim=1)
        z_q_chunks, idx_chunks, commit_losses = [], [], []
        for g, vq in enumerate(self.groups):
            z_q_st, idx, commit_loss = vq(chunks[g], training=training)
            z_q_chunks.append(z_q_st)
            idx_chunks.append(idx)
            commit_losses.append(commit_loss)
        z_q_full = torch.cat(z_q_chunks, dim=1)
        idx_full = torch.stack(idx_chunks, dim=1)  # (B, G)
        commit_loss_mean = torch.stack(commit_losses).mean()  # 對 G 取平均，見 §設計假定
        return z_q_full, idx_full, commit_loss_mean


class FactorizedCondVQVAE(nn.Module):
    """encoder(action_seg) -> G*D 連續 -> 每組各自量化 -> decoder(concat(z_q), obs_norm) -> action_seg。"""

    def __init__(self, seg_dim, obs_dim, G, K, D=GROUP_DIM, hidden=512, decay=0.99, dead_steps=500):
        super().__init__()
        self.G, self.K, self.D = G, K, D
        self.encoder = MLP(seg_dim, G * D, hidden)
        self.decoder = MLP(G * D + obs_dim, seg_dim, hidden)
        self.vq = MultiVQEMA(G, K, D, decay=decay, dead_steps=dead_steps)

    def forward(self, x, obs_norm, training):
        z_e = self.encoder(x)
        z_q, idx, commit_loss = self.vq(z_e, training=training)
        recon = self.decoder(torch.cat([z_q, obs_norm], dim=1))
        return recon, idx, commit_loss

    def decode_codes(self, code_idx, obs_norm):
        """code_idx: (B, G) long -> 用各組 codebook 向量組回 z_q -> decoder。"""
        z_q_chunks = [self.vq.groups[g].embed[code_idx[:, g]] for g in range(self.G)]
        z_q = torch.cat(z_q_chunks, dim=1)
        return self.decoder(torch.cat([z_q, obs_norm], dim=1))


# ------------------------------------------------------------------
# 訓練與量測
# ------------------------------------------------------------------
PLATEAU_WINDOW = 5          # 看最後 5 個紀錄點（每 1000 步一筆）
PLATEAU_REL_THRESH = 0.03   # 這段窗口內相對下降 >3% 判定「還沒走平」
EXTEND_TO_STEPS = 50000     # 沒走平就練到這裡


def _usage_stats(codes_1d, K):
    counts = np.bincount(codes_1d, minlength=K).astype(np.int64)
    total = counts.sum()
    frac = counts / max(total, 1)
    nz = frac[frac > 0]
    perplexity = float(np.exp(-np.sum(nz * np.log(nz)))) if len(nz) > 0 else 0.0
    order = np.argsort(-frac)
    top1 = float(frac[order[0]]) if K >= 1 else 0.0
    top3 = float(frac[order[: min(3, K)]].sum())
    n_active = int((counts > 0).sum())
    return dict(
        usage_counts=counts.tolist(),
        usage_frac=frac.tolist(),
        perplexity=perplexity,
        n_active_codes=n_active,
        top1_usage_frac=top1,
        top3_usage_frac=top3,
    )


def train_and_eval(
    data,
    G,
    K,
    D=GROUP_DIM,
    steps=20000,
    batch=512,
    lr=3e-4,
    beta=0.25,
    hidden=512,
    decay=0.99,
    dead_steps=500,
    seed=0,
    collapse_top1_thresh=0.5,
):
    torch.manual_seed(seed)
    device = torch.device("cpu")

    segs = data["segs"]
    obs_start = data["obs_start"]
    train_idx = data["train_idx"]
    val_idx = data["val_idx"]
    seg_dim = segs.shape[1]
    obs_dim = obs_start.shape[1]

    obs_mu, obs_sd = compute_obs_norm(obs_start, train_idx)
    obs_norm_all = (obs_start - obs_mu) / obs_sd

    model = FactorizedCondVQVAE(seg_dim, obs_dim, G, K, D=D, hidden=hidden, decay=decay, dead_steps=dead_steps).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    train_x = torch.from_numpy(segs[train_idx]).to(device)
    train_o = torch.from_numpy(obs_norm_all[train_idx].astype(np.float32)).to(device)
    val_x = torch.from_numpy(segs[val_idx]).to(device)
    val_o = torch.from_numpy(obs_norm_all[val_idx].astype(np.float32)).to(device)
    n_train = len(train_idx)

    batch_rng = np.random.default_rng(seed + 1)
    loss_curve = []

    import time

    t0 = time.time()

    def run_steps(n_steps, start_step):
        final_commit = None
        for i in range(n_steps):
            step = start_step + i
            bidx = batch_rng.integers(0, n_train, size=batch)
            x = train_x[bidx]
            o = train_o[bidx]
            recon, _idx, commit_loss = model(x, o, training=True)
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

    # --- plateau check ---
    extended = False
    if len(loss_curve) >= PLATEAU_WINDOW:
        window = [p[1] for p in loss_curve[-PLATEAU_WINDOW:]]
        rel_change = (window[0] - window[-1]) / max(window[0], 1e-8)
    else:
        rel_change = None
    plateau_info = dict(
        rel_change_at_check=rel_change,
        threshold=PLATEAU_REL_THRESH,
        window=PLATEAU_WINDOW,
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
        train_recon, train_codes, _ = model(train_x, train_o, training=False)
        recon_mse_train = float(F.mse_loss(train_recon, train_x).item())
        val_recon, val_codes, _ = model(val_x, val_o, training=False)
        recon_mse_val = float(F.mse_loss(val_recon, val_x).item())
        all_codes = torch.cat([train_codes, val_codes], dim=0).numpy()  # (N, G)

    baseline_vec = segs[train_idx].mean(0, keepdims=True)
    baseline_mse_train = float(np.mean((segs[train_idx] - baseline_vec) ** 2))
    baseline_mse_val = float(np.mean((segs[val_idx] - baseline_vec) ** 2))

    per_group = []
    any_collapsed = False
    for g in range(G):
        stats = _usage_stats(all_codes[:, g], K)
        collapsed_g = (stats["n_active_codes"] < K) or (stats["top1_usage_frac"] > collapse_top1_thresh)
        stats["collapsed"] = bool(collapsed_g)
        any_collapsed = any_collapsed or collapsed_g
        per_group.append(stats)

    metrics = dict(
        recon_mse_train=recon_mse_train,
        recon_mse_val=recon_mse_val,
        baseline_mse_train=baseline_mse_train,
        baseline_mse_val=baseline_mse_val,
        val_improvement_over_baseline_pct=100.0 * (1.0 - recon_mse_val / baseline_mse_val),
        commit_loss_final=final_commit,
        restarts_total=int(sum(vq.restarts_total.item() for vq in model.vq.groups)),
        any_group_collapsed=bool(any_collapsed),
        collapse_top1_thresh=collapse_top1_thresh,
        total_steps_used=total_steps_used,
    )

    results = dict(
        metrics=metrics,
        per_group_usage=per_group,
        plateau=plateau_info,
        loss_curve=loss_curve,
        timing=dict(train_seconds=train_seconds),
    )
    extra = dict(obs_mu=obs_mu, obs_sd=obs_sd)
    return model, results, all_codes, extra


# ------------------------------------------------------------------
# 因子有效性：只換第 g 組的字，量展開動作的 Δ MSE + 組間 Δ 方向相關
# ------------------------------------------------------------------
def factor_validity(model, data, obs_mu, obs_sd, G, K, n_samples=256, seed=0):
    rng = np.random.default_rng(seed)
    segs = data["segs"]
    obs_start = data["obs_start"]
    val_idx = data["val_idx"]

    pool = val_idx if len(val_idx) >= n_samples else np.concatenate([data["train_idx"], val_idx])
    pick = rng.choice(pool, size=min(n_samples, len(pool)), replace=False)
    n = len(pick)

    x = torch.from_numpy(segs[pick]).float()
    obs_norm = torch.from_numpy(((obs_start[pick] - obs_mu) / obs_sd).astype(np.float32))

    model.eval()
    with torch.no_grad():
        z_e = model.encoder(x)
        _, orig_idx, _ = model.vq(z_e, training=False)  # (n, G)
        orig_recon = model.decode_codes(orig_idx, obs_norm)  # (n, seg_dim)

        delta_mse_per_group = []
        delta_vecs = []  # list of (n, seg_dim) per group
        rng_torch = torch.Generator().manual_seed(seed + 1)
        for g in range(G):
            offset = torch.randint(1, K, (n,), generator=rng_torch)  # 1..K-1
            new_code_g = (orig_idx[:, g] + offset) % K  # 保證 != 原字
            mod_idx = orig_idx.clone()
            mod_idx[:, g] = new_code_g
            mod_recon = model.decode_codes(mod_idx, obs_norm)
            delta = mod_recon - orig_recon
            delta_mse = float((delta.pow(2).mean(dim=1)).mean().item())
            delta_mse_per_group.append(delta_mse)
            delta_vecs.append(delta)

        # 組間 Δ 方向相關（每個 sample 算 cosine，再對 sample 取平均）
        corr_matrix = np.zeros((G, G), dtype=np.float64)
        eps = 1e-8
        norms = [dv.norm(dim=1) for dv in delta_vecs]
        for g1 in range(G):
            for g2 in range(G):
                dot = (delta_vecs[g1] * delta_vecs[g2]).sum(dim=1)
                denom = norms[g1] * norms[g2] + eps
                cos = (dot / denom).numpy()
                cos = cos[np.isfinite(cos)]
                corr_matrix[g1, g2] = float(np.mean(cos)) if len(cos) > 0 else 0.0

        # 退化檢查（G=32 這類）：每組 delta 向量的能量是否集中在單一原始維度
        top_dim_energy_frac = []
        for g in range(G):
            abs_mean = delta_vecs[g].abs().mean(dim=0).numpy()  # (seg_dim,)
            total_e = abs_mean.sum()
            top_e = abs_mean.max()
            top_dim_energy_frac.append(float(top_e / total_e) if total_e > 1e-9 else 0.0)

    off_diag = corr_matrix[~np.eye(G, dtype=bool)] if G > 1 else np.array([0.0])
    return dict(
        n_samples=n,
        delta_mse_per_group=delta_mse_per_group,
        mean_delta_mse=float(np.mean(delta_mse_per_group)),
        min_delta_mse=float(np.min(delta_mse_per_group)),
        max_delta_mse=float(np.max(delta_mse_per_group)),
        pairwise_cosine_corr=corr_matrix.tolist(),
        mean_abs_offdiag_corr=float(np.mean(np.abs(off_diag))) if G > 1 else None,
        top_dim_energy_frac_per_group=top_dim_energy_frac,
        mean_top_dim_energy_frac=float(np.mean(top_dim_energy_frac)),
    )


# ------------------------------------------------------------------
# Pillow：膝點曲線
# ------------------------------------------------------------------
def _load_font():
    from PIL import ImageFont

    return ImageFont.load_default()


def draw_knee_curve(cells, out_path, knee_capacity=None, title="val MSE vs total capacity G*log2(K)"):
    """cells: list of dict(G,K,capacity,val_mse,collapsed,recommended[opt])"""
    from PIL import Image, ImageDraw

    W, H = 980, 620
    margin_l, margin_r, margin_t, margin_b = 80, 40, 60, 70
    plot_w = W - margin_l - margin_r
    plot_h = H - margin_t - margin_b

    img = Image.new("RGB", (W, H), hex_to_rgb(SURFACE))
    draw = ImageDraw.Draw(img)
    font = _load_font()

    xs = [c["capacity"] for c in cells]
    ys = [c["val_mse"] for c in cells]
    xmin, xmax = 0, max(xs) * 1.05
    ymin, ymax = 0, max(ys) * 1.1

    def to_px(x, y):
        px = margin_l + (x - xmin) / (xmax - xmin) * plot_w
        py = margin_t + plot_h - (y - ymin) / (ymax - ymin) * plot_h
        return px, py

    # gridlines
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = margin_t + plot_h - plot_h * frac
        draw.line([(margin_l, y), (margin_l + plot_w, y)], fill=hex_to_rgb(GRIDLINE), width=1)
        draw.text((8, y - 6), f"{ymax*frac:.3f}", fill=hex_to_rgb(INK_MUTED), font=font)
        x = margin_l + plot_w * frac
        draw.line([(x, margin_t), (x, margin_t + plot_h)], fill=hex_to_rgb(GRIDLINE), width=1)
        draw.text((x - 10, margin_t + plot_h + 6), f"{xmax*frac:.0f}", fill=hex_to_rgb(INK_MUTED), font=font)

    # lower envelope line: min val_mse at each distinct capacity
    by_cap = {}
    for c in cells:
        by_cap.setdefault(c["capacity"], []).append(c["val_mse"])
    env_caps = sorted(by_cap.keys())
    env_pts = [(cap, min(by_cap[cap])) for cap in env_caps]
    for i in range(len(env_pts) - 1):
        p0 = to_px(*env_pts[i])
        p1 = to_px(*env_pts[i + 1])
        draw.line([p0, p1], fill=hex_to_rgb(INK_MUTED), width=2)

    # scatter points
    color_g = {}
    gs = sorted(set(c["G"] for c in cells))
    for i, g in enumerate(gs):
        color_g[g] = hex_to_rgb(PALETTE_CATEGORICAL[i % len(PALETTE_CATEGORICAL)])

    for c in cells:
        px, py = to_px(c["capacity"], c["val_mse"])
        col = color_g[c["G"]]
        r = 7 if c.get("collapsed") else 5
        if c.get("collapsed"):
            draw.ellipse([px - r, py - r, px + r, py + r], outline=hex_to_rgb(INK_PRIMARY), width=2)
        draw.ellipse([px - r, py - r, px + r, py + r], fill=col)
        if c.get("recommended"):
            draw.ellipse([px - r - 6, py - r - 6, px + r + 6, py + r + 6], outline=hex_to_rgb("#e34948"), width=3)
        draw.text((px + 8, py - 6), f"G{c['G']}K{c['K']}", fill=hex_to_rgb(INK_SECONDARY), font=font)

    if knee_capacity is not None:
        px, _ = to_px(knee_capacity, ymin)
        draw.line([(px, margin_t), (px, margin_t + plot_h)], fill=hex_to_rgb("#e34948"), width=1)
        draw.text((px + 4, margin_t + 4), "knee", fill=hex_to_rgb("#e34948"), font=font)

    draw.line([(margin_l, margin_t), (margin_l, margin_t + plot_h)], fill=hex_to_rgb(AXIS), width=2)
    draw.line([(margin_l, margin_t + plot_h), (margin_l + plot_w, margin_t + plot_h)], fill=hex_to_rgb(AXIS), width=2)
    draw.text((margin_l, 16), title, fill=hex_to_rgb(INK_PRIMARY), font=font)
    draw.text((margin_l, H - 20), "capacity = G * log2(K)  [bits]", fill=hex_to_rgb(INK_MUTED), font=font)
    draw.text((10, margin_t - 20), "val MSE", fill=hex_to_rgb(INK_MUTED), font=font)

    # legend: color=G, ring=collapsed, red halo=recommended
    ly = margin_t
    for i, g in enumerate(gs):
        lx = margin_l + plot_w - 90
        yy = ly + i * 16
        draw.ellipse([lx, yy, lx + 10, yy + 10], fill=color_g[g])
        draw.text((lx + 14, yy - 2), f"G={g}", fill=hex_to_rgb(INK_SECONDARY), font=font)

    img.save(out_path)


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


# ------------------------------------------------------------------
# MuJoCo rollout（合成 action，需真的 step 物理引擎，因為沒有對應的真實 qpos）
# ------------------------------------------------------------------
def make_render_env(dataset_dir):
    import ogbench

    env = ogbench.make_env_and_datasets(DATASET_NAME, dataset_dir=dataset_dir, env_only=True)
    u = env.unwrapped
    env.reset()
    u.set_state(u.data.qpos.copy(), u.data.qvel.copy())
    _ = u.render()
    renderer = u.custom_renderer
    return dict(env=env, u=u, renderer=renderer)


def rollout_from_actions(env_state, qpos0, qvel0, actions):
    """從真實資料的段起點 qpos0/qvel0 出發，套用一串合成 actions，回傳逐步 qpos/qvel 軌跡（含起點）。"""
    u = env_state["u"]
    u.set_state(qpos0.copy(), qvel0.copy())
    qpos_seq = [u.data.qpos.copy()]
    qvel_seq = [u.data.qvel.copy()]
    for a in actions:
        a_clipped = np.clip(a, -1.0, 1.0).astype(np.float64)
        u.step(a_clipped)
        qpos_seq.append(u.data.qpos.copy())
        qvel_seq.append(u.data.qvel.copy())
    return qpos_seq, qvel_seq


def draw_ab_fallback_plot(actions_a, actions_b, out_path, title):
    """MuJoCo render 失敗時的退路：A/B 兩段合成動作的 8 維曲線疊圖對比。

    actions_a/actions_b: np.array[seg_len, act_dim]（原字 vs 換字後 decode 出來的動作）。
    """
    from PIL import Image, ImageDraw

    W, H = 900, 420
    img = Image.new("RGB", (W, H), hex_to_rgb(SURFACE))
    draw = ImageDraw.Draw(img)
    font = _load_font()
    dim_colors = [hex_to_rgb(c) for c in PALETTE_CATEGORICAL]

    seg_len, act_dim = actions_a.shape
    panels = [("A: original code", actions_a), ("B: swapped code", actions_b)]
    col_w = (W - 40) / 2
    for i, (label, acts) in enumerate(panels):
        px0 = 20 + i * col_w + 10
        px1 = 20 + (i + 1) * col_w - 10
        py0, py1 = 60, 340
        draw.rectangle([px0, py0, px1, py1], outline=hex_to_rgb(AXIS))
        mid_y = (py0 + py1) / 2
        draw.line([(px0, mid_y), (px1, mid_y)], fill=hex_to_rgb(GRIDLINE))
        T = acts.shape[0]
        for d in range(act_dim):
            pts = []
            for t in range(T):
                x = px0 + (px1 - px0) * (t / max(1, T - 1))
                v = float(np.clip(acts[t, d], -1, 1))
                y = mid_y - v * (py1 - py0) / 2 * 0.95
                pts.append((x, y))
            draw.line(pts, fill=dim_colors[d % len(dim_colors)], width=2)
        draw.text((px0, py1 + 4), label, fill=hex_to_rgb(INK_SECONDARY), font=font)

    leg_y = 400
    for d in range(act_dim):
        lx = 20 + d * 55
        draw.line([(lx, leg_y), (lx + 14, leg_y)], fill=dim_colors[d % len(dim_colors)], width=3)
        draw.text((lx + 16, leg_y - 6), f"a{d}", fill=hex_to_rgb(INK_SECONDARY), font=font)

    draw.text((20, 14), title, fill=hex_to_rgb(INK_PRIMARY), font=font)
    img.save(out_path)


def render_side_by_side_gif(env_state, qpos_a, qvel_a, qpos_b, qvel_b, out_path, label_a, label_b, mid_xy=None):
    import mujoco
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw

    u = env_state["u"]
    renderer = env_state["renderer"]
    font = _load_font()

    if mid_xy is None:
        mid_xy = qpos_a[0][:2]
    cam = mujoco.MjvCamera()
    cam.lookat[0] = float(mid_xy[0])
    cam.lookat[1] = float(mid_xy[1])
    cam.lookat[2] = 0.3
    cam.distance = 3.0
    cam.elevation = -35
    cam.azimuth = 135

    def render_seq(qpos_seq, qvel_seq, label):
        frames = []
        for qp, qv in zip(qpos_seq, qvel_seq):
            u.set_state(qp.copy(), qv.copy())
            renderer.update_scene(u.data, camera=cam)
            frame = renderer.render().copy()
            im = Image.fromarray(frame)
            draw = ImageDraw.Draw(im)
            draw.text((4, 4), label, fill=(255, 255, 0), font=font)
            frames.append(np.array(im))
        return frames

    frames_a = render_seq(qpos_a, qvel_a, label_a)
    frames_b = render_seq(qpos_b, qvel_b, label_b)
    n = min(len(frames_a), len(frames_b))
    combined = [np.concatenate([frames_a[i], frames_b[i]], axis=1) for i in range(n)]
    imageio.mimsave(out_path, combined, duration=0.3, loop=0)
    return n
