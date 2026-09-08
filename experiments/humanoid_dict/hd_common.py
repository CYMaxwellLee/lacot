"""G×K 因子化 VQ 掃描共用模組 —— humanoidmaze-medium 版（見 docs/NOTE-2026-09-08-humanoid-dict.md）。

主人裁示：把 experiments/gk_scan/（antmaze-medium-stitch-v0，見
docs/NOTE-2026-09-08-gk-scan-route.md）的同一套方法論搬到 humanoidmaze-medium，
抓 humanoid 的字典設計甜蜜點、跟 ant 對照。

本檔 = gk_scan/gk_common.py 的模型/訓練/量測/render 核心邏輯（幾乎逐行沿用，
只換資料集常數）＋ 從 walk_verify/wv_common.py 搬來的「量尺包」建構邏輯（ruler，
build_ruler + ruler_steps_for_distance + draw_hist_overlay）＋ 新增
render_single_traj_gif（給「單碼本推薦格的字」的原始段 gif 用，播真實 qpos，
不需要重新 step 物理引擎）。

⛔ 獨立成檔，不 import gk_scan/ 或 walk_verify/（避免跨實驗目錄依賴，同 gk_scan
docstring 的既定紀律）。⛔ 不修改被抄的來源檔案。

資料事實（一手實測，見 build_ruler.py 印出的 log）：
  humanoidmaze-medium-stitch-v0：5000 episodes × 401 steps/ep（= 2,005,000 步，
  跟 ant 的 5000×201 同 n_episodes、但 ep_len 是 2 倍）。
  act_dim=21（ant 是 8）、obs_dim=69（ant 是 29；humanoid 的 obs 不是單純
  concat(qpos,qvel)，是 ogbench HumanoidEnv.get_ob() 的自訂特徵：
  xy(2)+joint_angles(21)+head_height(1)+extremities(12)+torso_vert(3)+
  com_vel(3)+qvel(27)=69，見 ogbench/locomaze/humanoid.py:get_ob）。
  qpos 維度 28（7 自由關節 + 21 actuated）、qvel 維度 27（6+21）。
  軀幹（root/pelvis）z＝qpos[:,2]，跟 ant 的「軀幹 z」取法一致（不是 head_height，
  head_height 是 obs 裡的另一個獨立特徵，本次沒有另外用它）。
  goal_tol = 0.5（跟 ant 相同，ogbench/locomaze/maze.py:86 對 point 以外都是 0.5）。
"""
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------------------------------------------
# 常數（humanoidmaze-medium-stitch-v0 一手實測值，見上方 docstring）
# ------------------------------------------------------------------
ACT_DIM = 21
OBS_DIM = 69
EP_LEN = 401  # humanoidmaze-medium-stitch-v0：5000 episodes × 401 steps（本次實測）
DATASET_NAME = "humanoidmaze-medium-stitch-v0"
GROUP_DIM = 8  # 每組 embedding 維度：沿用 ant 掃描的假定值（任務指定的預設值），本次沒有另外掃描
GOAL_TOL = 0.5  # ogbench/locomaze/maze.py:86，humanoid 與 ant 同值

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


def _font():
    from PIL import ImageFont

    return ImageFont.load_default()


# ------------------------------------------------------------------
# 資料：切動作段 + 段起點 obs（跟 gk_scan 逐行同款，EP_LEN/ACT_DIM/OBS_DIM 換成 humanoid）
# ------------------------------------------------------------------
def episode_bounds(terminals):
    ends = np.flatnonzero(terminals)
    starts = np.concatenate([[0], ends[:-1] + 1])
    return starts, ends


def load_raw(data_dir, name=DATASET_NAME, split="train"):
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
    return out


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
    mu = obs_start[train_idx].mean(0, keepdims=True)
    sd = obs_start[train_idx].std(0, keepdims=True)
    sd = np.maximum(sd, eps)
    return mu.astype(np.float32), sd.astype(np.float32)


# ------------------------------------------------------------------
# 模型：MLP + 每組獨立 VQ-EMA + 條件化 decoder（跟 gk_scan 逐行同款）
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
    """單一 codebook，VQ-EMA + dead-code restart（沿用 M0/gk_scan 驗證過的實作邏輯）。"""

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
        chunks = z_e_full.split(self.D, dim=1)
        z_q_chunks, idx_chunks, commit_losses = [], [], []
        for g, vq in enumerate(self.groups):
            z_q_st, idx, commit_loss = vq(chunks[g], training=training)
            z_q_chunks.append(z_q_st)
            idx_chunks.append(idx)
            commit_losses.append(commit_loss)
        z_q_full = torch.cat(z_q_chunks, dim=1)
        idx_full = torch.stack(idx_chunks, dim=1)  # (B, G)
        commit_loss_mean = torch.stack(commit_losses).mean()  # 對 G 取平均，理由同 gk_scan
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
        z_q_chunks = [self.vq.groups[g].embed[code_idx[:, g]] for g in range(self.G)]
        z_q = torch.cat(z_q_chunks, dim=1)
        return self.decoder(torch.cat([z_q, obs_norm], dim=1))


# ------------------------------------------------------------------
# 訓練與量測（跟 gk_scan 逐行同款：plateau 判準、seed 由呼叫端決定）
# ------------------------------------------------------------------
PLATEAU_WINDOW = 5
PLATEAU_REL_THRESH = 0.03
EXTEND_TO_STEPS = 50000


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
# 因子有效性（跟 gk_scan 逐行同款）
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
        _, orig_idx, _ = model.vq(z_e, training=False)
        orig_recon = model.decode_codes(orig_idx, obs_norm)

        delta_mse_per_group = []
        delta_vecs = []
        rng_torch = torch.Generator().manual_seed(seed + 1)
        for g in range(G):
            offset = torch.randint(1, K, (n,), generator=rng_torch)
            new_code_g = (orig_idx[:, g] + offset) % K
            mod_idx = orig_idx.clone()
            mod_idx[:, g] = new_code_g
            mod_recon = model.decode_codes(mod_idx, obs_norm)
            delta = mod_recon - orig_recon
            delta_mse = float((delta.pow(2).mean(dim=1)).mean().item())
            delta_mse_per_group.append(delta_mse)
            delta_vecs.append(delta)

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

        top_dim_energy_frac = []
        for g in range(G):
            abs_mean = delta_vecs[g].abs().mean(dim=0).numpy()
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
# 量尺包（搬自 walk_verify/wv_common.py:build_ruler，泛化成吃 act_dim 參數）
# ------------------------------------------------------------------
def build_ruler(raw, dist_grid=None, quantiles=(1, 5, 10, 25, 50, 75, 90, 95, 99),
                 max_horizon=400, sub=1, seed=0):
    """從資料量出四把尺（同 walk_verify 的定義，見 wv_common.py:build_ruler）：

    (a) distance -> steps 查表：first-passage 步數的分位數
    (b) 步速分佈：每步 xy 位移
    (c) 軀幹（root）z 高度分佈：翻倒線 = p1（雙足更容易翻，這把尺對 humanoid 更重要）
    (d) 相鄰步動作差 |a_t - a_(t-1)| 分佈（L2 over act_dim 維，不跨 episode）
    """
    if dist_grid is None:
        dist_grid = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0]
    qpos = raw["qpos"]
    act = raw["actions"]
    starts, ends = episode_bounds(raw["terminals"])
    xy = qpos[:, :2]

    step_disp = []
    for s0, e0 in zip(starts, ends):
        seg = xy[s0 : e0 + 1]
        step_disp.append(np.linalg.norm(np.diff(seg, axis=0), axis=1))
    step_disp = np.concatenate(step_disp)

    z = qpos[:, 2]

    act_diff = []
    for s0, e0 in zip(starts, ends):
        a = act[s0 : e0 + 1]
        act_diff.append(np.linalg.norm(np.diff(a, axis=0), axis=1))
    act_diff = np.concatenate(act_diff)

    rng = np.random.default_rng(seed)
    n_ep = len(starts)
    ep_sel = np.arange(n_ep) if sub <= 1 else np.sort(
        rng.choice(n_ep, size=max(1, n_ep // sub), replace=False))
    hits_by_d = {f"{d:g}": [] for d in dist_grid}
    n_total_by_d = {f"{d:g}": 0 for d in dist_grid}
    n_cens_by_d = {f"{d:g}": 0 for d in dist_grid}
    for ei in ep_sel:
        s0, e0 = int(starts[ei]), int(ends[ei])
        seg = xy[s0 : e0 + 1]
        T = len(seg)
        D = np.linalg.norm(seg[None, :, :] - seg[:, None, :], axis=2)
        ii = np.arange(T)
        gap = ii[None, :] - ii[:, None]
        window = (gap >= 1) & (gap <= max_horizon)
        for d in dist_grid:
            key = f"{d:g}"
            hit = (D >= d) & window
            any_hit = hit.any(1)
            first_j = hit.argmax(1)
            rows = np.flatnonzero(any_hit)
            hits_by_d[key].append((first_j[rows] - ii[rows]).astype(np.int64))
            n_total_by_d[key] += T - 1
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
        act_dim=int(act.shape[1]),
        obs_dim_note="obs_dim 另由 load_segments_with_obs 實測回報，這裡不重複存",
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
# Pillow：分佈疊圖、膝點曲線（跟 gk_scan / walk_verify 同款手畫，這台沒裝 matplotlib）
# ------------------------------------------------------------------
def draw_hist_overlay(series, out_path, title, xlabel, bins=60, xrange=None, vlines=None):
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


def draw_knee_curve(cells, out_path, knee_capacity=None, title="val MSE vs total capacity G*log2(K)"):
    """cells: list of dict(G,K,capacity,val_mse,collapsed,recommended[opt])"""
    from PIL import Image, ImageDraw

    W, H = 980, 620
    margin_l, margin_r, margin_t, margin_b = 80, 40, 60, 70
    plot_w = W - margin_l - margin_r
    plot_h = H - margin_t - margin_b

    img = Image.new("RGB", (W, H), hex_to_rgb(SURFACE))
    draw = ImageDraw.Draw(img)
    font = _font()

    xs = [c["capacity"] for c in cells]
    ys = [c["val_mse"] for c in cells]
    xmin, xmax = 0, max(xs) * 1.05
    ymin, ymax = 0, max(ys) * 1.1

    def to_px(x, y):
        px = margin_l + (x - xmin) / (xmax - xmin) * plot_w
        py = margin_t + plot_h - (y - ymin) / (ymax - ymin) * plot_h
        return px, py

    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = margin_t + plot_h - plot_h * frac
        draw.line([(margin_l, y), (margin_l + plot_w, y)], fill=hex_to_rgb(GRIDLINE), width=1)
        draw.text((8, y - 6), f"{ymax*frac:.3f}", fill=hex_to_rgb(INK_MUTED), font=font)
        x = margin_l + plot_w * frac
        draw.line([(x, margin_t), (x, margin_t + plot_h)], fill=hex_to_rgb(GRIDLINE), width=1)
        draw.text((x - 10, margin_t + plot_h + 6), f"{xmax*frac:.0f}", fill=hex_to_rgb(INK_MUTED), font=font)

    by_cap = {}
    for c in cells:
        by_cap.setdefault(c["capacity"], []).append(c["val_mse"])
    env_caps = sorted(by_cap.keys())
    env_pts = [(cap, min(by_cap[cap])) for cap in env_caps]
    for i in range(len(env_pts) - 1):
        p0 = to_px(*env_pts[i])
        p1 = to_px(*env_pts[i + 1])
        draw.line([p0, p1], fill=hex_to_rgb(INK_MUTED), width=2)

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
# MuJoCo rollout（跟 gk_scan 逐行同款：A/B 對比需要合成 action、真的 step 物理引擎）
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
    """MuJoCo render 失敗時的退路：A/B 兩段合成動作的 act_dim 維曲線疊圖對比。"""
    from PIL import Image, ImageDraw

    W, H = 900, 460
    img = Image.new("RGB", (W, H), hex_to_rgb(SURFACE))
    draw = ImageDraw.Draw(img)
    font = _font()
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
            draw.line(pts, fill=dim_colors[d % len(dim_colors)], width=1)
        draw.text((px0, py1 + 4), label, fill=hex_to_rgb(INK_SECONDARY), font=font)

    draw.text((20, 14), title, fill=hex_to_rgb(INK_PRIMARY), font=font)
    draw.text((20, H - 18), f"act_dim={act_dim}（21 條線疊在一起，顏色循環 8 色）", fill=hex_to_rgb(INK_MUTED), font=font)
    img.save(out_path)


def draw_single_action_plot(actions, out_path, title):
    """render env 建不起來時的退路（給字典字 render 用）：單段 act_dim 維動作曲線圖。"""
    from PIL import Image, ImageDraw

    W, H = 760, 420
    img = Image.new("RGB", (W, H), hex_to_rgb(SURFACE))
    draw = ImageDraw.Draw(img)
    font = _font()
    dim_colors = [hex_to_rgb(c) for c in PALETTE_CATEGORICAL]

    seg_len, act_dim = actions.shape
    px0, px1 = 30, W - 30
    py0, py1 = 60, 340
    draw.rectangle([px0, py0, px1, py1], outline=hex_to_rgb(AXIS))
    mid_y = (py0 + py1) / 2
    draw.line([(px0, mid_y), (px1, mid_y)], fill=hex_to_rgb(GRIDLINE))
    T = actions.shape[0]
    for d in range(act_dim):
        pts = []
        for t in range(T):
            x = px0 + (px1 - px0) * (t / max(1, T - 1))
            v = float(np.clip(actions[t, d], -1, 1))
            y = mid_y - v * (py1 - py0) / 2 * 0.95
            pts.append((x, y))
        draw.line(pts, fill=dim_colors[d % len(dim_colors)], width=1)
    draw.text((20, 14), title, fill=hex_to_rgb(INK_PRIMARY), font=font)
    draw.text((20, H - 18), f"act_dim={act_dim}（顏色循環 8 色）", fill=hex_to_rgb(INK_MUTED), font=font)
    img.save(out_path)


def render_side_by_side_gif(env_state, qpos_a, qvel_a, qpos_b, qvel_b, out_path, label_a, label_b, mid_xy=None):
    import mujoco
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw

    u = env_state["u"]
    renderer = env_state["renderer"]
    font = _font()

    if mid_xy is None:
        mid_xy = qpos_a[0][:2]
    cam = mujoco.MjvCamera()
    cam.lookat[0] = float(mid_xy[0])
    cam.lookat[1] = float(mid_xy[1])
    cam.lookat[2] = 0.8
    cam.distance = 3.5
    cam.elevation = -25
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


def render_single_traj_gif(env_state, qpos_seq, qvel_seq, out_path, label, mid_xy=None):
    """播「真實」段的 qpos/qvel（不合成、不 step 物理引擎，直接 set_state 逐格 render）。

    給「單碼本推薦格的字」用：字對應到的是真實資料裡的一段，不是換字合成的動作，
    直接播真實 qpos 序列最忠實、也不需要假設物理引擎能重演出一樣的軌跡。
    """
    import mujoco
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw

    u = env_state["u"]
    renderer = env_state["renderer"]
    font = _font()

    if mid_xy is None:
        mid_xy = qpos_seq[0][:2]
    cam = mujoco.MjvCamera()
    cam.lookat[0] = float(mid_xy[0])
    cam.lookat[1] = float(mid_xy[1])
    cam.lookat[2] = 0.8
    cam.distance = 3.0
    cam.elevation = -20
    cam.azimuth = 135

    frames = []
    for qp, qv in zip(qpos_seq, qvel_seq):
        u.set_state(qp.copy(), qv.copy())
        renderer.update_scene(u.data, camera=cam)
        frame = renderer.render().copy()
        im = Image.fromarray(frame)
        draw = ImageDraw.Draw(im)
        draw.text((4, 4), label, fill=(255, 255, 0), font=font)
        frames.append(np.array(im))
    imageio.mimsave(out_path, frames, duration=0.3, loop=0)
    return len(frames)
