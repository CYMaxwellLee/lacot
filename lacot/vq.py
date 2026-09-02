"""VQ 錨定模組（離散化階梯第二層、2026-09-02）。

設計（DESIGN-DRAFT-2026-09-02-vq-anchor.md）：
- 每個 token 各自量化：u [B, K, D] → 每個 [D] 向量 snap 到 codebook（V 個 code）⇒ V^K 個「句子」。
- codebook 走 EMA 更新（VQ-VAE-2 / Sonnet 做法），⛔ 不進 optimizer；encoder 靠 commitment loss 被拉向 code。
- straight-through：前向用 u_q，反向梯度直通 u（u + (u_q − u).detach()）。
- 死 code 重置：連續 `dead_steps` 步沒被用到的 code，用當批隨機樣本覆蓋（防 codebook 崩到少數幾個 code）。
- stochastic 量化（可選、DLR 風）：訓練時以機率 p 在距離上加噪再取最近，避免早期鎖死。
- ⛔ flow 不吃 u_q：flow 仍對連續 u 建模（NF 不能對點質量建模）；推論時 flow.sample → snap → decoder。
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenVQ(nn.Module):
    def __init__(self, n_codes: int, dim: int, beta: float = 0.25, ema_decay: float = 0.99,
                 dead_steps: int = 200, noise_p: float = 0.0, noise_scale: float = 0.1, eps: float = 1e-5):
        super().__init__()
        self.V, self.D, self.beta = int(n_codes), int(dim), float(beta)
        self.decay, self.eps = float(ema_decay), float(eps)
        self.dead_steps, self.noise_p, self.noise_scale = int(dead_steps), float(noise_p), float(noise_scale)
        # codebook 與 EMA 統計都是 buffer（⛔ 不是 Parameter：EMA 更新、不吃梯度）
        self.register_buffer("codebook", torch.randn(self.V, self.D) * 0.1)
        self.register_buffer("ema_count", torch.ones(self.V))
        self.register_buffer("ema_sum", self.codebook.clone())
        self.register_buffer("unused", torch.zeros(self.V, dtype=torch.long))
        self.register_buffer("initialized", torch.tensor(0, dtype=torch.long))

    # ---- 查表 ----
    @torch.no_grad()
    def _nearest(self, flat: torch.Tensor, stochastic: bool) -> torch.Tensor:
        # flat [N, D] → idx [N]；距離 ||x−c||² = |x|² − 2x·c + |c|²
        d = (flat.pow(2).sum(1, keepdim=True) - 2 * flat @ self.codebook.t()
             + self.codebook.pow(2).sum(1)[None, :])
        if stochastic and self.noise_p > 0 and torch.rand(()) < self.noise_p:
            d = d + self.noise_scale * d.std() * torch.randn_like(d)
        return d.argmin(1)

    @torch.no_grad()
    def snap(self, u: torch.Tensor) -> torch.Tensor:
        """推論用：u [..., D] → 最近 code 的向量（無梯度、無噪）。"""
        shp = u.shape
        idx = self._nearest(u.reshape(-1, self.D), stochastic=False)
        return self.codebook[idx].reshape(shp)

    @torch.no_grad()
    def codes(self, u: torch.Tensor) -> torch.Tensor:
        return self._nearest(u.reshape(-1, self.D), stochastic=False).reshape(u.shape[:-1])

    # ---- 訓練 ----
    def forward(self, u: torch.Tensor):
        """u [B, K, D]（或 [..., D]）→ (u_st [同形], loss_commit [scalar], stats dict)。
        訓練模式下同時做 EMA codebook 更新與死 code 重置。"""
        shp = u.shape
        flat = u.reshape(-1, self.D)
        if self.training and int(self.initialized) == 0:        # 第一批：codebook 用真資料初始化（防一開始全死）
            n = min(flat.shape[0], self.V)
            pick = torch.randperm(flat.shape[0], device=flat.device)[:n]
            self.codebook[:n] = flat[pick].detach()
            self.ema_sum.copy_(self.codebook); self.ema_count.fill_(1.0); self.initialized.fill_(1)
        idx = self._nearest(flat.detach(), stochastic=self.training)
        u_q = self.codebook[idx]                                   # [N, D]
        # commitment：拉 encoder 往 code（codebook 本身走 EMA、不吃這項梯度）
        loss = self.beta * F.mse_loss(flat, u_q.detach())
        u_st = flat + (u_q - flat).detach()                        # straight-through
        with torch.no_grad():
            onehot = F.one_hot(idx, self.V).to(flat.dtype)         # [N, V]
            counts = onehot.sum(0)                                 # [V]
            probs = counts / counts.sum().clamp_min(1.0)
            perplexity = torch.exp(-(probs * (probs + 1e-10).log()).sum())
            if self.training:
                self.ema_count.mul_(self.decay).add_(counts, alpha=1 - self.decay)
                self.ema_sum.mul_(self.decay).add_(onehot.t() @ flat.detach(), alpha=1 - self.decay)
                nsum = self.ema_count.sum()
                cnt = (self.ema_count + self.eps) / (nsum + self.V * self.eps) * nsum   # Laplace 平滑
                self.codebook.copy_(self.ema_sum / cnt[:, None])
                # 死 code 重置
                self.unused = torch.where(counts > 0, torch.zeros_like(self.unused), self.unused + 1)
                dead = self.unused >= self.dead_steps
                if bool(dead.any()):
                    nd = int(dead.sum())
                    pick = torch.randint(0, flat.shape[0], (nd,), device=flat.device)
                    self.codebook[dead] = flat[pick].detach()
                    self.ema_sum[dead] = flat[pick].detach(); self.ema_count[dead] = 1.0
                    self.unused[dead] = 0
        stats = dict(perplexity=float(perplexity), used=int((counts > 0).sum()),
                     commit=float(loss.detach()))
        return u_st.reshape(shp), loss, stats
