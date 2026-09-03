"""FSQ 錨定（離散化階梯第二層、家族主候選；2026-09-03）。

出處：Mentzer 2023, arXiv 2309.15505《Finite Scalar Quantization: VQ-VAE Made Simple》＋
DESIGN-DRAFT-2026-09-02-vq-anchor.md「選擇與理由」。

- 每 token：D 維 → proj 下到 d 維 → tanh 有界 → round 到每維 L 格（STE）→ proj 回 D 維。
- 隱式 codebook＝L^d（d=8,L=8 ⇒ 1.7e7 格/token）：⛔ 沒有 codebook、沒有 EMA／死 code／commitment 機關。
- 格子由 tanh×(L−1)/2 固定 ⇒ 字彙跟 seed 無關（9/3 方言判決後選它的理由）。
- ⛔ flow 不對格點建模（NF 對點質量 log-density 爆）：訓練目標用 dequant()（格點＋均勻噪聲＝連續化）、
  推論 sample 後 snap()。
"""
from __future__ import annotations
import torch
import torch.nn as nn


class TokenFSQ(nn.Module):
    def __init__(self, dim: int, d: int = 8, L: int = 8):
        super().__init__()
        self.dim, self.d, self.L = int(dim), int(d), int(L)
        self.half = (self.L - 1) / 2.0
        self.down = nn.Linear(self.dim, self.d)
        self.up = nn.Linear(self.d, self.dim)

    def _z(self, u: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.down(u)) * self.half     # 有界 [-half, half]

    def _round_ste(self, z: torch.Tensor) -> torch.Tensor:
        return z + (torch.round(z) - z).detach()

    def forward(self, u: torch.Tensor):
        """訓練（fit）：u [..., D] → (u_q [..., D], z_round)。梯度走 STE。"""
        zq = self._round_ste(self._z(u))
        return self.up(zq), zq

    @torch.no_grad()
    def snap(self, u: torch.Tensor) -> torch.Tensor:
        """推論：u → 最近格點的 D 維向量（無梯度）。"""
        return self.up(torch.round(self._z(u)))

    @torch.no_grad()
    def dequant(self, u: torch.Tensor) -> torch.Tensor:
        """flow 訓練目標：格點＋U(−.5,.5)（z 空間格距＝1）→ proj 回。離散支撐連續化。"""
        zq = torch.round(self._z(u))
        return self.up(zq + torch.rand_like(zq) - 0.5)

    @torch.no_grad()
    def codes(self, u: torch.Tensor) -> torch.Tensor:
        """每 token 的格點座標（[..., d]、整數值）——可讀、可計數（缺課地圖用）。"""
        return torch.round(self._z(u))
