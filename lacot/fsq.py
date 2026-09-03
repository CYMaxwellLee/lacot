"""FSQ 錨定（離散化階梯第二層、家族主候選；2026-09-03）。

出處：Mentzer 2023, arXiv 2309.15505《Finite Scalar Quantization: VQ-VAE Made Simple》＋
DESIGN-DRAFT-2026-09-02-vq-anchor.md「選擇與理由」。

- 每 token：D 維 → proj 下到 d 維 → 有界（tanh）→ round 到每維 L 格（STE）→ proj 回 D 維。
- 隱式 codebook＝L^d：⛔ 沒有 codebook、沒有 EMA／死 code／commitment 機關。
- ⭐ 9/3 修刻度（主人「這八個數字有幾種選項」抓出來的）：v1 對 even L 沒做半格偏移，
  tanh×(L−1)/2 再 round 實際只有 L−1 個整數刻度、且 tanh 浮點飽和時溢出到 ±L/2。
  v2：even L 用半整數格 {−(L−1)/2, …, −0.5, 0.5, …, (L−1)/2}（先 round(z−0.5) 再 +0.5）、
  odd L 用整數格；一律 clamp 擋飽和溢出 ⇒ 刻度數恰為 L、永不溢出。
- ⛔ flow 不對格點建模（NF 對點質量／低維薄片 log-density 病態——9/3 臂 B 實證）：
  u 空間版（歷史）目標用 dequant()；z 空間版（9/3 晚設計）flow 直接在 d 維 z 上建模，
  目標用 z_of()（甲：連續）或 dequant_z()（乙：格點＋均勻噪聲），推論 sample 後
  quantize_z → up 回 D 維。
"""
from __future__ import annotations
import torch
import torch.nn as nn


class TokenFSQ(nn.Module):
    def __init__(self, dim: int, d: int = 8, L: int = 8):
        super().__init__()
        self.dim, self.d, self.L = int(dim), int(d), int(L)
        self.even = (self.L % 2 == 0)
        # tanh 的放大倍率：even 用 L/2（半整數格覆蓋 (−L/2, L/2)）、odd 用 (L−1)/2（整數格）
        self.scale = self.L / 2.0 if self.even else (self.L - 1) / 2.0
        self.down = nn.Linear(self.dim, self.d)
        self.up = nn.Linear(self.d, self.dim)

    # ---- z 空間（d 維）----
    def z_of(self, u: torch.Tensor) -> torch.Tensor:
        """u [..., D] → 有界連續 z [..., d]（甲目標；未量化）。"""
        return torch.tanh(self.down(u)) * self.scale

    def _grid(self, z: torch.Tensor) -> torch.Tensor:
        """z → 格點值（無梯度路徑；clamp 擋 tanh 浮點飽和的溢出）。刻度數恰為 L。"""
        if self.even:
            q = torch.round(z - 0.5).clamp(-self.L // 2, self.L // 2 - 1) + 0.5
        else:
            h = (self.L - 1) // 2
            q = torch.round(z).clamp(-h, h)
        return q

    def quantize_z(self, z: torch.Tensor) -> torch.Tensor:
        """z → 格點（STE：前向格點值、反向梯度直通）。"""
        return z + (self._grid(z) - z).detach()

    @torch.no_grad()
    def dequant_z(self, u: torch.Tensor) -> torch.Tensor:
        """乙目標：格點＋U(−.5,.5)（z 空間格距＝1、d 維滿秩 ⇒ 無薄片病）。"""
        q = self._grid(self.z_of(u))
        return q + torch.rand_like(q) - 0.5

    # ---- u 空間（D 維；9/3 白天兩臂用、留作歷史對照）----
    def forward(self, u: torch.Tensor):
        """訓練（fit）：u [..., D] → (u_q [..., D], z_q)。梯度走 STE。"""
        zq = self.quantize_z(self.z_of(u))
        return self.up(zq), zq

    @torch.no_grad()
    def snap(self, u: torch.Tensor) -> torch.Tensor:
        """u → 最近格點的 D 維向量（無梯度）。"""
        return self.up(self._grid(self.z_of(u)))

    @torch.no_grad()
    def dequant(self, u: torch.Tensor) -> torch.Tensor:
        """（u 空間版 flow 目標；⛔ 9/3 臂 B 實證：目標塌在 up 的 d 維薄片、NF 作弊刷分 ⇒ 判負勿再用）"""
        q = self._grid(self.z_of(u))
        return self.up(q + torch.rand_like(q) - 0.5)

    @torch.no_grad()
    def codes(self, u: torch.Tensor) -> torch.Tensor:
        """每 token 的格點座標（[..., d]）——可讀、可計數（缺課地圖用）。"""
        return self._grid(self.z_of(u))
