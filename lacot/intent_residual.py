"""intent 接法 (iii)：residual 雙軌 —— categorical 出輪廓、連續模型只學殘差。

三接法（i embed／ii per-token 錨／iii residual）共用同一份介面：NAME、TAG、
IntentAdapter.{cond_extra_dim, pertoken_dim, cond_global, cond_pertoken,
target_fwd, target_inv}；本檔只實作 (iii)，另兩支由別的 agent 並行寫，互不
import。做法：anchors [B,T_A,2] 先沿弧長上採樣成 [B,T,2] 的「輪廓」（跟真
軌跡同長度、同節奏，只有 T_A 個錨的折線解析度），連續側（flow）只學「真
軌跡 − 輪廓」這份更小更好學的殘差。cond_global 刻意跟接法 (i) 同款 MLP：
同條件、只差目標變換，(iii) vs (i) 的對照才拆得出殘差本身的貢獻。

⚠️ 已知風險：殘差 scale 若學得比輪廓還大＝輪廓被架空、退化成 (i) 直接學絕對軌跡——本模組不管，要由訓練端量 ‖residual‖/‖輪廓‖ 比例來診斷。
⚠️ 成本註記：訓練目標從絕對軌跡換成殘差是不同語言 ⛔ stage 1 decoder 不能直接載入別接法（學絕對軌跡）的凍結 ckpt，要在這個目標下重教殘差語言。
"""
import torch
import torch.nn as nn

NAME = "residual"
TAG = "itr"

def _upsample_arclength(anchors: torch.Tensor, T: int) -> torch.Tensor:
    """anchors [B,T_A,2] → 沿弧長插值到 [B,T,2]（torch 版、batched，對照
    lacot/intent.py 的 anchors_resample）。每個樣本各自算累積弧長、正規化
    到 [0,1] 再對均勻 t_dst 插——⛔ 不按索引均攤（錨點間距不均）。錨是常數
    輸入，全程包在 no_grad。退化（總弧長 < 1e-9）⇒ 全部 T 點 tile 第一錨。
    """
    B, T_A, _ = anchors.shape
    if T_A == 1:
        # 只有一個錨（起訖同格）⇒ 直接 tile，避免下面 clamp(1, T_A-1) 在 T_A=1
        # 時 min>max 反轉、idx0=-1 的風險（見 fix list #5）。
        return anchors.expand(-1, T, -1)
    with torch.no_grad():
        a = anchors.detach()
        seg = torch.linalg.norm(a[:, 1:] - a[:, :-1], dim=-1)               # [B,T_A-1]
        cum = torch.cat([a.new_zeros(B, 1), torch.cumsum(seg, dim=1)], 1)   # [B,T_A]
        total = cum[:, -1]
        degenerate = total < 1e-9
        t_src = cum / total.clamp_min(1e-9).unsqueeze(1)                    # [B,T_A]∈[0,1]
        t_dst = torch.linspace(0.0, 1.0, T, dtype=a.dtype, device=a.device).expand(B, T)
        idx1 = torch.searchsorted(t_src.contiguous(), t_dst.contiguous(), right=True).clamp(1, T_A - 1)
        idx0 = idx1 - 1
        t0 = torch.gather(t_src, 1, idx0)
        t1 = torch.gather(t_src, 1, idx1)
        w = ((t_dst - t0) / (t1 - t0).clamp_min(1e-12)).clamp(0.0, 1.0).unsqueeze(-1)
        A0 = torch.gather(a, 1, idx0.unsqueeze(-1).expand(-1, -1, 2))
        A1 = torch.gather(a, 1, idx1.unsqueeze(-1).expand(-1, -1, 2))
        out = A0 + w * (A1 - A0)
        out = torch.where(degenerate.view(B, 1, 1), a[:, :1, :].expand(-1, T, -1), out)
    return out

class IntentAdapter(nn.Module):
    """接法 (iii)：cond 給錨的全域摘要，target 變換是「減／加輪廓」。"""

    def __init__(self, t_anchor: int = 32, k_tokens: int = 8,
                 token_dim: int = 8, cond_dim: int = 128):
        super().__init__()
        # k_tokens/token_dim/cond_dim：共用簽名要求接的參數，本接法用不到。
        self.cond_extra_dim = 64   # 殘差分佈 depends on 錨形狀，flow 要看得到錨
        self.pertoken_dim = 0      # 沒有 per-token 分支
        self.mlp = nn.Sequential(
            nn.Linear(t_anchor * 2, 128), nn.SiLU(),
            nn.Linear(128, self.cond_extra_dim),
        )

    def cond_global(self, anchors: torch.Tensor) -> torch.Tensor:
        """[B,T_A,2] → [B,64]：flatten 過兩層 MLP（跟接法 (i) 同款）。"""
        assert anchors.dim() == 3 and anchors.shape[-1] == 2, f"anchors 要 [B,T_A,2]，拿到 {tuple(anchors.shape)}"
        return self.mlp(anchors.reshape(anchors.shape[0], -1))

    def cond_pertoken(self, anchors: torch.Tensor):
        """(iii) 沒有 per-token 分支，恆回 None。"""
        return None

    def target_fwd(self, traj: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
        """traj [B,T,2] → 殘差 = traj − 輪廓；T 讀 traj.shape[1]，⛔ 不寫死。"""
        return traj - _upsample_arclength(anchors, traj.shape[1]).to(traj)

    def target_inv(self, out: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
        """殘差 [B,T,2] → 還原軌跡 = out + 輪廓；T 讀 out.shape[1]。"""
        return out + _upsample_arclength(anchors, out.shape[1]).to(out)
