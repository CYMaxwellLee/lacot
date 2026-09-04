"""intent 接法 (i)：全域 embed —— 整條路線的錨點序列，壓成一個向量併進 condvec 尾巴。

背景（見 lacot/intent.py 開頭）：intent = 一條路線的錨點序列，沿弧長重採樣成固定
[T_A,2]（T_A=32）。餵給下層連續模型有三種接法，本檔只做 (i)：

  (i)   embed（本檔）  anchors 整段 flatten 過 MLP → 一個全域向量，接進 condvec。
                        這是「軟約束」—— flow 看到的只是一團被壓縮過的資訊，anchors
                        長什麼樣子、模型要不要理、理多少，全部留給 loss 自己學，
                        沒有任何一步強迫輸出序列的某個點對齊某個錨點。
  (ii)  per-token 硬錨（別的 agent 實作）逐 token 對齊，是硬約束，對照組。
  (iii) residual（別的 agent 實作）。

⛔ 本檔只管 (i) 這個接法本身：不碰 anchors 怎麼來（lacot/intent.py 的事）、
不碰另外兩接法、不碰 condvec 主檔怎麼併（呼叫端接線）。
"""
import torch
import torch.nn as nn

NAME = "embed"
TAG = "ite"          # 檔名段


class IntentAdapter(nn.Module):
    """(i) embed：cond_global 把 anchors 壓成一個 64 維全域向量；不吐 per-token
    條件；target_fwd/target_inv 恆等 —— 這個接法只加一路軟條件，不動訓練目標。"""

    def __init__(self, t_anchor: int = 32, k_tokens: int = 8,
                 token_dim: int = 8, cond_dim: int = 128):
        super().__init__()
        self.cond_extra_dim = 64
        self.pertoken_dim = 0
        self.mlp = nn.Sequential(
            nn.Linear(t_anchor * 2, 128),
            nn.SiLU(),
            nn.Linear(128, self.cond_extra_dim),
        )

    def cond_global(self, anchors: torch.Tensor) -> torch.Tensor:
        """anchors [B,T_A,2] float32 -> [B,64]：flatten 後過兩層 MLP。"""
        assert anchors.dim() == 3 and anchors.shape[-1] == 2, f"anchors 要 [B,T_A,2]，拿到 {tuple(anchors.shape)}"
        return self.mlp(anchors.reshape(anchors.shape[0], -1))

    def cond_pertoken(self, anchors: torch.Tensor):
        """(i) 不吐 per-token 條件，恆回 None（對照組 (ii) 才會用到這個）。"""
        return None

    def target_fwd(self, traj: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
        """恆等：(i) 不碰訓練目標，anchors 只走 cond_global 那條軟路。"""
        return traj

    def target_inv(self, out: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
        """target_fwd 的反變換，(i) 同樣恆等。"""
        return out
