"""intent 接法 (ii)：per-token 硬錨 —— 每個 token 拿自己那段的局部錨點，不經全域壓縮。

跟 (i) embed（把整條錨點序列壓成一個全域向量、K 個 token 共享同一份全域摘要）不同，
這裡把 [T_A,2] 的錨點序列切成 K 段（token k 只看第 k 段、連續 T_A/K 個錨點），
段內 flatten 後過同一個共用 Linear+SiLU 得到該 token 專屬的局部條件向量。
「硬」約束的意思就在這裡：每個 token 直接看得到自己該負責的那一小段路的座標，
不會被跟其他 token 共用的全域向量稀釋掉 —— 改動某一段錨點，只影響對應那個 token。

⛔ 本模組不碰 traj 的值本身 —— target_fwd／target_inv 恆等，錨點只走 cond_pertoken 這條路。
"""
import torch
import torch.nn as nn

NAME = "anchor"
TAG = "ita"


class IntentAdapter(nn.Module):
    def __init__(self, t_anchor: int = 32, k_tokens: int = 8,
                 token_dim: int = 8, cond_dim: int = 128):
        super().__init__()
        assert t_anchor % k_tokens == 0, (
            f"T_A 必須是 K 的倍數，拿到 t_anchor={t_anchor}, k_tokens={k_tokens}"
        )
        # token_dim／cond_dim 是三接法共用介面的一部分，本接法（硬局部錨）用不到
        # ——不壓成全域向量，所以沒有 cond_dim 大小的東西要建。
        self.t_anchor = t_anchor
        self.k_tokens = k_tokens
        self.seg_len = t_anchor // k_tokens  # 每個 token 分到的連續錨點數
        self.cond_extra_dim = 0
        self.pertoken_dim = 16
        self.proj = nn.Sequential(
            nn.Linear(self.seg_len * 2, self.pertoken_dim),
            nn.SiLU(),
        )

    def cond_global(self, anchors):
        return None

    def cond_pertoken(self, anchors):
        """[B,T_A,2] -> [B,K,16]：切 K 段（每段 seg_len 個連續錨點）、段內 flatten
        後過共用 Linear+SiLU —— 每個 token 只吃自己那一段，段間互不干擾。"""
        assert anchors.dim() == 3 and anchors.shape[-1] == 2, f"anchors 要 [B,T_A,2]，拿到 {tuple(anchors.shape)}"
        B = anchors.shape[0]
        seg = anchors.reshape(B, self.k_tokens, self.seg_len, 2)
        seg = seg.reshape(B, self.k_tokens, self.seg_len * 2)
        return self.proj(seg)

    def target_fwd(self, traj, anchors):
        return traj

    def target_inv(self, out, anchors):
        return out
