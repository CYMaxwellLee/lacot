"""u → 座標序列的 decoder。

⭐ 兩個地方要用同一顆，所以抽出來共用：
   ① `experiments/exp_decode_probe.py` —— 量「u 裡裝了多少路線資訊」（2026-08-28 已封盤，見索引 Q1）
   ② 主線的 recon encoder 目標 ＋ E_geo（幾何 energy 要靠它把 u 解成可微的座標點）

⛔ 不要在別處另寫一份 —— 這個 repo 已經被「同族東西兩份實作」咬過。
"""
import torch
from torch import nn

from lacot.e_target import CrossAttention, FeedForward


class TrajDecoder(nn.Module):
    """[B, M, d] 的 latent → [B, T, 2] 的座標序列。PerceiverPooler 的鏡像。

    encoder 用 K 個 query 讀 T 個點；decoder 用 T 個 query 讀 M 個 token。積木相同，方向相反。

    🚨 `pos_q` 必須 std=1.0，⛔ 不是 PerceiverPooler 用的 0.02。
       這裡的 query 是【每個輸出點唯一的辨識訊號】，初始差異太小 ⇒ 梯度訊號弱
       ⇒ 訓練會收斂到「條件平均路」（⛔ 不是初始化就塌 —— 2026-08-28 實測 std=0.02
       初始沿時間 std 還有 0.149，離塌掉很遠）。
       ⇒ ⭐ 所以下面那個 assert 只抓得到「完全死透」；真正要抓「吐平均路」得用
         shuffle 對照（把 context 沿 batch 打亂，看誤差跳不跳）—— 見 `ctx_usage_probe`。
    """

    def __init__(self, d_model: int, t_out: int, num_layers: int = 2, num_heads: int = 4,
                 pos_std: float = 1.0):
        super().__init__()
        self.t_out = t_out
        self.pos_q = nn.Parameter(torch.zeros(t_out, d_model))
        nn.init.normal_(self.pos_q, std=pos_std)
        self.layers = nn.ModuleList(
            nn.ModuleDict({
                "q_norm": nn.LayerNorm(d_model),
                "ctx_norm": nn.LayerNorm(d_model),
                "cross_attn": CrossAttention(d_model, num_heads),
                "ff_norm": nn.LayerNorm(d_model),
                "ff": FeedForward(d_model),
            }) for _ in range(num_layers)
        )
        self.out_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 2)      # ⛔ 沒有 sigmoid：座標在正規化空間，不在 [0,1]
        self.check_p = 0.0                     # 抽查機率，呼叫端自己開

    def forward(self, ctx: torch.Tensor) -> torch.Tensor:
        q = self.pos_q.unsqueeze(0).expand(len(ctx), -1, -1)
        for L in self.layers:
            q = q + L["cross_attn"](L["q_norm"](q), L["ctx_norm"](ctx))
            q = q + L["ff"](L["ff_norm"](q))
        out = self.head(self.out_norm(q))
        if self.check_p and self.training and torch.rand(()) < self.check_p:
            spread = out.std(dim=1).mean()
            assert spread > 1e-3, (
                f"⛔ 解碼點沿時間幾乎沒變化（std={spread:.2e}）⇒ 塌成單點。查 pos_q 的初始化")
        return out


@torch.no_grad()
def ctx_usage_probe(dec: TrajDecoder, ctx: torch.Tensor, tgt: torch.Tensor):
    """decoder 到底讀了多少 context —— 把 context 沿 batch 打亂再解一次。

    ⭐ 這是抓「不管給什麼都吐同一條平均路」的尺。那種失敗最毒：
       誤差落在不高不低的地方，然後被讀成「部分資訊在」。
    回傳 (正常內部點 RMSE, 打亂後內部點 RMSE, 差值)。⇒ 差值近 0 ⇒ ⛔ 這次的 RMSE 不能拿來談 u。
    """
    def inner_rmse(p):
        return float((p - tgt).pow(2).sum(-1)[:, 1:-1].mean().sqrt())
    normal = inner_rmse(dec(ctx))
    perm = torch.randperm(len(ctx), device=ctx.device)
    shuffled = inner_rmse(dec(ctx[perm]))
    return normal, shuffled, shuffled - normal
