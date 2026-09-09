"""experiments/gen_sft/model.py —— 小 transformer decoder：(s0, 路標 xy 序列) 一次
自回歸生成整段步法字串（code 序列）。

條件當 prefix token：s0(29,) 過線性投影成 1 個 token；每個路標 xy(2,) 各過線性
投影成 1 個 token（可變長 M，pad+mask）。輸出接在 BOS 之後，vocab=K(32)+EOS，
teacher forcing CE 訓練。整個序列（prefix + BOS + 輸出）套同一個 causal mask
（純 GPT 風格：prefix 內部也只能看回過去，不特別做 bidirectional prefix-LM —
路標本身沿路徑排序，causal 順序不損失資訊，用純 causal 换取實作簡單、少 bug 面）。

⛔ 新檔，不改動任何既有檔案。
"""
import math

import numpy as np
import torch
import torch.nn as nn

K_CODES = 32
EOS_ID = K_CODES              # 32 —— 只當 target 用
BOS_ID = K_CODES + 1          # 33 —— 只當輸入用（序列第一個輸出位置的種子）
INPUT_VOCAB = K_CODES + 2     # 0..31 code, 32(=EOS，理論上輸入不會餵到，保留為安全邊界), 33=BOS
TARGET_VOCAB = K_CODES + 1    # 0..31 code, 32=EOS


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, attn_mask):
        # x: (B,T,D); attn_mask: (B,1,T,T) additive (0 或 -inf)
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B,H,T,hd)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = att + attn_mask
        att = att.softmax(dim=-1)
        att = self.drop(att)
        out = att @ v
        out = out.transpose(1, 2).reshape(B, T, D)
        return self.proj(out)


class Block(nn.Module):
    def __init__(self, d_model, n_heads, mlp_mult=4, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_mult * d_model), nn.GELU(),
            nn.Linear(mlp_mult * d_model, d_model), nn.Dropout(dropout),
        )

    def forward(self, x, attn_mask):
        x = x + self.attn(self.ln1(x), attn_mask)
        x = x + self.mlp(self.ln2(x))
        return x


class GenSFT(nn.Module):
    def __init__(self, d_model=128, n_layers=3, n_heads=4, dropout=0.1, max_len=96):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.s0_proj = nn.Linear(29, d_model)
        self.wp_proj = nn.Linear(2, d_model)
        self.tok_emb = nn.Embedding(INPUT_VOCAB, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.type_emb = nn.Embedding(3, d_model)  # 0=s0, 1=waypoint, 2=output(BOS/code)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([Block(d_model, n_heads, dropout=dropout) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, TARGET_VOCAB)

    def _embed(self, s0_obs, wp_xy, wp_mask, out_tokens):
        B, Mmax, _ = wp_xy.shape
        Lmax = out_tokens.shape[1]
        s0_e = self.s0_proj(s0_obs).unsqueeze(1)
        wp_e = self.wp_proj(wp_xy)
        out_e = self.tok_emb(out_tokens)
        x = torch.cat([s0_e, wp_e, out_e], dim=1)
        T = x.shape[1]
        assert T <= self.max_len, f"⛔ 序列長度 {T} 超過 max_len={self.max_len}"
        pos_ids = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)
        type_ids = torch.cat([
            torch.zeros(B, 1, dtype=torch.long, device=x.device),
            torch.ones(B, Mmax, dtype=torch.long, device=x.device),
            torch.full((B, Lmax), 2, dtype=torch.long, device=x.device),
        ], dim=1)
        x = x + self.pos_emb(pos_ids) + self.type_emb(type_ids)
        x = self.drop(x)
        valid = torch.cat([
            torch.ones(B, 1, dtype=torch.bool, device=x.device),
            wp_mask,
            torch.ones(B, Lmax, dtype=torch.bool, device=x.device),
        ], dim=1)
        return x, valid, Mmax, Lmax

    def _run_blocks(self, x, valid):
        B, T, _ = x.shape
        key_mask = torch.zeros(B, 1, 1, T, device=x.device)
        key_mask.masked_fill_(~valid.view(B, 1, 1, T), float("-inf"))
        causal = torch.triu(torch.full((T, T), float("-inf"), device=x.device), diagonal=1).view(1, 1, T, T)
        attn_mask = causal + key_mask
        for blk in self.blocks:
            x = blk(x, attn_mask)
        return self.ln_f(x)

    def forward(self, s0_obs, wp_xy, wp_mask, out_tokens):
        """teacher forcing：out_tokens=(B,Lmax) 輸入端 token(BOS+code，右側 pad 任意值皆可，
        pad 位置的 loss 由呼叫端用 ignore_index 蓋掉)。回傳 logits (B,Lmax,TARGET_VOCAB)，
        位置 i 預測「接在 out_tokens[i] 之後的下一個 token」。
        """
        x, valid, Mmax, Lmax = self._embed(s0_obs, wp_xy, wp_mask, out_tokens)
        x = self._run_blocks(x, valid)
        logits_full = self.head(x)
        return logits_full[:, 1 + Mmax:, :]

    @torch.no_grad()
    def generate(self, s0_obs, wp_xy, wp_mask, max_new_tokens=60, temperature=0.0, rng=None):
        """自回歸生成。temperature=0 -> greedy；>0 -> 依溫度取樣（用傳入的 torch.Generator
        `rng` 控制隨機性，呼叫端必須自己固定 seed）。逐 batch 各自在遇到 EOS 時停止累積
        （用 done mask），回傳 list[np.ndarray] 每個樣本自己的 code 序列（不含 EOS/BOS）。
        """
        self.eval()
        device = s0_obs.device
        B = s0_obs.shape[0]
        Mmax = wp_xy.shape[1]
        out_tokens = torch.full((B, 1), BOS_ID, dtype=torch.long, device=device)
        done = torch.zeros(B, dtype=torch.bool, device=device)
        generated = [[] for _ in range(B)]
        for _step in range(max_new_tokens):
            x, valid, Mmax_, Lmax = self._embed(s0_obs, wp_xy, wp_mask, out_tokens)
            x = self._run_blocks(x, valid)
            logits_full = self.head(x)
            next_logits = logits_full[:, -1, :]  # (B,TARGET_VOCAB) -- 最後一個輸出位置預測下一個
            if temperature <= 0:
                next_id = next_logits.argmax(dim=-1)
            else:
                probs = torch.softmax(next_logits / temperature, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1, generator=rng).squeeze(-1)
            for b in range(B):
                if done[b]:
                    continue
                if int(next_id[b].item()) == EOS_ID:
                    done[b] = True
                else:
                    generated[b].append(int(next_id[b].item()))
            if bool(done.all()):
                break
            # 下一輪輸入：EOS 之後的 token 對已完成序列沒有意義，塞 0（不影響，done 樣本不再讀取輸出)
            feed_id = torch.where(next_id == EOS_ID, torch.zeros_like(next_id), next_id)
            out_tokens = torch.cat([out_tokens, feed_id.unsqueeze(1)], dim=1)
        # 回傳每個樣本 (codes, ended_with_eos)：ended_with_eos=True 表示在
        # max_new_tokens 內真的吐出 EOS 停下來；False 表示撞到 max_new_tokens 都
        # 沒吐 EOS（codes 長度=max_new_tokens，是被截斷的，不是模型判斷「講完了」）。
        done_list = done.tolist()
        return [
            (np.asarray(generated[b], dtype=np.int64), bool(done_list[b]))
            for b in range(B)
        ]
