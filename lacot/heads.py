"""Output heads: discretized action head, plus future and value stubs.

Design doc reference: WPM-Design-0803.md S4.3.

- `DiscretizedActionHead` is the only head trained at the GCBC floor (depth 0,
  milestone "GCBC baseline"). It turns one hidden vector into a chunk of
  actions, by predicting a category (a "bin") for every action dimension at
  every chunk step, instead of predicting one continuous number directly. A
  categorical head can represent several separate, likely action values at
  once (a mixture); a single continuous (Gaussian) head would instead average
  them into one, weaker value — bad for rare, decisive actions such as a
  gripper toggle (design doc S4.3).
- `FutureHead` and `ValueHead` are stubs: their forward pass produces
  correctly-shaped output, so later milestones can wire in real losses
  without changing any calling code. Neither is trained nor called during
  depth-0 training.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

# The pre-declared execution-convention menu (CLAUDE.md, "Evaluation rules";
# design doc S6 / S14 decision log). `None` means greedy (argmax, no
# sampling); a number is the softmax temperature.
DECODE_MODES = {"greedy": None, "t0.5": 0.5, "t1.0": 1.0}


class DiscretizedActionHead(nn.Module):
    """Predicts an action chunk as independent per-dimension, per-step bins.

    ⛔ NOT for continuous control — use `ContinuousActionHead` instead. Measured
    2026-08-23 on pointmaze: every binned variant scores WORSE than predicting the
    dataset mean, even with capacity matched to the MLP head. Kept for discrete or
    genuinely multi-modal action spaces. See that class's docstring for the numbers.

    Each of the `action_dim` action dimensions, at each of the `chunk_len`
    steps in a chunk, gets its own categorical distribution over `num_bins`
    bins spaced uniformly across [-1, 1] (dataset actions are assumed to
    already be scaled to that range).
    """

    def __init__(
        self,
        d_model: int,
        action_dim: int,
        chunk_len: int,
        num_bins: int = 256,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.chunk_len = chunk_len
        self.num_bins = num_bins
        self.bin_width = 2.0 / num_bins
        self.proj = nn.Linear(d_model, chunk_len * action_dim * num_bins)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: (..., d_model) -> logits: (..., chunk_len, action_dim, num_bins)."""
        out = self.proj(h)
        return out.reshape(*h.shape[:-1], self.chunk_len, self.action_dim, self.num_bins)

    def encode(self, actions: torch.Tensor) -> torch.Tensor:
        """Continuous actions in [-1, 1] -> bin index per value (same shape, long)."""
        idx = torch.floor((actions + 1.0) / self.bin_width)
        return idx.clamp(0, self.num_bins - 1).long()

    def decode_bins(self, bin_idx: torch.Tensor) -> torch.Tensor:
        """Bin index -> the bin's center value (the inverse of `encode`, up to quantization)."""
        return -1.0 + (bin_idx.float() + 0.5) * self.bin_width

    def nll(self, logits: torch.Tensor, target_actions: torch.Tensor) -> torch.Tensor:
        """Per-example negative log-likelihood, averaged over chunk step and action dim.

        logits: (B, chunk_len, action_dim, num_bins); target_actions: (B, chunk_len, action_dim).
        Returns: (B,).
        """
        target_bins = self.encode(target_actions)
        per_position = F.cross_entropy(
            logits.reshape(-1, self.num_bins), target_bins.reshape(-1), reduction="none"
        )
        per_position = per_position.reshape(target_actions.shape)  # (B, chunk_len, action_dim)
        return per_position.mean(dim=(1, 2))

    def loss(
        self, logits: torch.Tensor, target_actions: torch.Tensor, weight: torch.Tensor
    ) -> torch.Tensor:
        """The proposal (BC) loss L_prop = mean(weight * per-example NLL) (design doc S5.3)."""
        return (weight * self.nll(logits, target_actions)).mean()

    @torch.no_grad()
    def accuracy(self, logits: torch.Tensor, target_actions: torch.Tensor) -> torch.Tensor:
        """Fraction of chunk steps where the top bin matches the target bin, per action dim.

        Returns: (action_dim,), averaged over batch and chunk step.
        """
        target_bins = self.encode(target_actions)
        pred_bins = logits.argmax(dim=-1)
        correct = (pred_bins == target_bins).float()  # (B, chunk_len, action_dim)
        return correct.mean(dim=(0, 1))

    @torch.no_grad()
    def entropy(self, logits: torch.Tensor) -> torch.Tensor:
        """Mean categorical entropy per action dimension (design doc S4.3, needed for Claim 4).

        Returns: (action_dim,), averaged over batch and chunk step, in nats.
        """
        probs = F.softmax(logits, dim=-1)
        ent = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1)  # (B, chunk_len, action_dim)
        return ent.mean(dim=(0, 1))

    @torch.no_grad()
    def decode(
        self,
        logits: torch.Tensor,
        mode: str = "greedy",
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """logits -> a concrete action chunk, per the pre-declared decode menu.

        mode: one of DECODE_MODES ("greedy", "t0.5", "t1.0"). Sampling modes
        use `generator` if given, so callers can get bit-exact, reproducible
        draws from their own seeded RNG (random-number generator) stream, as
        the evaluation rules require (CLAUDE.md, "Evaluation rules").
        """
        if mode not in DECODE_MODES:
            raise ValueError(f"unknown decode mode {mode!r}, expected one of {list(DECODE_MODES)}")
        temperature = DECODE_MODES[mode]

        if temperature is None:
            bin_idx = logits.argmax(dim=-1)
        else:
            probs = F.softmax(logits / temperature, dim=-1)
            flat_probs = probs.reshape(-1, self.num_bins)
            flat_idx = torch.multinomial(flat_probs, 1, generator=generator).squeeze(-1)
            bin_idx = flat_idx.reshape(probs.shape[:-1])

        return self.decode_bins(bin_idx)


class FutureHead(nn.Module):
    """Stub: predicts the future-slice and progress tokens from one hidden vector.

    Design doc S4.3 / S4.4: at full depth, this reads out K_z = 4 tokens (3
    future-slice embeddings, matching the frozen image-encoder EMA's output
    size, plus 1 scalar progress value) from the hidden state produced right
    after a tried action is appended to context. Not trained or called during
    depth-0 (GCBC-floor) training; this class only guarantees the shapes the
    later training loop will need.
    """

    def __init__(self, d_model: int, num_slices: int = 3):
        super().__init__()
        self.num_slices = num_slices
        self.slice_proj = nn.Linear(d_model, num_slices * d_model)
        self.progress_proj = nn.Linear(d_model, 1)
        self.d_model = d_model

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """h: (..., d_model) -> (slices: (..., num_slices, d_model), progress: (..., 1))."""
        slices = self.slice_proj(h).reshape(*h.shape[:-1], self.num_slices, self.d_model)
        progress = self.progress_proj(h)
        return slices, progress


class ValueHead(nn.Module):
    """Stub: predicts a scalar value from one hidden vector.

    Design doc S4.3 / S5.2: trained only as a distillation target against the
    frozen value function V*; never used to score or rank actions at
    inference (CLAUDE.md hard constraint R5). The same module is reused
    later both for the value read from h_0 and for the future value/progress
    read from z tokens — both are just "one hidden vector in, one scalar
    out." Not trained or called during depth-0 training.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.proj = nn.Linear(d_model, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: (..., d_model) -> value: (..., 1)."""
        return self.proj(h)


class ContinuousActionHead(nn.Module):
    """Predicts an action chunk directly as continuous values (MLP + MSE).

    This is the head the state track actually uses. `DiscretizedActionHead` was
    tried first and measured — on `pointmaze-medium-navigate`, decoding to
    continuous actions and comparing on the same metric (MSE after decode):

        head                          MSE @5k steps
        ------------------------------------------
        this one (MLP + MSE)               0.375
        Discretized  32 bins + same MLP    0.845
        Discretized 256 bins + same MLP    0.903
        Discretized 256 bins, proj only    1.286
        "predict the dataset mean"         0.494   <- baseline

    Every discretized variant lands ABOVE the do-nothing baseline even with the
    capacity matched, so the loss is the binning itself: cross-entropy gives no
    credit for landing in a neighbouring bin, which is exactly the signal a
    continuous control task needs. Bin count matters a little (32 beats 256
    consistently) but is not the main term. See docs/FINDINGS-2026-08-23.md.

    ⚠️ 2026-08-23: 這個 head 必須同時吃 `cond` 與 `u`，⛔ 不能只吃 `u`。
    08-22 的實驗腳本就是這樣寫的（`sota_mlp(COND + DIM, ...)`，forward 吃 `(cond, u)`），
    而本體的 `LaCoTActor` 只餵 `u` —— 這個落差讓 K=4 的 ORACLE 從 08-22 的 100%
    掉到 12%。理由：`u` 是【未來軌跡】的壓縮表徵，而動作要回答的是「從【現在這個位置】
    往哪走」。K 小的時候 `u` 裡的位置資訊被壓掉了，head 沒有 `cond` 就推不出來。
    ⇒ `cond`（精確的 s,g）與 `u`（規劃）是互補的，兩個都要。
    """

    def __init__(
        self,
        d_model: int,          # = cond_dim + k*d_model（兩者 concat 後的寬度）
        action_dim: int,
        chunk_len: int,
        hidden: int = 512,
        n_layers: int = 3,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.chunk_len = chunk_len
        layers: list[nn.Module] = []
        prev = d_model
        for _ in range(n_layers):
            lin = nn.Linear(prev, hidden)
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)
            layers += [lin, nn.GELU(), nn.LayerNorm(hidden)]
            prev = hidden
        out = nn.Linear(prev, chunk_len * action_dim)
        nn.init.xavier_uniform_(out.weight)
        nn.init.zeros_(out.bias)
        self.net = nn.Sequential(*layers, out)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: (B, d_model) -> action chunk (B, chunk_len, action_dim)."""
        return self.net(h).reshape(-1, self.chunk_len, self.action_dim)

    def nll(self, pred: torch.Tensor, target_actions: torch.Tensor) -> torch.Tensor:
        """Per-example squared error, averaged over chunk step and action dim.

        Named `nll` so it is drop-in compatible with `DiscretizedActionHead.nll`
        (a Gaussian with fixed variance has exactly this as its negative
        log-likelihood, up to a constant).
        """
        return (pred - target_actions).pow(2).mean(dim=(1, 2))

    def loss(
        self, pred: torch.Tensor, target_actions: torch.Tensor, weight: torch.Tensor
    ) -> torch.Tensor:
        return (weight * self.nll(pred, target_actions)).mean()

    @torch.no_grad()
    def act(self, h: torch.Tensor) -> torch.Tensor:
        """Inference: clamp to the environment's action range."""
        return self.forward(h).clamp(-1.0, 1.0)


class TokenActionHead(nn.Module):
    """Action head that keeps `u`'s token structure instead of flattening it.

    ⚠️ Why this exists (measured 2026-08-23). The previous head did

        head(cat([cond, u.flatten()]))        # 256 + 4*256 -> one 1280-vector

    which destroys structure twice over: `u` is K learned latent tokens (each a
    different facet of the trajectory, produced by PerceiverPooler), and flatten
    turns them into 1024 anonymous numbers; then `cat` mixes them with `cond` in
    the very first Linear, after which nothing can tell the two apart.

    The tell was that scaling the MLP did not help at all — feeding it the TRUE
    e_target beat cond-only by roughly the same margin no matter the size:

        head 3 layers x 512   (1.2M params)   gain +12.2%
        head 4 layers x 1024  (4.5M params)   gain +11.9%
        head 5 layers x 2048  (19.4M params)  gain +10.7%

    Capacity was never the bottleneck; the information was already scrambled at
    the door. Chain-of-Goals (arXiv 2602.03389) keeps state / goal / each latent
    subgoal / action as separate tokens and lets them talk via mixing layers —
    79% vs GCBC's 1% on pointmaze-giant.

    Here `cond` becomes one token and each of the K latents stays its own token;
    self-attention lets them exchange what they need while keeping their identity.

    ⚠️ 第一版輸給 concat（增益 4.0~8.4% vs concat 的 12.2%），但那個比較不算數：
    它同時背著三個【不公平】而不是設計上的差異。所以下面每一項都做成開關，
    可以一項一項加上去看各自貢獻（主人 2026-08-23：「我想看 ablation，
    不要只單純全部加」）：

      deep_readout  讀出用 3 層 MLP —— 第一版只有 Linear(256->8)，
                    而 concat 版輸出前有 3 層。讀出容量差太多。
      wide          內部寬度 —— 第一版 d_model 只有 256（跟著 u 的 token 寬度走），
                    concat 版內部是 512~2048。
      u_proj        給 u 一個投影層 —— 第一版 cond 有 cond_proj，u 卻是直接加
                    embedding 就進去。⚠️ 這個特別諷刺：這個 head 本來就是為了修
                    「cond 被服侍、u 是生的」而做的，結果在新 head 裡又犯一次。
                    關掉時走【零填補】而不是投影 —— 見 __init__ 裡的註解，
                    為的是讓 wide 跟 u_proj 兩個開關真的獨立。
      readout_mode  從哪讀出動作：
                    "cond"  只讀 cond token（第一版，照抄 BERT 的 [CLS] 習慣，
                            ⚠️ 沒有任何理由）——  u 的資訊得先被 attention 搬到
                            cond token 上才影響得了動作，中間多一道關卡
                    "pool"  平均全部 token
                    "query" 專用的 action query token
    """

    def __init__(
        self,
        cond_dim: int,
        d_model: int,
        k: int,
        action_dim: int,
        chunk_len: int,
        n_layers: int = 2,
        n_heads: int = 4,
        deep_readout: bool = False,
        wide: int = 0,
        u_proj: bool = False,
        readout_mode: str = "cond",
    ):
        super().__init__()
        self.k, self.action_dim, self.chunk_len = k, action_dim, chunk_len
        self.readout_mode = readout_mode
        w = wide or d_model
        self.d_model = w
        self.cond_proj = nn.Linear(cond_dim, w)
        # ⚠️ wide 跟 u_proj 會黏在一起：w != d_model 時 u 的維度對不上，
        #    「一定要有個東西」把它抬上去 —— 那樣 A2 一開就自動帶著 A3，
        #    ablation 就分不出是誰的功勞。所以 u_proj=False 時用【零填補】
        #    （不含參數、不學習）當對照，兩個開關才真的獨立。
        self.u_in = d_model
        self.u_proj = nn.Linear(d_model, w) if u_proj else None
        if self.u_proj is None and w < d_model:
            raise ValueError(f"wide={w} 比 u 的維度 {d_model} 還小，零填補抬不上去")
        self.type_emb = nn.Parameter(torch.zeros(2, w))
        self.pos_emb = nn.Parameter(torch.zeros(k, w))
        nn.init.trunc_normal_(self.type_emb, std=0.02)
        nn.init.trunc_normal_(self.pos_emb, std=0.02)
        self.query = nn.Parameter(torch.zeros(1, 1, w)) if readout_mode == "query" else None
        if self.query is not None:
            nn.init.trunc_normal_(self.query, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=w, nhead=n_heads, dim_feedforward=4 * w,
            batch_first=True, norm_first=True, dropout=0.0,
        )
        self.mixer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.out_norm = nn.LayerNorm(w)
        if deep_readout:
            L, p = [], w
            for _ in range(3):
                lin = nn.Linear(p, w); nn.init.xavier_uniform_(lin.weight); nn.init.zeros_(lin.bias)
                L += [lin, nn.GELU(), nn.LayerNorm(w)]
                p = w
            out = nn.Linear(p, chunk_len * action_dim)
            nn.init.xavier_uniform_(out.weight); nn.init.zeros_(out.bias)
            self.readout = nn.Sequential(*L, out)
        else:
            self.readout = nn.Linear(w, chunk_len * action_dim)
            nn.init.xavier_uniform_(self.readout.weight)
            nn.init.zeros_(self.readout.bias)

    def forward(self, cond: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """cond (B, cond_dim), u (B, K, d_model_in) -> action chunk (B, chunk_len, action_dim)."""
        b = cond.shape[0]
        c = self.cond_proj(cond).unsqueeze(1) + self.type_emb[0]
        if self.u_proj is not None:
            z = self.u_proj(u)
        elif self.d_model > self.u_in:
            z = F.pad(u, (0, self.d_model - self.u_in))     # 零填補，不含參數
        else:
            z = u
        z = z + self.type_emb[1] + self.pos_emb
        toks = [c, z]
        if self.query is not None:
            toks = [self.query.expand(b, -1, -1)] + toks
        h = self.mixer(torch.cat(toks, dim=1))
        if self.readout_mode == "pool":
            pooled = h.mean(dim=1)
        elif self.readout_mode == "query":
            pooled = h[:, 0]                      # 專用 query token
        else:
            pooled = h[:, 0]                      # cond token
        return self.readout(self.out_norm(pooled)).reshape(b, self.chunk_len, self.action_dim)

    def nll(self, pred: torch.Tensor, target_actions: torch.Tensor) -> torch.Tensor:
        """Same signature as the other heads so callers stay interchangeable."""
        return (pred - target_actions).pow(2).mean(dim=(1, 2))

    @torch.no_grad()
    def act(self, cond: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return self.forward(cond, u).clamp(-1.0, 1.0)

class MixerActionHead(nn.Module):
    """跟 TokenActionHead 同介面，但跨 token 的混合改用 MLP-Mixer 而不是 self-attention。

    為什麼要這個（2026-08-23 實測出來的動機）：
      concat MLP 的增益 +8.0%，token/self-attention 只有 +4.6%，四組對照同向。
      拆下來的原因是 attention 的跨 token 混合是【加權平均】—— softmax 權重非負、
      和為一 ⇒ 做不出減法；而 concat MLP 的第一層是全部維度的自由線性組合。
      再加上 attention 的投影權重對每顆 token 共用，token 只有 4 顆時沒東西好 generalize。

    Mixer 剛好卡在中間：跨 token 的混合是一層【普通線性層】（自由加減，不受
    加權平均限制），但仍保留「每顆 token 是獨立單位」的結構、不像 concat 那樣攤平。
    Chain-of-Goals（arXiv 2602.03389）用的就是 Mixer 不是 attention。

    ⚠️ Mixer 的 token-mixing 層綁死 token 數（線性層的輸入維度就是 N），
       所以 ⛔ 不能像 attention 那樣吃變長序列。這裡 N = 1 + K 固定，沒問題。
    """

    def __init__(
        self,
        cond_dim: int,
        d_model: int,
        k: int,
        action_dim: int,
        chunk_len: int,
        n_layers: int = 2,
        tok_hidden: int = 0,
        deep_readout: bool = False,
        wide: int = 0,
        u_proj: bool = False,
        readout_mode: str = "pool",
    ):
        super().__init__()
        self.k, self.action_dim, self.chunk_len = k, action_dim, chunk_len
        self.readout_mode = readout_mode
        w = wide or d_model
        self.d_model, self.u_in = w, d_model
        n_tok = 1 + k + (1 if readout_mode == "query" else 0)
        th = tok_hidden or 4 * n_tok
        self.cond_proj = nn.Linear(cond_dim, w)
        self.u_proj = nn.Linear(d_model, w) if u_proj else None
        if self.u_proj is None and w < d_model:
            raise ValueError(f"wide={w} 比 u 的維度 {d_model} 還小，零填補抬不上去")
        self.type_emb = nn.Parameter(torch.zeros(2, w))
        self.pos_emb = nn.Parameter(torch.zeros(k, w))
        nn.init.trunc_normal_(self.type_emb, std=0.02)
        nn.init.trunc_normal_(self.pos_emb, std=0.02)
        self.query = nn.Parameter(torch.zeros(1, 1, w)) if readout_mode == "query" else None
        if self.query is not None:
            nn.init.trunc_normal_(self.query, std=0.02)

        self.norm_t = nn.ModuleList(nn.LayerNorm(w) for _ in range(n_layers))
        self.norm_c = nn.ModuleList(nn.LayerNorm(w) for _ in range(n_layers))
        self.mix_t = nn.ModuleList(
            nn.Sequential(nn.Linear(n_tok, th), nn.GELU(), nn.Linear(th, n_tok))
            for _ in range(n_layers)
        )
        self.mix_c = nn.ModuleList(
            nn.Sequential(nn.Linear(w, 4 * w), nn.GELU(), nn.Linear(4 * w, w))
            for _ in range(n_layers)
        )
        self.out_norm = nn.LayerNorm(w)
        if deep_readout:
            L = []
            for _ in range(3):
                lin = nn.Linear(w, w); nn.init.xavier_uniform_(lin.weight); nn.init.zeros_(lin.bias)
                L += [lin, nn.GELU(), nn.LayerNorm(w)]
            out = nn.Linear(w, chunk_len * action_dim)
            nn.init.xavier_uniform_(out.weight); nn.init.zeros_(out.bias)
            self.readout = nn.Sequential(*L, out)
        else:
            self.readout = nn.Linear(w, chunk_len * action_dim)
            nn.init.xavier_uniform_(self.readout.weight)
            nn.init.zeros_(self.readout.bias)

    def forward(self, cond: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        b = cond.shape[0]
        c = self.cond_proj(cond).unsqueeze(1) + self.type_emb[0]
        if self.u_proj is not None:
            z = self.u_proj(u)
        elif self.d_model > self.u_in:
            z = F.pad(u, (0, self.d_model - self.u_in))
        else:
            z = u
        z = z + self.type_emb[1] + self.pos_emb
        toks = [c, z]
        if self.query is not None:
            toks = [self.query.expand(b, -1, -1)] + toks
        h = torch.cat(toks, dim=1)                                   # (B, N, w)
        for nt, nc, mt, mc in zip(self.norm_t, self.norm_c, self.mix_t, self.mix_c):
            h = h + mt(nt(h).transpose(1, 2)).transpose(1, 2)        # 跨 token（自由線性）
            h = h + mc(nc(h))                                        # 跨 channel
        pooled = h.mean(dim=1) if self.readout_mode == "pool" else h[:, 0]
        return self.readout(self.out_norm(pooled)).reshape(b, self.chunk_len, self.action_dim)

    def nll(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """跟其他 head 同簽名，讓呼叫端可以互換。"""
        return (pred - target).pow(2).mean(dim=(-1, -2))
