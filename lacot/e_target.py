"""e_target generator (target side) for M4 -- built block by block.

Block 1 (this file, for now): `PerceiverPooler` -- pool a variable-length
sequence of per-frame features [B, T, d_in] into a FIXED-size latent
e_target [B, K, d_model] via learned-query cross-attention (Perceiver Resampler
/ Set-Transformer PMA / DETR-style queries). It is ORDER-AWARE (temporal
positional embedding), because e_target compresses a future STATE TRAJECTORY,
not an unordered set.

The per-frame encoder is the SOTA-aligned `ImpalaSmall`
(`wpm.train_value_official.ImpalaSmall`, a faithful port of OGBench's
`impala_small`); it runs upstream and produces the [B, T, d_in] frame features
this module pools. Later blocks added here: a frame decoder
(e_target -> reconstructed future frames) and the full encoder+decoder wired for
reconstruction pretraining (then the encoder is frozen). Smoke test:
`wpm/models/e_target_smoke.py`.
"""
from __future__ import annotations

import math

import torch
from torch import nn


class CrossAttention(nn.Module):
    """Multi-head cross-attention: `query` attends to a `context` key/value seq."""

    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model {d_model} not divisible by num_heads {num_heads}")
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.kv_proj = nn.Linear(d_model, 2 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # query [B, Q, D], context [B, T, D], key_padding_mask [B, T] (True == PAD).
        B, Q, D = query.shape
        T = context.shape[1]
        q = self.q_proj(query).view(B, Q, self.num_heads, self.d_head).transpose(1, 2)
        kv = self.kv_proj(context).view(B, T, 2, self.num_heads, self.d_head)
        k = kv[:, :, 0].transpose(1, 2)  # [B, H, T, d_head]
        v = kv[:, :, 1].transpose(1, 2)  # [B, H, T, d_head]
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)  # [B, H, Q, T]
        if key_padding_mask is not None:
            attn = attn.masked_fill(key_padding_mask[:, None, None, :], float("-inf"))
        attn = attn.softmax(dim=-1)
        out = attn @ v  # [B, H, Q, d_head]
        out = out.transpose(1, 2).reshape(B, Q, D)
        return self.out_proj(out)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, mult: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model * mult),
            nn.GELU(),
            nn.Linear(d_model * mult, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PerceiverPooler(nn.Module):
    """Pool [B, T, d_in] frame features into a fixed [B, K, d_model] e_target.

    `K` learned queries cross-attend to temporally-positioned frame features over
    `num_layers` pre-norm (cross-attention + feed-forward) blocks. Handles a
    variable number of frames `T` and an optional key-padding mask for
    ragged/padded batches. Output is the fixed-size latent trajectory summary.
    """

    def __init__(
        self,
        d_in: int,
        d_model: int,
        k: int,
        num_layers: int = 2,
        num_heads: int = 4,
        max_len: int = 512,
    ):
        super().__init__()
        self.k = k
        self.d_model = d_model
        self.max_len = max_len
        self.in_proj = nn.Linear(d_in, d_model) if d_in != d_model else nn.Identity()
        self.pos_emb = nn.Parameter(torch.zeros(max_len, d_model))
        self.queries = nn.Parameter(torch.zeros(k, d_model))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)
        nn.init.trunc_normal_(self.queries, std=0.02)
        self.layers = nn.ModuleList(
            nn.ModuleDict(
                {
                    "q_norm": nn.LayerNorm(d_model),
                    "ctx_norm": nn.LayerNorm(d_model),
                    "cross_attn": CrossAttention(d_model, num_heads),
                    "ff_norm": nn.LayerNorm(d_model),
                    "ff": FeedForward(d_model),
                }
            )
            for _ in range(num_layers)
        )
        self.out_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        frame_feats: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # frame_feats [B, T, d_in]; key_padding_mask [B, T] (True == PAD) -> e_target [B, K, d_model].
        B, T, _ = frame_feats.shape
        if T > self.max_len:
            raise ValueError(f"sequence length {T} exceeds max_len {self.max_len}")
        ctx = self.in_proj(frame_feats) + self.pos_emb[:T].unsqueeze(0)
        q = self.queries.unsqueeze(0).expand(B, -1, -1)
        for layer in self.layers:
            q = q + layer["cross_attn"](
                layer["q_norm"](q),
                layer["ctx_norm"](ctx),
                key_padding_mask=key_padding_mask,
            )
            q = q + layer["ff"](layer["ff_norm"](q))
        return self.out_norm(q)  # [B, K, d_model]


class FrameDecoder(nn.Module):
    """Reconstruct future frames from e_target (reconstruction-pretraining only).

    e_target [B, K, d_model] + a target frame count T -> frames [B, T, C, H, W].
    `T` temporally-positioned frame queries cross-attend to the K e_target tokens
    to recover per-frame features; a small transposed-conv head renders each
    feature to an image (sigmoid -> [0, 1], matching the encoder's x/255 input
    range). This decoder exists ONLY to pressure e_target into faithfully
    representing the future during reconstruction pretraining; it is discarded
    (encoder frozen) before the flow/policy training and never runs at inference.
    """

    def __init__(
        self,
        d_model: int,
        out_ch: int = 3,
        img_size: int = 64,
        base_ch: int = 128,
        base_hw: int = 8,
        num_layers: int = 2,
        num_heads: int = 4,
        max_len: int = 512,
    ):
        super().__init__()
        up = img_size // base_hw
        if img_size % base_hw != 0 or up < 1 or (up & (up - 1)) != 0:
            raise ValueError("img_size / base_hw must be a power of two >= 1")
        self.d_model = d_model
        self.base_ch = base_ch
        self.base_hw = base_hw
        self.max_len = max_len
        # Frame queries are the DEFINING signal for each output frame (there are no
        # other per-frame inputs), so they need unit-scale init -- a 0.02 positional
        # scale gets washed out by the query LayerNorm and every frame collapses to the
        # same (mean) image (caught by the reconstruction smoke's time-variance check).
        self.frame_pos = nn.Parameter(torch.zeros(max_len, d_model))
        nn.init.normal_(self.frame_pos, std=1.0)
        self.layers = nn.ModuleList(
            nn.ModuleDict(
                {
                    "q_norm": nn.LayerNorm(d_model),
                    "ctx_norm": nn.LayerNorm(d_model),
                    "cross_attn": CrossAttention(d_model, num_heads),
                    "ff_norm": nn.LayerNorm(d_model),
                    "ff": FeedForward(d_model),
                }
            )
            for _ in range(num_layers)
        )
        self.to_spatial = nn.Linear(d_model, base_ch * base_hw * base_hw)
        convs: list[nn.Module] = []
        c = base_ch
        for _ in range(int(round(math.log2(up)))):
            nxt = max(out_ch * 4, c // 2)
            # GroupNorm after each transposed conv keeps activations at unit scale so
            # the signal does not vanish through the upsampling stack (without it the
            # default conv init attenuates the input ~100x and every frame collapses to
            # a constant -- caught by the reconstruction smoke's time-variance check).
            convs += [
                nn.ConvTranspose2d(c, nxt, kernel_size=4, stride=2, padding=1),
                nn.GroupNorm(math.gcd(8, nxt), nxt),
                nn.GELU(),
            ]
            c = nxt
        convs += [nn.Conv2d(c, out_ch, kernel_size=3, padding=1)]
        self.render = nn.Sequential(*convs)

    def forward(self, e_target: torch.Tensor, num_frames: int) -> torch.Tensor:
        # e_target [B, K, d_model] -> frames [B, T, C, H, W] in [0, 1].
        B, K, _ = e_target.shape
        T = num_frames
        if T > self.max_len:
            raise ValueError(f"num_frames {T} exceeds max_len {self.max_len}")
        q = self.frame_pos[:T].unsqueeze(0).expand(B, -1, -1)
        for layer in self.layers:
            q = q + layer["cross_attn"](layer["q_norm"](q), layer["ctx_norm"](e_target))
            q = q + layer["ff"](layer["ff_norm"](q))
        h = self.to_spatial(q).reshape(B * T, self.base_ch, self.base_hw, self.base_hw)
        img = torch.sigmoid(self.render(h))  # [B*T, C, H, W]
        return img.reshape(B, T, img.shape[1], img.shape[2], img.shape[3])


class ETargetGenerator(nn.Module):
    """Future frames [B, T, C, H, W] -> e_target [B, K, d_model].

    A per-frame image encoder (INJECTED -- the SOTA-aligned ImpalaSmall, matched to
    the OGBench visual baseline) encodes each frame; the PerceiverPooler compresses
    the resulting variable-length feature sequence into the fixed K x D e_target.
    After reconstruction pretraining (this module + a FrameDecoder) the whole module
    is FROZEN and used as the honest target generator. The injected encoder also runs
    at inference to encode the (s, g) conditioning, so it is capacity-matched to the
    baselines -- the real fairness lever.
    """

    def __init__(
        self,
        encoder: nn.Module,
        encoder_out: int,
        d_model: int,
        k: int,
        num_layers: int = 2,
        num_heads: int = 4,
        max_len: int = 512,
    ):
        super().__init__()
        self.encoder = encoder
        self.pooler = PerceiverPooler(
            d_in=encoder_out,
            d_model=d_model,
            k=k,
            num_layers=num_layers,
            num_heads=num_heads,
            max_len=max_len,
        )

    def forward(
        self,
        frames: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # frames [B, T, C, H, W] (already scaled to [0, 1]) -> e_target [B, K, d_model].
        B, T = frames.shape[0], frames.shape[1]
        feats = self.encoder(frames.reshape(B * T, *frames.shape[2:])).reshape(B, T, -1)
        return self.pooler(feats, key_padding_mask=key_padding_mask)
