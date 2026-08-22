"""Causal transformer backbone for LaCoT.

Design doc reference: WPM-Design-0803.md S4.2 (recursion trace), S4.3
(architecture instance).

This file has two parts:

1. `CausalTransformer` — a plain decoder-only transformer stack (causal
   means each position can only attend to itself and earlier positions, so
   the model cannot "see the future" of its own input sequence). It knows
   nothing about images, actions, or roles; it only turns a sequence of
   vectors into another sequence of vectors.
2. `WPMModel` — owns the image encoder, the transformer, and the output
   heads. It builds the token sequence described in the design doc (content
   + four additive embeddings: role, environment-time, recursion round —
   the fourth, "content", is not a lookup table; it is whatever the encoder
   or an action embedding produces) and runs the depth-0 path: propose one
   action chunk from the goal and the recent context, with no recursion.

`WPMModel` also runs the oracle-future path of the milestone after it
(issue #127): the TRUE future frames are appended as FUT tokens and a second
action read is taken after them, so one pass yields both the plain proposal
and the future-conditioned readout. Recursion proper — the model imagining
its own future from a tried action — is still not implemented here. The
token-layout machinery below (role ids, the environment-time embedding, the
round embedding table) already has room for it; see wpm/models/README.md for
exactly what a later milestone adds.

**History padding.** Near the start of a trajectory there is no full N_h
worth of real past pairs, so the data pipeline pads (wpm/README.md): a
padded frame is held from the trajectory's first frame (real, legitimate
content — never zeroed), but a padded action-chunk is sometimes a real short
held chunk and sometimes exactly zero, and the model cannot tell those two
apart from the numbers alone. `history_mask` (1 real, 0 padded) is consumed
by giving BOTH tokens of a padded pair a distinct role, ROLE_CTX_PAD, so the
model learns how much to trust a padded pair instead of the code guessing —
their content is never zeroed or replaced, since the frame side is always
legitimate and roughly 60% of padded action-chunks at N_h=4 are too (see
wpm/README.md, "Two things the model side must know about history_mask").
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from lacot.heads import DiscretizedActionHead, FutureHead, ValueHead

# Token roles (design doc S4.3, additive embedding 2). Every input token to
# the backbone gets exactly one of these. ROLE_CTX_PAD marks a past pair
# that the data pipeline padded (history_mask == 0) — same content, a
# different role, so the model can learn its own discount for it.
ROLE_GOAL, ROLE_CTX, ROLE_CTX_PAD, ROLE_ACT, ROLE_FUT = range(5)
NUM_ROLES = 5


class CausalTransformer(nn.Module):
    """A stack of standard transformer layers with a causal attention mask."""

    def __init__(
        self,
        d_model: int = 512,
        n_layers: int = 8,
        n_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=int(d_model * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.layers = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm_out = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d_model) -> (B, T, d_model), causally masked."""
        seq_len = x.size(1)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            seq_len, device=x.device, dtype=x.dtype
        )
        h = self.layers(x, mask=causal_mask, is_causal=True)
        return self.norm_out(h)
