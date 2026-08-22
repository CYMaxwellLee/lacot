"""Normalizing-flow head for M4 latent reasoning — the "generate u" block.

Design doc: docs/M4-NF-latent-planning-design.md S3-S5 (component 1), S7
(NF-CoT method). Engine = TARFlow ("Normalizing Flows are Capable Generative
Models", arXiv:2412.06329, github.com/apple/ml-tarflow).

M4 replaces WPM's explicit future-image imagination with a compact continuous
latent "thought" `u`, sampled and refined by a normalizing flow, then decoded
to an action chunk. This module is **component 1**: the flow itself — the
density `p(u | cond)` we can (a) evaluate *exactly* (log-likelihood) and (b)
sample from. Nothing here knows about actions, images, or WPM yet; it only
turns a sequence of latent token vectors into a base-Gaussian code and back,
exactly invertibly, with a tractable log-determinant.

**TARFlow in one paragraph.** A normalizing flow warps a simple base density
(here a standard Gaussian `z`) into a complex one (`u`) through an invertible
map with a tractable Jacobian. TARFlow makes the map an *autoregressive affine*
transform driven by a causal transformer: for each token position t, a shift
`mu_t` and a log-scale `alpha_t` are predicted from the EARLIER tokens
`u_{<t}` only, and u_t is normalized to `z_t = (u_t - mu_t) * exp(-alpha_t)`.
Because each token's transform depends only on earlier tokens, the Jacobian is
triangular and its log-determinant is *exactly* the sum of the log-scales — so
the change-of-variables likelihood is exact, not a bound. Several such blocks
are stacked with the token order flipped between them, so across blocks every
token can condition on every other.

**Why exact likelihood matters for M4** (design doc S1, S7): `u` is a "thought"
we must score (the flow loss aligns u to a compressed future trajectory) and,
later, optimize in latent space. A diffusion model gives only an ELBO and needs
many denoising steps to sample; a plain autoregressive transformer over
continuous values needs a chosen output family or discretization bins; a GAN
gives no likelihood at all. An exact, invertible, one-pass density is what the
method needs — and TARFlow is what makes a normalizing flow expressive enough
to be worth it.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from lacot.backbone import CausalTransformer

LOG_2PI = math.log(2 * math.pi)


class ARFlowBlock(nn.Module):
    """One autoregressive affine flow step (a TARFlow block).

    Maps a token sequence `u` (B, K, D) to a code `z` (B, K, D) of the same
    shape, invertibly. For each position t a causal transformer reads the
    earlier tokens `u_{<t}` (plus an optional conditioning prefix) and emits an
    affine transform (mu_t, alpha_t); the normalize direction is

        z_t = (u_t - mu_t) * exp(-alpha_t),

    with `log|det d z / d u| = -sum_t alpha_t` (triangular Jacobian). The sample
    direction inverts this token by token.

    `max_log_scale` softly bounds alpha through a tanh, so `exp(+/-alpha)` can
    never overflow; the final parameter layer is zero-initialized, so the block
    starts as the identity map (mu=0, alpha=0) and learns away from it — which
    is what keeps a deep stack stable at the start of training.
    """

    def __init__(
        self,
        token_dim: int,
        d_hidden: int = 256,
        n_layers: int = 2,
        n_heads: int = 4,
        max_log_scale: float = 7.0,
        cond_dim: int | None = None,
    ):
        super().__init__()
        self.token_dim = token_dim
        self.max_log_scale = max_log_scale
        self.embed = nn.Linear(token_dim, d_hidden)
        # A learned "start" prefix, used when no conditioning is given, so the
        # first token's transform (mu_0, alpha_0) is still a function of
        # something. When `cond_dim` is set, a projection of the conditioning
        # vector replaces it (see `_prefix`).
        self.start = nn.Parameter(torch.randn(d_hidden) * 0.02)
        self.cond_proj = nn.Linear(cond_dim, d_hidden) if cond_dim else None
        self.transformer = CausalTransformer(d_model=d_hidden, n_layers=n_layers, n_heads=n_heads)
        self.to_params = nn.Linear(d_hidden, 2 * token_dim)
        # Identity init: the block starts as z = u (mu=0, alpha=0). Stable stacks.
        nn.init.zeros_(self.to_params.weight)
        nn.init.zeros_(self.to_params.bias)

    def _prefix(self, batch_size: int, cond: torch.Tensor | None, device: torch.device) -> torch.Tensor:
        """The length-1 prefix token that seeds position 0's transform."""
        if cond is not None:
            if self.cond_proj is None:
                raise ValueError("this block was built without cond_dim but received a cond tensor")
            return self.cond_proj(cond).unsqueeze(1)  # (B, 1, H)
        return self.start.to(device).view(1, 1, -1).expand(batch_size, 1, -1)

    def _params(self, seq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """seq (B, L, H) -> (mu, alpha), each (B, L, token_dim), alpha softly bounded."""
        raw = self.to_params(self.transformer(seq))
        mu, alpha_raw = raw.chunk(2, dim=-1)
        alpha = self.max_log_scale * torch.tanh(alpha_raw / self.max_log_scale)
        return mu, alpha

    def forward(self, u: torch.Tensor, cond: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Normalize: `u` (B, K, D) -> (`z` (B, K, D), `logdet` (B,))."""
        batch_size, seq_len, _ = u.shape
        prefix = self._prefix(batch_size, cond, u.device)
        # Position t must see only u_{<t}: feed [prefix, u_0, ..., u_{K-2}] so
        # the causal output at t is a function of exactly u_{<t}.
        inp = torch.cat([prefix, self.embed(u[:, :-1])], dim=1)  # (B, K, H)
        mu, alpha = self._params(inp)
        z = (u - mu) * torch.exp(-alpha)
        logdet = -alpha.sum(dim=(1, 2))
        return z, logdet

    def inverse(self, z: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        """Sample direction: `z` (B, K, D) -> `u` (B, K, D), one token at a time.

        u_t depends on u_{<t}, so the inverse is sequential: recover u_0, then
        u_1 given u_0, and so on. O(K) transformer passes — fine for the small
        K a latent "thought" uses.
        """
        batch_size, seq_len, _ = z.shape
        prefix = self._prefix(batch_size, cond, z.device)
        u = torch.zeros_like(z)
        for t in range(seq_len):
            seq = prefix if t == 0 else torch.cat([prefix, self.embed(u[:, :t])], dim=1)
            mu, alpha = self._params(seq)  # params at the last position are for token t
            u[:, t] = z[:, t] * torch.exp(alpha[:, -1]) + mu[:, -1]
        return u


class Permutation(nn.Module):
    """Reverses token order between flow blocks (its own inverse, logdet 0).

    An autoregressive block conditions token t only on the tokens before it;
    flipping the order between blocks lets a later block condition on what an
    earlier one treated as "future", so the stack as a whole is not order-biased.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.flip(x, dims=(1,))

    inverse = forward


class Flow(nn.Module):
    """A stack of autoregressive affine blocks — an exact-likelihood density
    over token sequences `u` (B, K, token_dim), optionally conditioned on `cond`.

    - `log_prob(u, cond)` returns exact `log p(u | cond)` of shape (B,).
    - `sample(n, cond)` draws `u ~ p(. | cond)`.

    The base density is a standard Gaussian of the same shape as `u`.
    """

    def __init__(
        self,
        token_dim: int,
        seq_len: int,
        n_blocks: int = 4,
        d_hidden: int = 256,
        n_layers: int = 2,
        n_heads: int = 4,
        cond_dim: int | None = None,
    ):
        super().__init__()
        self.token_dim = token_dim
        self.seq_len = seq_len
        self.blocks = nn.ModuleList(
            ARFlowBlock(token_dim, d_hidden, n_layers, n_heads, cond_dim=cond_dim)
            for _ in range(n_blocks)
        )
        self.perm = Permutation()

    def forward(self, u: torch.Tensor, cond: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """`u` -> (`z`, total `logdet`). The token order is flipped between blocks."""
        logdet = u.new_zeros(u.shape[0])
        z = u
        for i, block in enumerate(self.blocks):
            z, ld = block(z, cond)
            logdet = logdet + ld
            if i < len(self.blocks) - 1:
                z = self.perm(z)
        return z, logdet

    def log_prob(self, u: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        """Exact `log p(u | cond)`, shape (B,)."""
        z, logdet = self.forward(u, cond)
        base = (-0.5 * z.pow(2) - 0.5 * LOG_2PI).sum(dim=(1, 2))
        return base + logdet

    def nll(self, u: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        """Mean negative log-likelihood in nats — the training loss."""
        return -self.log_prob(u, cond).mean()

    @torch.no_grad()
    def sample(
        self,
        n: int,
        cond: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Draw `n` sequences `u ~ p(. | cond)` by inverting the flow from Gaussian noise."""
        device = device or next(self.parameters()).device
        z = torch.randn(n, self.seq_len, self.token_dim, device=device, generator=generator)
        u = z
        # Invert the forward op sequence exactly in reverse: the perm that
        # FOLLOWED block i is undone before block i's own inverse.
        for i in reversed(range(len(self.blocks))):
            if i < len(self.blocks) - 1:
                u = self.perm.inverse(u)
            u = self.blocks[i].inverse(u, cond)
        return u
