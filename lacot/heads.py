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
