"""[LEGACY — 視覺路線] WPM 的 top-level model，吃 uint8 影像。
LaCoT 走 state 路線、完全不用這支；保留只為了追溯。
⛔ 這份仍帶 wpm 的批次契約與 encoder_arch="wpm"，不要 import 進 lacot。
"""
"""Causal transformer backbone, token assembly, and the top-level WPMModel.

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

from m4model.heads import DiscretizedActionHead, FutureHead, ValueHead

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


class WPMModel(nn.Module):
    """Encoder + backbone + heads, wired for the depth-0 (GCBC-floor) path.

    Fixed batch contract (set by the data pipeline; see wpm/models/README.md):
        obs:          uint8  (B, N_h+1, 64, 64, 3)   frames at t-N_h*k, ..., t-k, t (current last)
        past_act:     float32 (B, N_h, chunk_len, A) the k-step chunk executed from each past frame
        history_mask: float32 (B, N_h)               1.0 real, 0.0 padded (near a trajectory start)
        goal:         uint8  (B, 64, 64, 3)
        act_chunk:    float32 (B, chunk_len, A)       target action chunk, in [-1, 1]
        weight:       float32 (B,)                    per-example advantage weight

    Two further keys are read only by the oracle-future path
    (`forward_oracle`, issue #127) and only when the model was built with
    `fut_slice_offsets`:

        slices:       uint8   (B, S, 64, 64, 3)      TRUE future frames at t+delta_j
        progress:     float32 (B,)                   V*(s_{t+H}, g) - V*(s_t, g)

    History is chunk-spaced, not raw-step-spaced (design doc S4.1, decided
    2026-08-03): past pair j is the state k steps further back than pair
    j-1, together with the whole k-step chunk executed from it. This matches
    receding-horizon execution, where the model only ever sees history at
    chunk cadence, so training and evaluation see the same layout.

    N_h (history length) is read from `obs`'s shape at call time, so the
    same model works with any N_h the batch provides. `action_dim` (A) and
    `chunk_len` are fixed at construction, because they size the linear
    layers below.
    """

    def __init__(
        self,
        action_dim: int,
        chunk_len: int = 4,
        d_model: int = 512,
        n_layers: int = 8,
        n_heads: int = 8,
        num_bins: int = 256,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        max_rounds: int = 8,
        env_time_size: int = 256,
        env_time_bias: int = 64,
        encoder_clean_skip: bool = False,
        encoder_arch: str = "wpm",
        fut_slice_offsets: tuple[int, ...] | list[int] | None = None,
        progress_norm: dict | None = None,
        progress_init_norm: float | None = None,
        progress_init_spread: float | None = None,
        fut_null_tokens: bool = False,
        fut_progress_off: bool = False,
        proprio_dim: int | None = None,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.chunk_len = chunk_len
        self.d_model = d_model

        if encoder_arch == "wpm":
            self.encoder = ImageEncoder(d_model=d_model, clean_skip=encoder_clean_skip)
        elif encoder_arch == "resnet_gn":
            # Build (and discard) a throwaway default encoder first, purely
            # to advance the shared global RNG stream by exactly what the
            # "wpm" branch above consumes constructing its own encoder.
            # ResNetGNEncoder costs the stream nothing to build (it isolates
            # itself, wpm/models/encoder.py) — without this throwaway,
            # every module built below (backbone, heads, embeddings) would
            # start from different weights depending on which encoder_arch
            # was chosen, for the same seed, since the two encoders are
            # different architectures with different default-initialization
            # footprints. Same-seed comparability (PR #121, encoder_clean_skip)
            # extends across architectures this way.
            ImageEncoder(d_model=d_model)
            self.encoder = ResNetGNEncoder(d_model=d_model)
        else:
            raise ValueError(f"unknown encoder_arch: {encoder_arch!r}")
        self.backbone = CausalTransformer(d_model, n_layers, n_heads, mlp_ratio, dropout)
        self.action_head = DiscretizedActionHead(d_model, action_dim, chunk_len, num_bins)
        # Stubs: correct output shapes are ready, but nothing trains or calls
        # these yet (design doc S4.3; team instructions for this milestone).
        self.future_head = FutureHead(d_model)
        self.value_head = ValueHead(d_model)

        # Content embedding for one past pair's executed chunk (chunk_len *
        # action_dim numbers — history is chunk-spaced, design doc S4.1). A
        # *separate* linear layer of the same input size will be added later
        # for embedding the model's own refined chunk a_{i-1} when it is
        # re-appended as context for the next recursion round (design doc
        # S4.2). It is kept separate rather than shared with this one because
        # a_{i-1} is the model's own (possibly wrong) guess, role ACT, while
        # past_act is the dataset's real executed chunk, role CTX — the two
        # coincide in shape only, not in what they mean. That layer does not
        # exist yet, since no round is implemented this milestone.
        self.past_act_embed = nn.Linear(chunk_len * action_dim, d_model)

        # The ACT-slot query token: one learned vector, shared across every
        # example and every position that asks "what should the action be?"
        # (the same pattern as a ViT class token / a BERT [CLS] embedding).
        self.act_query = nn.Parameter(torch.randn(d_model) * 0.02)

        # The four additive embeddings (design doc S4.3). "Content" is not a
        # table — it comes from the encoder or an action-embedding layer
        # above. The other three are small lookup tables:
        self.role_embed = nn.Embedding(NUM_ROLES, d_model)
        self.env_time_embed = nn.Embedding(env_time_size, d_model)
        self.round_embed = nn.Embedding(max_rounds, d_model)
        self.env_time_size = env_time_size
        self.env_time_bias = env_time_bias

        # --- the oracle-future (FUT) tokens, issue #127 -------------------
        # `fut_slice_offsets` None means the plain depth-0 model: no FUT
        # parameter exists at all, so every milestone-0 config and checkpoint
        # loads and behaves exactly as before. When offsets are given, the
        # model gains K_z = len(offsets) + 1 future tokens (one per true
        # future slice, plus one progress scalar).
        #
        # These are built LAST on purpose. Constructing an nn.Linear or
        # sampling an nn.Parameter draws from the shared global RNG stream,
        # so building them earlier would shift the starting weights of every
        # module built after them, and a same-seed oracle run would no longer
        # share the floor's initial backbone/head weights (the same
        # comparability rule wpm/models/encoder.py's local generators keep,
        # PR #121).
        self.fut_slice_offsets = tuple(fut_slice_offsets) if fut_slice_offsets is not None else None
        # How the progress scalar is standardized before it is embedded, and
        # how loud the resulting token is allowed to be at initialization.
        # Both are constants measured once and written into the config, not
        # statistics recomputed per run — see `normalized_progress` and the
        # rescale below for why each exists (PR #128 review).
        self.progress_norm = dict(progress_norm) if progress_norm else None
        if self.progress_norm is not None and not self.progress_norm.get("std", 0) > 0:
            raise ValueError(f"progress_norm needs a positive std, got {progress_norm!r}")
        self.progress_init_norm = progress_init_norm
        self.progress_init_spread = progress_init_spread
        self.fut_null_tokens = bool(fut_null_tokens)
        # The progress-off arm (issue #127 It-3, headroom-mapping sweep): the
        # future FRAMES stay real, but the progress token is PERMANENTLY that
        # slot's own learned absent vector — the eval-time
        # `--steering-fut frames` mask, made a training arm, so a value
        # function never has to be trained or loaded for this cell at all.
        # Mutually exclusive with fut_null_tokens (which reads no future of
        # any kind); `check_oracle_config` refuses a yaml that sets both.
        self.fut_progress_off = bool(fut_progress_off)
        # Proprioception diagnostic (owner question 2026-08-10, design in
        # docs/RESULTS_milestone1.md): one extra token carrying qpos-without-
        # global-xy plus qvel at the CURRENT step. LayerNorm on the way in
        # (joint angles and velocities live on wildly different scales) and on
        # the way out — the progress token taught us the hard way that a
        # mis-scaled injected token can dominate three image tokens 143 to 1,
        # and a d_model LayerNorm pins the content norm to sqrt(d_model),
        # within tolerance of an image token's own loudness.
        self.proprio_dim = int(proprio_dim) if proprio_dim else None
        if self.fut_slice_offsets is not None and self.proprio_dim:
            self.proprio_embed = nn.Sequential(
                nn.LayerNorm(self.proprio_dim),
                nn.Linear(self.proprio_dim, d_model),
                nn.LayerNorm(d_model),
            )
        if self.fut_slice_offsets is not None:
            # Content embedding for the progress scalar V*(s_{t+H}, g) -
            # V*(s_t, g) (design doc S4.4). One number in, one token out.
            self.progress_embed = nn.Linear(1, d_model)
            self._scale_progress_embed(progress_init_norm, progress_init_spread)
            # One learned "absent" content vector per FUT slot (presence
            # dropout, issue #127). Per slot rather than shared, so a model
            # asked for "only the farthest slice" sees a token that says
            # "the delta=16 slice is missing", not a generic blank.
            self.fut_absent = nn.Parameter(torch.randn(self.num_fut_tokens, d_model) * 0.02)
            if fut_null_tokens:
                # The trained-null control's future content: one learned vector
                # per position, grounded in nothing. Same initialization scale
                # as the absent embeddings above, so the arm starts where the
                # oracle arm's own "no future here" tokens start.
                self.fut_null = nn.Parameter(torch.randn(self.num_fut_tokens, d_model) * 0.02)

    @torch.no_grad()
    def _scale_progress_embed(self, target_norm: float | None, target_spread: float | None) -> None:
        """Give the progress token an image token's loudness AND its variability.

        Two different things, and the review's finding is that only fixing the
        first is not enough (PR #128, measured at the real geometry):

        * **loudness** is the content's L2 norm. Raw, the progress token
          entered at 133 against 20 for every image token. Matching it alone
          still left a readout that responded ~24x more to swapping the one
          scalar than to swapping all three future frames.
        * **variability** is how far the content MOVES when the input
          changes. An image token is mostly a large constant — the scenes in
          one dataset look alike, so the encoder's outputs sit close together
          — with a small varying part. A token built from one standardized
          scalar is almost all varying part. Equal norms, wildly unequal
          response.

        For a standardized input p (mean 0, variance 1), content = W p + b, so
        the movement is exactly ||W|| per standard deviation and the expected
        squared norm is ||W||^2 + ||b||^2. Setting ||W|| = target_spread and
        ||b||^2 = target_norm^2 - target_spread^2 therefore pins both at once.
        Both targets are this environment's own measured image-token numbers
        (`scripts/wpm_token_probe.py`), written into the config.

        This only rescales values that were already drawn, so it consumes
        nothing from the shared random stream and leaves a same-seed floor
        model's weights untouched.
        """
        if target_norm is None and target_spread is None:
            return
        weight, bias = self.progress_embed.weight, self.progress_embed.bias
        if target_spread is None:  # loudness only
            current = torch.sqrt(weight.pow(2).sum() + bias.pow(2).sum())
            weight.mul_(target_norm / current)
            bias.mul_(target_norm / current)
            return
        weight.mul_(target_spread / weight.norm())
        if target_norm is None:
            return
        if target_norm <= target_spread:
            raise ValueError(
                f"progress_init_norm ({target_norm}) must exceed progress_init_spread "
                f"({target_spread}): the norm is the whole content, the spread only the part "
                "that moves with the input."
            )
        bias_norm = float(np.sqrt(target_norm**2 - target_spread**2))
        bias.mul_(bias_norm / bias.norm())

    def _env_time_ids(self, offsets: torch.Tensor) -> torch.Tensor:
        """Map a signed environment-time offset (0 = "now") to an embedding row.

        Past context tokens use negative offsets, spaced at chunk_len (-N_h *
        chunk_len, ..., -chunk_len); the current state, the goal, and the
        depth-0 ACT slot all use offset 0 — see wpm/models/README.md for why
        the goal is pinned to 0 today, and what changes once the data
        pipeline carries a real goal timestep. Future rounds will use
        positive offsets (the slice offsets delta_j, design doc S4.4), which
        this table already has room for.

        An offset that does not fit the table is a configuration bug (too
        much history or too coarse a table for `env_time_size` /
        `env_time_bias`), not a value to silently paper over by clamping —
        clamping would alias two different environment times onto the same
        embedding row.
        """
        idx = offsets + self.env_time_bias
        assert idx.min() >= 0 and idx.max() <= self.env_time_size - 1, (
            f"environment-time offset out of range: id range [{int(idx.min())}, "
            f"{int(idx.max())}], table holds [0, {self.env_time_size - 1}] "
            f"(env_time_bias={self.env_time_bias}, env_time_size={self.env_time_size})"
        )
        return idx.long()

    def assemble_h0(
        self,
        obs: torch.Tensor,
        past_act: torch.Tensor,
        goal: torch.Tensor,
        history_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Build h_0 = [GOAL] [CTX: past pairs + current state] [ACT slot] (design doc S4.2).

        `history_mask` (B, N_h; 1.0 real, 0.0 padded) picks the role of each
        past pair's two tokens (state and action) per example: ROLE_CTX where
        real, ROLE_CTX_PAD where the data pipeline padded it. Content is
        never zeroed or replaced — see the module docstring for why.

        Returns: (B, T, d_model) where T = 1 + 2*N_h + 1 + 1.
        """
        batch_size = obs.shape[0]
        n_hist = obs.shape[1] - 1
        device = obs.device

        goal_content = self.encoder(goal)  # (B, d_model)
        state_content = self.encoder(obs)  # (B, N_h + 1, d_model)
        # past_act: (B, N_h, chunk_len, A) -> flatten the last two dims so one
        # linear layer embeds a whole executed chunk per past pair.
        past_act_content = (
            self.past_act_embed(past_act.reshape(batch_size, n_hist, -1)) if n_hist > 0 else None
        )  # (B, N_h, d_model)

        def const_col(value: int, dtype: torch.dtype) -> torch.Tensor:
            return torch.full((batch_size, 1), value, device=device, dtype=dtype)

        contents = [goal_content.unsqueeze(1)]
        roles = [const_col(ROLE_GOAL, torch.long)]
        offsets = [const_col(0, torch.float32)]
        rounds = [const_col(0, torch.long)]

        for h in range(n_hist):
            # History is chunk-spaced (design doc S4.1): past pair h is
            # (n_hist - h) whole chunks before "now", i.e. chunk_len steps
            # apart on the raw environment timeline.
            offset = -(n_hist - h) * self.chunk_len
            real = const_col(ROLE_CTX, torch.long)
            padded = const_col(ROLE_CTX_PAD, torch.long)
            role_h = torch.where(history_mask[:, h : h + 1] > 0.5, real, padded)

            contents.append(state_content[:, h : h + 1])
            roles.append(role_h)
            offsets.append(const_col(offset, torch.float32))
            rounds.append(const_col(0, torch.long))

            contents.append(past_act_content[:, h : h + 1])
            roles.append(role_h)
            offsets.append(const_col(offset, torch.float32))
            rounds.append(const_col(0, torch.long))

        contents.append(state_content[:, -1:])  # current state
        roles.append(const_col(ROLE_CTX, torch.long))
        offsets.append(const_col(0, torch.float32))
        rounds.append(const_col(0, torch.long))

        contents.append(self.act_query.view(1, 1, -1).expand(batch_size, 1, -1))  # ACT slot
        roles.append(const_col(ROLE_ACT, torch.long))
        offsets.append(const_col(0, torch.float32))
        rounds.append(const_col(0, torch.long))

        content = torch.cat(contents, dim=1)
        role_ids = torch.cat(roles, dim=1)
        offset_vals = torch.cat(offsets, dim=1)
        round_ids = torch.cat(rounds, dim=1)

        return (
            content
            + self.role_embed(role_ids)
            + self.env_time_embed(self._env_time_ids(offset_vals))
            + self.round_embed(round_ids)
        )

    def forward_depth0(self, batch: dict) -> torch.Tensor:
        """The full depth-0 (N_z = 0) path: h_0 -> u_0 logits.

        u_0 is both the proposal and the executed action at this depth
        (design doc S4.2: "Depth N_z = 0 runs only the proposal — this is
        B0-int"). Returns: action logits (B, chunk_len, action_dim, num_bins).
        """
        tokens = self.assemble_h0(batch["obs"], batch["past_act"], batch["goal"], batch["history_mask"])
        hidden = self.backbone(tokens)
        act_slot_hidden = hidden[:, -1]  # the ACT slot is always the last token at depth 0
        return self.action_head(act_slot_hidden)

    # -- oracle-future path (issue #127) -----------------------------------

    @property
    def num_fut_tokens(self) -> int:
        """K_z: one token per future slice, the progress token, and — for the
        proprioception diagnostic only — one proprioception token; 0 when off."""
        if self.fut_slice_offsets is None:
            return 0
        return len(self.fut_slice_offsets) + 1 + (1 if self.proprio_dim else 0)

    def normalized_progress(self, progress: torch.Tensor) -> torch.Tensor:
        """The progress scalar, standardized by the constants in the config.

        `V*(s_{t+H}, g) - V*(s_t, g)` is in whatever units the frozen value
        function happens to use — on cube-double, mean near 5 with a standard
        deviation near 11. Feeding that raw into a linear embedding gave the
        progress token an L2 norm of about 104 against about 40 for every
        image token, and the readout listened to it far more than to the
        frames before training even started (PR #128 review).

        The two constants are measured once, from the training split, by
        `scripts/wpm_progress_norm.py`, and written into the config
        (`model.progress_norm`). They are deliberately not recomputed at run
        time: a per-run statistic would make two runs of the same config
        normalize differently, and a constant in a committed file can be
        reviewed. `None` leaves the scalar raw (every pre-review checkpoint).
        """
        progress = progress.float()
        if self.progress_norm is None:
            return progress
        return (progress - self.progress_norm["mean"]) / self.progress_norm["std"]

    def assemble_fut(
        self,
        slices: torch.Tensor | None = None,
        progress: torch.Tensor | None = None,
        fut_present: torch.Tensor | None = None,
        batch_size: int | None = None,
        proprio: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Build the K_z future tokens z* (design doc S4.4).

            slices:      uint8   (B, S, 64, 64, 3)  the TRUE frames at t+delta_j
            progress:    float32 (B,)               V*(s_{t+H}, g) - V*(s_t, g)
            fut_present: float   (B, K_z) 1 present / 0 absent, or None = all present
            batch_size:  needed only by the trained-null arm, which reads no input

        **The trained-null arm** (`fut_null_tokens`, design doc S5.6 item 3)
        replaces the content of all K_z tokens with per-position learned
        parameters and reads no future at all — not the frames, not the value
        scalar. It is the control that separates "depth helps because the
        futures say something" from "depth helps because there are more
        tokens to compute over": same architecture, same token count, same
        presence dropout, same losses, no grounding. The design document is
        explicit that it has to be trained end to end, because plugging blank
        tokens into the main checkpoint would measure input shock instead.

        **The progress-off arm** (`fut_progress_off`, issue #127 It-3) keeps
        the slice tokens real but forces the progress token to its slot's
        learned "absent" vector permanently — no value function is trained or
        loaded for it at all. This is the headroom-mapping sweep's own arm:
        cheap to build across many (recipe, environment) cells because it
        needs no per-cell frozen V*, at the cost of never showing the readout
        a progress scalar.

        Slice tokens are the LIVE encoder's features of the true future
        frames — the same encoder and the same pooling the goal and current
        frame go through, so a future frame and a current frame are
        comparable objects to the backbone. The progress token is the frozen
        value function's scalar through one linear layer.

        An absent token keeps its role and environment-time embeddings and
        swaps only its CONTENT for that slot's learned "absent" vector, so
        "the delta=24 slice is missing" stays a statement about that slot at
        that environment time, at a fixed token count. Training draws
        `fut_present` at random (presence dropout, p = 0.2 per token) which
        is what makes every evaluation variant — farthest-slice-only, one
        slice at a time, all absent — an in-distribution input rather than a
        distribution shock (plan objective row 11).

        Returns: (B, K_z, d_model).
        """
        if self.fut_slice_offsets is None:
            raise RuntimeError(
                "this model was built without fut_slice_offsets, so it has no future tokens; "
                "set model.fut_slice_offsets in the training config to use the oracle path"
            )
        n_slices = len(self.fut_slice_offsets)
        if self.fut_null_tokens:
            if batch_size is None:
                if slices is None:
                    raise ValueError("the null arm needs a batch_size when no slices are given")
                batch_size = slices.shape[0]
            device = self.fut_null.device
            content = self.fut_null.expand(batch_size, -1, -1)
        elif self.fut_progress_off:
            # `progress` is deliberately not inspected: any value passed in
            # here is ignored, the same way the null arm above ignores an
            # incidental `slices`/`progress`. Loud rejection of a progress
            # scalar that should never exist lives one layer up, in
            # `train_wpm.py`'s config guard and its training-step re-check —
            # this function's job is only to build the tensor.
            if slices is None:
                raise ValueError(
                    "the progress-off arm still needs real `slices`; only the trained-null arm "
                    "(fut_null_tokens) may run without any future input at all"
                )
            if slices.shape[1] != n_slices:
                raise ValueError(
                    f"the batch carries {slices.shape[1]} future slices but the model expects "
                    f"{n_slices} (offsets {self.fut_slice_offsets}); the data pipeline and the "
                    "model config disagree about the future-slice geometry"
                )
            batch_size = slices.shape[0]
            device = slices.device

            slice_content = self.encoder(slices)  # (B, S, d_model)
            # The progress token is PERMANENTLY this slot's own learned absent
            # vector -- never computed, because this arm never trains or loads
            # a value function at all (the eval-time `--steering-fut frames`
            # mask, made a training arm). `fut_present` masking below is a
            # no-op on this column for exactly that reason: present or not,
            # its content already equals `fut_absent[-1]`.
            # With a proprio slot the absent-progress vector is at index -2
            # (proprio owns -1); without one it stays at -1 as it always was.
            prog_slot = -2 if self.proprio_dim else -1
            progress_content = self.fut_absent[prog_slot].expand(batch_size, 1, -1)
            parts = [slice_content, progress_content]
            if self.proprio_dim:
                if proprio is None:
                    raise ValueError(
                        "this model was built with proprio_dim, so assemble_fut needs the "
                        "`proprio` tensor; the data pipeline emits it under include_proprio"
                    )
                parts.append(self.proprio_embed(proprio).unsqueeze(1))
            content = torch.cat(parts, dim=1)
        else:
            if slices is None or progress is None:
                raise ValueError(
                    "the oracle path needs both `slices` and `progress`; only the trained-null "
                    "arm (fut_null_tokens) or the progress-off arm (fut_progress_off) may run "
                    "without them"
                )
            if slices.shape[1] != n_slices:
                raise ValueError(
                    f"the batch carries {slices.shape[1]} future slices but the model expects "
                    f"{n_slices} (offsets {self.fut_slice_offsets}); the data pipeline and the "
                    "model config disagree about the future-slice geometry"
                )
            batch_size = slices.shape[0]
            device = slices.device

            slice_content = self.encoder(slices)  # (B, S, d_model)
            progress_content = self.progress_embed(
                self.normalized_progress(progress).reshape(batch_size, 1)
            )
            parts = [slice_content, progress_content.unsqueeze(1)]
            if self.proprio_dim:
                if proprio is None:
                    raise ValueError(
                        "this model was built with proprio_dim, so assemble_fut needs the "
                        "`proprio` tensor; the data pipeline emits it under include_proprio"
                    )
                parts.append(self.proprio_embed(proprio).unsqueeze(1))
            content = torch.cat(parts, dim=1)

        if fut_present is not None:
            present = fut_present.to(content.dtype).reshape(batch_size, self.num_fut_tokens, 1)
            content = present * content + (1.0 - present) * self.fut_absent.to(content.dtype)
        else:
            # `expand` returns a view; every later op here is out-of-place, but
            # make the null arm's content its own tensor anyway so no caller can
            # write through it into the shared parameter.
            content = content.contiguous() if self.fut_null_tokens else content

        # The progress token summarizes the future at t + H_imag, which is the
        # farthest slice offset, so it shares that environment time.
        offset_list = list(self.fut_slice_offsets) + [max(self.fut_slice_offsets)]
        if self.proprio_dim:
            # Proprioception describes the CURRENT step, so its environment
            # time is 0 — unlike every other FUT token, which is the point:
            # it supplies phase for now, the frames supply route for later.
            offset_list.append(0)
        offsets = torch.tensor(
            offset_list, device=device, dtype=torch.float32,
        ).expand(batch_size, self.num_fut_tokens)
        roles = torch.full((batch_size, self.num_fut_tokens), ROLE_FUT, device=device, dtype=torch.long)
        rounds = torch.zeros((batch_size, self.num_fut_tokens), device=device, dtype=torch.long)

        return (
            content
            + self.role_embed(roles)
            + self.env_time_embed(self._env_time_ids(offsets))
            + self.round_embed(rounds)
        )

    def forward_oracle(self, batch: dict, fut_present: torch.Tensor | None = None) -> dict:
        """One forward over [GOAL][CTX][ACT][FUT], returning both action reads.

        Two logits come out of a single pass over a single trunk:

          proposal_logits  read at the ACT slot, the last token of h_0. Causal
                           attention means that position attends only to
                           itself and earlier tokens, so it CANNOT see any FUT
                           token — the proposal is the same q(a | s, g) the
                           GCBC floor computes, and `forward_depth0` on this
                           same checkpoint returns exactly these logits (the
                           B0-int floor).
          readout_logits   read at the last FUT token, which attends to
                           everything: the goal, the context, and z*. This is
                           pi(a | s, g, z*), chosen implicitly — no scoring,
                           no candidates (hard constraints R3/R7).

        The action head is shared between the two reads. It is the same
        "one transformer plays every role through role tokens" rule the
        design doc states (S4.3); a second head would also add ~5M
        parameters the floors do not have, which would confound every
        matched-capacity comparison in this milestone.

        `batch` needs the depth-0 keys plus `slices` and `progress`.
        """
        h0 = self.assemble_h0(batch["obs"], batch["past_act"], batch["goal"], batch["history_mask"])
        # `.get`, not `[...]`: the trained-null arm's batches carry no future
        # frames and no progress scalar at all, which is the point of it.
        fut = self.assemble_fut(
            batch.get("slices"),
            batch.get("progress"),
            fut_present,
            batch_size=batch["obs"].shape[0],
            proprio=batch.get("proprio"),
        )
        hidden = self.backbone(torch.cat([h0, fut], dim=1))
        return {
            "proposal_logits": self.action_head(hidden[:, h0.shape[1] - 1]),
            "readout_logits": self.action_head(hidden[:, -1]),
        }
