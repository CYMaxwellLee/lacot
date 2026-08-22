"""M4 full-model wiring (build order S5 step 5): compose the validated blocks.

Design doc: docs/M4-NF-latent-planning-design.md (S3 losses, S4 components, S5
build order, S8 F=identity, S10 refine+consistency, S11 target side, Q1/Q2/Q3).

Pieces, each already validated in isolation:
  - ETargetGenerator (wpm/models/e_target.py)  -- future frames -> e_target; FROZEN
    after reconstruction pretraining (the honest target; e_target_recon_smoke).
  - Flow (wpm/models/nf_head.py)               -- component 1 density p(u|cond)
    (nf_head_smoke, e_target_lnf_smoke, e_target_e2e_smoke).
  - DiscretizedActionHead (wpm/models/heads.py) -- decode action chunk from u
    (laction_smoke).
  - RefineOperator (here)                       -- round-level refine loop
    (refine_consistency_smoke).

⚠️ DESIGN CHOICES made in this wiring -- FLAG for 主人 to confirm / redirect:
  (a) cond = the FROZEN generator's encoder applied to (s,g) frames, concatenated
      -> [B, 2*encoder_out]. (Design S11: the DEPLOYED encoder is the
      reconstruction-pretrained, frozen one; reused for the (s,g) conditioning.)
  (b) F = IDENTITY (design S8): u_target = e_target; L_NF trains the density
      directly on e_target. (DO-NOT-FORGET S8: swap for a shallow ARFlowBlock if
      the density underfits e_target.)
  (c) the refine loop starts from a SAMPLED u^0 ~ p(u|cond) (design Q1/Q3
      "sampled-style u^0"); action loss is applied at the target-u anchor
      (u_target, design Q3-A) AND at every refine round (deep supervision, S10 #2).
  (d) gradient routing (design Q2): the generator (incl. its encoder) is FROZEN;
      the flow, refine op and action head train. At F=identity the action
      gradient reaches the head + refine op, NOT the flow (u_target = e_target is
      frozen and the refine u^0 is a no-grad sample) -- the action<->flow
      end-to-end coupling turns on when F becomes a real warp (S8 TODO).
  (e) SINGLE-STAGE joint training here; the two-stage curriculum (S7: density
      first, then joint) is deferred.
"""
from __future__ import annotations

import torch
from torch import nn

from lacot.heads import DiscretizedActionHead
from lacot.nf_head import Flow


class RefineOperator(nn.Module):
    """Round-level refine op: u^{r+1} = LayerNorm(u^r + f(cond, u^r)), identity-init.

    The residual gives a stable identity start; the per-token LayerNorm keeps u on
    a UNIT-SCALE manifold every round, so the iterate cannot explode past the
    trained depth (a bare residual MLP does -- caught by refine_consistency_smoke).
    Mirrors the real design, where u lives right after the pooler's out_norm
    LayerNorm, i.e. is unit-scale by construction. This minimal operator stands in
    for the conditioning-backbone + NF-head operator (design S10 #1) until that
    wiring is finalized.
    """

    def __init__(self, cond_dim: int, k: int, d: int, hidden: int = 128):
        super().__init__()
        self.k, self.d = k, d
        self.net = nn.Sequential(
            nn.Linear(cond_dim + k * d, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, k * d),
        )
        self.norm = nn.LayerNorm(d)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, cond: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        b = u.shape[0]
        delta = self.net(torch.cat([cond, u.reshape(b, -1)], dim=-1)).reshape(b, self.k, self.d)
        return self.norm(u + delta)


class LaCoTActor(nn.Module):
    """Compose the M4 blocks into one model (training + inference paths).

    The `generator` must already be reconstruction-pretrained and FROZEN before it
    is handed in (its encoder is reused, frozen, for the (s,g) conditioning).
    """

    def __init__(
        self,
        generator: nn.Module,   # FROZEN ETargetGenerator (also provides the cond encoder)
        encoder_out: int,
        d_model: int,
        k: int,
        action_dim: int,
        chunk_len: int,
        num_bins: int = 256,
        n_flow_blocks: int = 4,
        refine_hidden: int = 128,
    ):
        super().__init__()
        self.generator = generator
        self.k = k
        self.d_model = d_model
        self.cond_dim = 2 * encoder_out  # concat(enc(s), enc(g))
        self.flow = Flow(token_dim=d_model, seq_len=k, n_blocks=n_flow_blocks, cond_dim=self.cond_dim)
        self.refine = RefineOperator(self.cond_dim, k, d_model, refine_hidden)
        self.action_head = DiscretizedActionHead(k * d_model, action_dim, chunk_len, num_bins)

    def encode_cond(self, s_frame: torch.Tensor, g_frame: torch.Tensor) -> torch.Tensor:
        """(s,g) frames [B,3,H,W] -> cond [B, 2*encoder_out] via the FROZEN encoder."""
        enc = self.generator.encoder
        return torch.cat([enc(s_frame), enc(g_frame)], dim=-1)

    def e_target(self, future_frames: torch.Tensor) -> torch.Tensor:
        """Future frames [B,T,3,H,W] -> e_target [B,K,d_model] (FROZEN; F=identity so u=e_target)."""
        return self.generator(future_frames)

    def refine_rounds(self, cond: torch.Tensor, u0: torch.Tensor, rounds: int) -> list[torch.Tensor]:
        us = [u0]
        u = u0
        for _ in range(rounds):
            u = self.refine(cond, u)
            us.append(u)
        return us

    def losses_given(
        self,
        cond: torch.Tensor,
        u_target: torch.Tensor,
        actions: torch.Tensor,
        rounds: int = 3,
        lam_cons: float = 0.5,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """All three M4 losses given PRECOMPUTED (frozen) cond + u_target.

        Split out so callers can cache the frozen front-end (encoder + generator)
        for a fixed batch instead of re-encoding every step.
        """
        b = cond.shape[0]
        l_nf = self.flow.nll(u_target, cond)          # density directly on e_target
        l_act_anchor = self.action_head.nll(          # decode action from the clean target-u (Q3-A)
            self.action_head(u_target.reshape(b, -1)), actions).mean()

        u0 = self.flow.sample(b, cond).detach()       # sampled-style u^0 (Q1/Q3), no-grad
        us = self.refine_rounds(cond, u0, rounds)
        l_cons = u_target.new_zeros(())
        l_act_refine = u_target.new_zeros(())
        for r in range(rounds):
            l_cons = l_cons + (us[r] - us[r + 1].detach()).pow(2).mean()  # ||u^r - sg(u^{r+1})||^2
            l_act_refine = l_act_refine + self.action_head.nll(          # deep supervision every round
                self.action_head(us[r + 1].reshape(b, -1)), actions).mean()
        l_cons = l_cons / rounds
        l_act_refine = l_act_refine / rounds

        total = l_nf + l_act_anchor + l_act_refine + lam_cons * l_cons
        return total, {
            "l_nf": l_nf.item(),
            "l_act_anchor": l_act_anchor.item(),
            "l_act_refine": l_act_refine.item(),
            "l_cons": l_cons.item(),
        }

    def training_losses(
        self,
        s_frame: torch.Tensor,
        g_frame: torch.Tensor,
        future_frames: torch.Tensor,
        actions: torch.Tensor,
        rounds: int = 3,
        lam_cons: float = 0.5,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """All three M4 losses on one raw batch (encodes cond + e_target, then delegates)."""
        cond = self.encode_cond(s_frame, g_frame)     # frozen encoder
        u_target = self.e_target(future_frames)       # frozen; F=identity -> u=e_target
        return self.losses_given(cond, u_target, actions, rounds, lam_cons)

    @torch.no_grad()
    def infer_action(
        self, s_frame: torch.Tensor, g_frame: torch.Tensor, rounds: int = 3, mode: str = "greedy"
    ) -> torch.Tensor:
        """Inference path (design Q1: NO future, NO e_target): (s,g) -> action chunk.

        cond = enc(s,g) -> u^0 ~ p(u|cond) -> refine -> u* -> decode action.
        """
        b = s_frame.shape[0]
        cond = self.encode_cond(s_frame, g_frame)
        u0 = self.flow.sample(b, cond)
        us = self.refine_rounds(cond, u0, rounds)
        return self.action_head.decode(self.action_head(us[-1].reshape(b, -1)), mode=mode)
