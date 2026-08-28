"""LaCoT full-model wiring (build order S5 step 5): compose the validated blocks.

Design doc: docs/LaCoT-NF-latent-planning-design.md (S3 losses, S4 components, S5
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

from lacot.e_target import PerceiverPooler
from lacot.heads import ContinuousActionHead, DiscretizedActionHead
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
    """Compose the LaCoT blocks into one model (training + inference paths).

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
        """All three LaCoT losses given PRECOMPUTED (frozen) cond + u_target.

        Split out so callers can cache the frozen front-end (encoder + generator)
        for a fixed batch instead of re-encoding every step.
        """
        b = cond.shape[0]
        # ⚠️⚠️ 2026-08-23 未修：這個（影像版）的 action_head 仍然【只吃 u】。
        # state 版已改成吃 [cond, u]，理由見 ContinuousActionHead 的 docstring
        # （u 是未來軌跡的壓縮表徵，動作要回答「從現在這個位置往哪走」）。
        # 影像版還沒在真資料上跑過，改動前先確認 cond_dim = 2*encoder_out 的寬度。
        # ⚠️ 2026-08-23: flow.nll 是【整條 u】的 nats，量級 ~ k*d_model 倍於逐元素 loss。
        # 未正規化時實測 l_nf ≈ -1479 而 l_act_anchor ≈ 5.0 —— action head 的梯度被
        # 完全淹沒（訓練 2000 步後 anchor 4.98 仍【差於】action 的邊際熵 4.7736，
        # 也就是比「什麼都不學」還糟）。除以維度讓三個 loss 回到同一個量級。
        l_nf = self.flow.nll(u_target, cond) / (self.k * self.d_model)
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
        """All three LaCoT losses on one raw batch (encodes cond + e_target, then delegates)."""
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


# ---------------------------------------------------------------------------
# State-based front-end (2026-08-23)
# ---------------------------------------------------------------------------
# `LaCoTActor` above takes image frames ([B,3,H,W]) — that is why the 08-22
# experiment scripts wrote their own MLP head instead of using it, and why every
# number from that day was measured on a *stand-in* head (continuous MSE) rather
# than LaCoT's real DiscretizedActionHead (classification over bins).
#
# This variant keeps every real LaCoT component — Flow, RefineOperator,
# DiscretizedActionHead — and only swaps the frozen image front-end for a
# low-dimensional state encoder, so the state track can be measured on the
# actual model instead of a replica.


def _mlp(i: int, h: int, o: int, n: int = 2) -> nn.Module:
    layers: list[nn.Module] = []
    p = i
    for _ in range(n):
        lin = nn.Linear(p, h)
        nn.init.xavier_uniform_(lin.weight)
        nn.init.zeros_(lin.bias)
        layers += [lin, nn.GELU(), nn.LayerNorm(h)]
        p = h
    lin = nn.Linear(p, o)
    nn.init.xavier_uniform_(lin.weight)
    nn.init.zeros_(lin.bias)
    return nn.Sequential(*layers, lin)


class LaCoTActorState(nn.Module):
    """LaCoT over low-dimensional states (e.g. pointmaze xy).

    Same three losses and the same real components as `LaCoTActor`; only the
    front-end differs. The trajectory encoder + pooler (which produce e_target)
    are trained separately by a contrastive stage and then FROZEN, matching the
    "generator is pretrained and frozen" contract of the image version.
    """

    def __init__(
        self,
        state_dim: int,
        d_model: int,
        k: int,
        action_dim: int,
        chunk_len: int,
        enc_hidden: int = 512,
        enc_out: int = 512,
        cond_dim: int = 256,
        head: str = "continuous",   # "continuous" (實測較優) | "discretized"
        num_bins: int = 32,         # 只在 head="discretized" 時用；32 實測優於 256
        n_flow_blocks: int = 4,
        refine_hidden: int = 256,
    ):
        super().__init__()
        self.k, self.d_model, self.cond_dim = k, d_model, cond_dim
        # frozen-after-stage-1 front end (produces e_target)
        self.traj_enc = _mlp(state_dim, enc_hidden, enc_out)
        self.e_pooler = PerceiverPooler(enc_out, d_model, k, 2, 4)
        # (s,g) conditioning
        self.cond_enc = _mlp(state_dim, enc_hidden, enc_out)
        self.cond_head = _mlp(2 * enc_out, enc_hidden, cond_dim)
        # the real LaCoT pieces
        self.flow = Flow(token_dim=d_model, seq_len=k, n_blocks=n_flow_blocks, cond_dim=cond_dim)
        self.refine = RefineOperator(cond_dim, k, d_model, refine_hidden)
        # ⛔ 預設走連續 head：離散 head 在 pointmaze 上實測比「猜資料平均」還差，
        # 就算把容量補到一樣也是（見 ContinuousActionHead 的 docstring）。
        # ⚠️ head 吃 [cond, u] 兩者 —— cond 給精確的 (s,g)、u 給規劃。
        # 只餵 u 會讓 K 小的時候位置資訊被壓掉，實測 ORACLE 從 100% 掉到 12%。
        head_in = cond_dim + k * d_model
        self.action_head = (
            ContinuousActionHead(head_in, action_dim, chunk_len)
            if head == "continuous"
            else DiscretizedActionHead(head_in, action_dim, chunk_len, num_bins)
        )
        self.head_kind = head

    # -- front end ---------------------------------------------------------
    def e_target(self, traj: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        """traj [B,T,state_dim] -> e_target [B,K,d_model]."""
        b, t, _ = traj.shape
        feats = self.traj_enc(traj.reshape(b * t, -1)).reshape(b, t, -1)
        return self.e_pooler(feats, key_padding_mask=key_padding_mask)

    def encode_cond(self, s: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """s,g [B,state_dim] -> cond [B,cond_dim]."""
        return self.cond_head(torch.cat([self.cond_enc(s), self.cond_enc(g)], dim=-1))

    def freeze_front_end(self) -> None:
        """Freeze the e_target producer — the image version's generator is frozen too."""
        for m in (self.traj_enc, self.e_pooler):
            m.eval()
            for p in m.parameters():
                p.requires_grad_(False)

    # -- inference ---------------------------------------------------------
    @torch.no_grad()
    def act(self, cond: torch.Tensor, rounds: int = 0, u: torch.Tensor | None = None,
            temperature: float = 1.0) -> torch.Tensor:
        """cond [B,cond_dim] -> action chunk [B,chunk_len,action_dim] in [-1,1].

        `u=None` draws from the flow (at `temperature`); pass `u` to inject a
        specific latent (e.g. the oracle e_target).
        """
        if u is None:
            u = self.sample_u(cond, temperature)
        for _ in range(rounds):
            u = self.refine(cond, u)
        out = self.action_head(torch.cat([cond, u.reshape(u.shape[0], -1)], dim=-1))
        if self.head_kind == "continuous":
            return out.clamp(-1.0, 1.0)
        return self.action_head.decode_bins(out.argmax(-1))

    @torch.no_grad()
    def sample_u(self, cond: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """Draw u ~ p(.|cond); `temperature` scales the base Gaussian (0 = the mode)."""
        z = torch.randn(cond.shape[0], self.k, self.d_model, device=cond.device) * temperature
        u = z
        for i in reversed(range(len(self.flow.blocks))):
            if i < len(self.flow.blocks) - 1:
                u = self.flow.perm.inverse(u)
            u = self.flow.blocks[i].inverse(u, cond)
        return u

    # -- training ----------------------------------------------------------
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
        """The same three losses as `LaCoTActor.losses_given`, on the state front-end.

        Kept as its own copy rather than inherited because the two classes build
        different front-ends; the loss bodies touch only flow / refine / action_head,
        which are identical objects in both.
        """
        b = cond.shape[0]
        l_nf = self.flow.nll(u_target, cond) / (self.k * self.d_model)   # 見上：量級正規化
        cat = lambda uu: torch.cat([cond, uu.reshape(b, -1)], dim=-1)   # head 吃 [cond, u]
        l_act_anchor = self.action_head.nll(
            self.action_head(cat(u_target)), actions).mean()

        u0 = self.flow.sample(b, cond).detach()
        us = self.refine_rounds(cond, u0, rounds)
        l_cons = u_target.new_zeros(())
        l_act_refine = u_target.new_zeros(())
        for r in range(rounds):
            l_cons = l_cons + (us[r] - us[r + 1].detach()).pow(2).mean()
            l_act_refine = l_act_refine + self.action_head.nll(
                self.action_head(cat(us[r + 1])), actions).mean()
        l_cons = l_cons / rounds
        l_act_refine = l_act_refine / rounds

        total = l_nf + l_act_anchor + l_act_refine + lam_cons * l_cons
        return total, {
            "l_nf": l_nf.item(),
            "l_act_anchor": l_act_anchor.item(),
            "l_act_refine": l_act_refine.item(),
            "l_cons": l_cons.item(),
        }
