"""Generated-plan objective (DESIGN-v3 / proposal-v2 C1--C4).

Only the sample and teacher targets stop gradients. Student iterates and cond
keep their entire graph. Clean is a shape/dtype reference, never a quality target.
"""
from dataclasses import dataclass
from typing import Callable

import torch

from lacot.refine_quality import QualityReport


def validate_rounds(rounds):
    if type(rounds) is not int or rounds < 0:
        raise ValueError("rounds must be a nonnegative integer (not bool)")


def validate_context(cond, clean):
    if not isinstance(cond, torch.Tensor) or not isinstance(clean, torch.Tensor):
        raise ValueError("cond and clean must be tensors")
    if (clean.ndim != 3 or cond.ndim != 2 or min(clean.shape) < 1
            or min(cond.shape) < 1 or cond.shape[0] != clean.shape[0]):
        raise ValueError("expected cond[B,C], latent[B,K,D] with positive dimensions")
    if any(not x.is_floating_point() or x.device != clean.device for x in (cond, clean)):
        raise ValueError("floating tensors on the same device required")


def validate(cond, clean, sampled, noise, rounds):
    validate_rounds(rounds)
    validate_context(cond, clean)
    latents = (clean, sampled) if noise is None else (clean, sampled, noise)
    if any(not isinstance(x, torch.Tensor) for x in latents):
        raise ValueError("latent inputs must be tensors")
    if any(x.shape != clean.shape or x.dtype != clean.dtype for x in latents):
        raise ValueError("latent shape/dtype mismatch")
    if any(not x.is_floating_point() or x.device != clean.device for x in (*latents, cond)):
        raise ValueError("floating tensors on the same device required")
    # cond may have a different dtype. No hot-path finiteness scans: AMP owns skips.


def reduction_value(value):
    return value if value.dtype == torch.float64 else value.float()


def mean_valid(report, reference):
    if not isinstance(report, QualityReport):
        raise ValueError("quality must return refine_quality.QualityReport")
    if report.cost.shape != (len(reference),) or report.cost.device != reference.device:
        raise ValueError("quality batch/device mismatch")
    if not bool(report.valid.any()):
        raise ValueError("all candidates invalid; refusing quality supervision")
    return report.cost[report.valid].mean()


@dataclass
class Terms:
    quality: torch.Tensor
    consistency: torch.Tensor
    states: list[torch.Tensor]
    reports: list[QualityReport]
    auxiliary_states: list[torch.Tensor]


def refinement_terms(refine, teacher, quality: Callable, cond, clean, sampled, *, rounds, noise=None):
    """Main arm uses fresh flow; optional .25-weight auxiliary arm adds noise.

    Teacher parameters are checked by the training service. The kernel also
    stops targets explicitly, even for a callable used in a CPU wiring witness.
    """
    validate(cond, clean, sampled, noise, rounds)
    u = sampled.detach()
    v = u + noise.detach() if noise is not None else None
    q = clean.new_zeros(())
    c = clean.new_zeros(())
    states, reports, auxiliary = [u], [], [] if v is None else [v]
    for _ in range(rounds):
        un = refine(cond, u)
        if un.shape != clean.shape:
            raise ValueError("refine shape mismatch")
        report = quality(un)
        qr = mean_valid(report, un)
        reports.append(report)
        if v is not None:
            v = refine(cond, v)
            if v.shape != clean.shape:
                raise ValueError("auxiliary refine shape mismatch")
            vr = quality(v)
            qr = (qr + .25 * mean_valid(vr, v)) / 1.25
            reports.append(vr)
            auxiliary.append(v)
        with torch.no_grad():
            target = teacher(cond.detach(), u.detach())
        if target.shape != un.shape:
            raise ValueError("teacher shape mismatch")
        q = q + qr
        c = c + (reduction_value(un) - reduction_value(target)).square().mean()
        u = un
        states.append(u)
    return Terms(q / max(rounds, 1), c / max(rounds, 1), states, reports, auxiliary)
