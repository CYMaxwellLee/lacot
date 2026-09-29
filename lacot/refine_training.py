"""One loss composition for image, state and scratch training.

Adapter semantics follow proposal-v2 C6. Services consolidate its frozen teacher,
quality_factory and same-plan exposure dependencies, and fail closed before any
sample. There is deliberately no default oracle: experiment 1 did not qualify Aω.
"""
from __future__ import annotations

from dataclasses import dataclass
import copy
import math
from typing import Callable

import torch
from torch import nn

from lacot import refine_objective as objective
from lacot.refine_quality import QualityService

OBJECTIVE_VERSION = "generated-plan-v3"


@dataclass
class Adapter:
    density: Callable
    anchor: Callable
    sample: Callable
    features: Callable
    # Receives detached plan, returns (loss, validity_info) like exposure_loss.
    exposure: Callable | None = None


def assert_frozen(module, name):
    if not isinstance(module, nn.Module):
        raise ValueError(f"{name} must be an explicit frozen module")
    if any(m.training for m in module.modules()) or any(p.requires_grad for p in module.parameters()):
        raise ValueError(f"{name} must be frozen and eval")


@dataclass
class TrainingServices:
    teacher: nn.Module
    quality_service: QualityService
    quality_factory: Callable
    # exposure(plan, features): (loss, validity_info), matching exposure_loss.
    exposure: Callable | None
    # Explicit decoder/reference modules captured by callbacks, for freeze checks.
    frozen_modules: tuple[nn.Module, ...]
    horizon: int = 4
    jitter_std: float = 0.0
    noise_generator: torch.Generator | None = None
    ema_decay: float = .99
    successful_steps: int = 0
    attempted_steps: int = 0
    # Scratch binds raw state/goal/intent anchors each batch; actors receive an
    # already-bound service from their caller. No normalization/decoder invented here.
    batch_factory: Callable | None = None

    def authorize(self):
        if not isinstance(self.quality_service, QualityService):
            raise ValueError("qualified QualityService required for R>0")
        self.quality_service.authorize_training(self.horizon, same_plan=True)
        assert_frozen(self.teacher, "EMA teacher")
        assert_frozen(self.quality_service, "quality teacher")
        if not self.frozen_modules:
            raise ValueError("explicit frozen decoder modules required")
        for module in self.frozen_modules:
            assert_frozen(module, "decoder/reference")
        if self.exposure is None:
            raise ValueError("qualified same-plan action teacher/exposure unavailable; R>0 refused")
        if not callable(self.quality_factory) or not callable(self.exposure):
            raise ValueError("quality factory and same-plan exposure must be callable")
        if not math.isfinite(self.jitter_std) or self.jitter_std < 0:
            raise ValueError("jitter_std must be finite and nonnegative")
        if self.jitter_std and self.noise_generator is None:
            raise ValueError("auxiliary noise requires a dedicated generator")
        if not math.isfinite(self.ema_decay) or not 0 <= self.ema_decay < 1:
            raise ValueError("ema_decay must be in [0,1)")

    def noise_for(self, sampled):
        if not self.jitter_std:
            return None
        return torch.randn(sampled.shape, dtype=sampled.dtype, device=sampled.device,
                           generator=self.noise_generator) * self.jitter_std

    def for_batch(self, *, state, goal, anchors):
        if self.batch_factory is None:
            raise ValueError("scratch R>0 requires a same-plan batch service binder")
        bound = self.batch_factory(state=state, goal=goal, anchors=anchors)
        if (not isinstance(bound, TrainingServices) or bound.teacher is not self.teacher
                or bound.metadata() != self.metadata()
                or bound.noise_generator is not self.noise_generator):
            raise ValueError("batch service changed teacher/configuration/noise stream")
        return bound

    def metadata(self):
        return dict(objective_version=OBJECTIVE_VERSION, Rtrain=[0, 1, 2, 3], cons="ema",
                    teacher_fingerprint=self.quality_service.calibration.teacher_fingerprint,
                    horizon=self.horizon, jitter_std=self.jitter_std, ema_decay=self.ema_decay)

    def checkpoint_state(self):
        return dict(metadata=self.metadata(), teacher=copy.deepcopy(self.teacher.state_dict()),
                    successful_steps=self.successful_steps, attempted_steps=self.attempted_steps,
                    noise_rng=None if self.noise_generator is None else self.noise_generator.get_state())

    def restore(self, state):
        self.authorize()
        if not state or state.get("metadata") != self.metadata():
            raise ValueError("refine resume metadata missing/mismatched; explicit migration required")
        if (state.get("noise_rng") is None) != (self.noise_generator is None):
            raise ValueError("refine resume noise generator mismatch")
        self.teacher.load_state_dict(state["teacher"], strict=True)
        self.successful_steps = state["successful_steps"]
        self.attempted_steps = state["attempted_steps"]
        if self.noise_generator is not None:
            self.noise_generator.set_state(state["noise_rng"])

    def after_step(self, student, *, succeeded):
        self.attempted_steps += 1
        if succeeded:
            update_ema(self.teacher, student, self.ema_decay)
            self.successful_steps += 1


def require_services(services):
    if not isinstance(services, TrainingServices):
        raise ValueError("R>0 requires qualified refine training_services; Aω/exposure not available by default")
    services.authorize()
    return services


def compose(adapter, refine, cond, clean, *, rounds, training_services=None, noise=None, lam_cons=.1):
    """Return base_total, tensor logs, own states; R0 is exactly nf + anchor."""
    objective.validate_rounds(rounds)
    objective.validate_context(cond, clean)
    if not isinstance(lam_cons, (int, float)) or not math.isfinite(lam_cons) or lam_cons < 0:
        raise ValueError("lam_cons must be finite and nonnegative")
    services = require_services(training_services) if rounds else None
    nf = adapter.density()
    anchor = adapter.anchor(adapter.features(clean.detach()))
    zero = nf.new_zeros(())
    logs = dict(l_nf=nf, l_act_anchor=anchor, l_act_refine=zero,
                l_plan_quality=zero, l_cons=zero, l_head_exposure=zero,
                quality_valid_fraction=zero, exposure_valid_fraction=zero)
    if not rounds:
        return nf + anchor, logs, []
    sampled = adapter.sample().detach()
    quality = services.quality_factory(sampled, cond)
    if noise is None:
        noise = services.noise_for(sampled)
    terms = objective.refinement_terms(refine, services.teacher, quality, cond, clean,
                                       sampled, rounds=rounds, noise=noise)
    fingerprint = services.quality_service.calibration.teacher_fingerprint
    if any(r.teacher_fingerprint != fingerprint for r in terms.reports):
        raise ValueError("QualityReport teacher fingerprint mismatch")
    exposure_fn = adapter.exposure
    if exposure_fn is None:
        exposure_fn = lambda u: services.exposure(u, adapter.features(u))
    exposure_terms, exposure_coverage = [], []
    for u in terms.states[1:]:
        loss, info = exposure_fn(u.detach())
        if (not isinstance(loss, torch.Tensor) or loss.ndim != 0
                or loss.device != clean.device or not loss.is_floating_point()):
            raise ValueError("exposure loss must be a floating scalar on the latent device")
        if not info.get("valid") or info.get("n_valid", 0) <= 0:
            raise ValueError("same-plan exposure invalid: " + str(info.get("reason", "no valid labels")))
        if info.get("n_total") != len(u) or info["n_valid"] > len(u):
            raise ValueError("same-plan exposure coverage mismatch")
        exposure_terms.append(loss)
        exposure_coverage.append(info["n_valid"] / info["n_total"])
    exposure = torch.stack(exposure_terms).mean()
    total = nf + anchor + terms.quality + lam_cons * terms.consistency + .25 * exposure
    logs.update(l_plan_quality=terms.quality, l_cons=terms.consistency,
                l_head_exposure=exposure,
                exposure_valid_fraction=nf.new_tensor(sum(exposure_coverage) / len(exposure_coverage)),
                quality_valid_fraction=torch.stack([r.valid.float().mean() for r in terms.reports]).mean())
    return total, logs, terms.states


def optimizer_step(optimizer, scaler=None):
    """Detect this optimizer's actual step, even when another optimizer AMP-skips.

    GradScaler.step returns the optimizer's return value (usually None), and its
    global scale cannot identify *which* optimizer skipped. A public post hook can.
    """
    stepped = []
    handle = optimizer.register_step_post_hook(lambda *args, **kwargs: stepped.append(True))
    try:
        if scaler is None:
            optimizer.step()
        else:
            scaler.step(optimizer)
    finally:
        handle.remove()
    return bool(stepped)


@torch.no_grad()
def update_ema(teacher, student, decay):
    parameters, buffers = dict(student.named_parameters()), dict(student.named_buffers())
    for name, target in teacher.named_parameters():
        target.mul_(decay).add_(parameters[name], alpha=1 - decay)
    for name, target in teacher.named_buffers():
        target.copy_(buffers[name])
