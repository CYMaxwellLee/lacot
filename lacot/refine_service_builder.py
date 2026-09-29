"""Exp1 checkpoint -> batch-bound quality half of TrainingServices.

Loading W_phi is not permission to train: A_omega/exposure is still unavailable.
The existing three qualification checks and generated gates remain authoritative
in QualityService.authorize_training. No environment/simulator is imported here.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field, replace
import math
import os
from pathlib import Path
import pickle

import torch

from lacot.refine_models import (VERSION, HORIZONS, Normalizer, ConsequenceEnsemble,
                                 PairedController, freeze, subspaces)
from lacot.refine_quality import (Calibration, OfflineCalibrator, QualityService,
                                  qualification_reasons, readings_digest)
from lacot.refine_training import TrainingServices, assert_frozen


def _finite_number(value, name, *, nonnegative=False):
    if (type(value) not in (int, float) or not math.isfinite(value)
            or (nonnegative and value < 0)):
        raise ValueError(f"invalid qualification reading: {name}")


def _check_readings(service, horizon):
    """Check W's two revised gates; record, never waive, A/generated blockers."""
    cal, readings = service.calibration, service.readings
    for name in ("teacher_fingerprint", "split_hash", "source_kind"):
        if readings.get(name) != getattr(cal, name):
            raise ValueError(f"qualification {name} mismatch")
    if readings_digest(readings) != service.calibrator._readings_digest:
        raise ValueError("checkpoint qualification readings digest mismatch")
    info = readings["horizons"].get(str(horizon), {})
    if info.get("available") is not True:
        raise ValueError("W qualification: horizon unavailable")
    expected = subspaces(cal.domain, service.world.dim)
    if set(info["subspaces"]) != set(expected):
        raise ValueError("W qualification: missing/unknown subspaces")
    for name, metric in info["subspaces"].items():
        for key in ("world_nmse", "obs_only_nmse"):
            _finite_number(metric[key], name + ":" + key, nonnegative=True)
        ci = metric["paired_improvement_ci95"]
        if len(ci) != 2:
            raise ValueError("invalid paired CI")
        for value in ci:
            _finite_number(value, name + ":paired CI")
        if ci[0] > ci[1]:
            raise ValueError("invalid paired CI order")
        if metric["world_nmse"] > .9 * metric["obs_only_nmse"]:
            raise ValueError(f"W qualification: {name}: NMSE vs obs-only")
        if ci[0] <= 0:
            raise ValueError(f"W qualification: {name}: paired CI")
    _finite_number(readings["controller_action_nmse"], "A action NMSE", nonnegative=True)
    for key in ("validity", "nominal_plus_alternative_coverage"):
        value = readings["generated_candidates"].get(key)
        if value is not None:
            _finite_number(value, "generated " + key, nonnegative=True)
            if value > 1:
                raise ValueError("invalid generated coverage")
    return qualification_reasons(readings, horizon, require_controller=True)


def load_quality_teacher(path, *, domain, horizon=4, device="cpu",
                         expected_fingerprint=None, diagnostic_cpu=False):
    """Read exp1's weights-only artifact on CPU and verify before device transfer.

    diagnostic_cpu is a Python-only test option: toy provenance is retained and
    authorize_training still rejects it. It cannot admit failing W measurements.
    """
    if domain not in ("pointmaze", "ant") or horizon not in HORIZONS:
        raise ValueError("unsupported quality domain/horizon")
    device = torch.device(device)
    if diagnostic_cpu and device.type != "cpu":
        raise ValueError("diagnostic_cpu requires CPU")
    if not path or not str(path).strip():
        raise ValueError("LACOT_REFINE_QUALITY_CKPT is required")
    try:
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if saved["version"] != VERSION:
            raise ValueError("checkpoint representation version mismatch")
        if saved["metadata"]["domain"] != domain:
            raise ValueError("checkpoint domain mismatch")
        source = saved["metadata"]["source_kind"]
        if source != "offline_dataset" and not (diagnostic_cpu and source == "toy_offline"):
            raise ValueError("toy/unverified checkpoint cannot qualify production")
        if saved["metadata"]["episode_split"] != "train" or saved["metadata"]["eval_only"] is not False:
            raise ValueError("checkpoint must originate from offline train split")
        w = saved["models"]["world"]
        # Model constructors initialize weights before load; don't perturb training RNG.
        with torch.random.fork_rng(devices=[]):
            sn = Normalizer(w["state_norm.mean"], w["state_norm.scale"])
            an = Normalizer(w["action_norm.mean"], w["action_norm.scale"])
            models = {
                "world": ConsequenceEnsemble(sn, an, hidden=saved["hidden"]),
                "obs_only": ConsequenceEnsemble(sn, an, hidden=saved["hidden"], obs_only=True),
                "controller": PairedController(sn, an, saved["t_cap"], hidden=saved["hidden"]),
            }
        for name, model in models.items():
            model.load_state_dict(saved["models"][name], strict=True)
            freeze(model)
            if any(not torch.isfinite(t).all() for t in model.state_dict().values()):
                raise ValueError("nonfinite checkpoint weights/normalizers")
        cal = OfflineCalibrator.__new__(OfflineCalibrator)
        cal.models, cal.world = models, models["world"]
        cal.train = {"metadata": saved["metadata"]}
        cal.banks = saved["support_banks"]
        cal.reference_nll = cal.reference_fingerprint = None
        cal.artifact = Calibration(**saved["calibration"])
        cal._readings_digest = saved["readings_digest"]
        if cal.artifact.reference_fingerprint is not None:
            raise ValueError("reference flow requires an explicit frozen adapter")
        if cal.artifact.version != "oracle-calibration-v1":
            raise ValueError("checkpoint calibration version mismatch")
        for name in ("dataset_hash", "split_hash", "teacher_fingerprint"):
            value = getattr(cal.artifact, name)
            if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError(f"invalid checkpoint {name}")
        # Existing service verifies joint model/normalizer/support/calibration fingerprint.
        service = QualityService(cal, readings=saved["readings"],
                                 diagnostic=diagnostic_cpu,
                                 expected_fingerprint=expected_fingerprint)
        _check_readings(service, horizon)
        threshold = cal.artifact.thresholds[str(horizon)]
        for key in ("support_p95", "disagreement_p95"):
            _finite_number(threshold[key], key, nonnegative=True)
        for key in ("arrival", "hold", "behavior_support"):
            value = cal.artifact.scales[str(horizon)][key]
            _finite_number(value, key)
            if value <= 0:
                raise ValueError("calibration scales must be positive")
        bank = cal.banks[horizon]
        if (bank.ndim != 2 or not len(bank) or
                bank.shape[1] != service.world.dim + horizon * service.world.adim
                or not torch.isfinite(bank).all()):
            raise ValueError("invalid support bank")
        for model in models.values():
            model.to(device=device, dtype=torch.float32)
        cal.banks = {h: bank.to(device=device, dtype=torch.float32) for h, bank in cal.banks.items()}
        service.to(device=device, dtype=torch.float32)
        assert_frozen(service, "quality teacher")
        return service
    except (OSError, EOFError, pickle.UnpicklingError, KeyError, TypeError, IndexError, RuntimeError) as exc:
        raise ValueError(f"invalid quality checkpoint: {exc}") from exc


@dataclass
class QualityTrainingServices(TrainingServices):
    """Extend checkpoint metadata without changing exp2 authorization/compose."""
    quality_provenance: dict = field(default_factory=dict)

    def metadata(self):
        return {**super().metadata(), "quality_teacher": copy.deepcopy(self.quality_provenance)}


def build_training_services(*, checkpoint, domain, teacher, decoder, intent_inverse,
                            frozen_modules, state_mean, state_scale, t_cap,
                            device="cpu", horizon=4, ema_decay=.99,
                            expected_fingerprint=None, diagnostic_cpu=False):
    """Bind normalized scratch batches; decoder is the original _dec(_q(u), s).

    Each predicted raw state is re-normalized for that decoder. Path/goal are
    converted back to raw XY for A/W. All constants detach; u's graph stays live.
    No exposure labels or authorization override are supplied by this builder.
    """
    assert_frozen(teacher, "EMA teacher")
    if not frozen_modules or not callable(decoder) or not callable(intent_inverse):
        raise ValueError("explicit frozen decoder callbacks/modules required")
    for module in frozen_modules:
        assert_frozen(module, "decoder/reference")
    service = load_quality_teacher(checkpoint, domain=domain, horizon=horizon, device=device,
                                   expected_fingerprint=expected_fingerprint,
                                   diagnostic_cpu=diagnostic_cpu)
    if t_cap != service.controller.t_cap:
        raise ValueError("decoder/checkpoint T_CAP mismatch")
    mean = torch.as_tensor(state_mean, device=device, dtype=torch.float32).detach().clone()
    scale = torch.as_tensor(state_scale, device=device, dtype=torch.float32).detach().clone()
    if (mean.shape != (service.world.dim,) or scale.shape != mean.shape
            or not torch.isfinite(mean).all() or not torch.isfinite(scale).all() or not (scale > 0).all()):
        raise ValueError("scratch/oracle state normalization mismatch")
    cal = service.calibration
    provenance = dict(checkpoint=str(Path(checkpoint).resolve()), domain=domain,
                      representation_version=cal.representation_version,
                      teacher_fingerprint=cal.teacher_fingerprint, split_hash=cal.split_hash,
                      dataset_hash=cal.dataset_hash, source_kind=cal.source_kind,
                      readings_digest=service.calibrator._readings_digest,
                      frozen=True, eval=True, diagnostic_cpu=diagnostic_cpu,
                      world_qualification=("diagnostic_numeric_gates_passed" if diagnostic_cpu else "passed"),
                      exposure_available=False,
                      training_blocks=_check_readings(service, horizon))

    def unbound(*args):
        raise ValueError("quality requires for_batch(state, goal, anchors)")

    result = QualityTrainingServices(teacher=teacher, quality_service=service,
        quality_factory=unbound, exposure=None, frozen_modules=tuple(frozen_modules),
        horizon=horizon, ema_decay=ema_decay, quality_provenance=provenance)

    def bind(*, state, goal, anchors):
        if domain != "pointmaze":
            raise ValueError("ant R>0 unsupported")
        if (state.ndim != 2 or state.shape[1] != service.world.dim or not len(state)
                or goal.ndim != 2 or len(goal) != len(state) or goal.shape[1] not in (2, service.world.dim)
                or state.device != mean.device or goal.device != mean.device
                or not torch.isfinite(state).all() or not torch.isfinite(goal).all()):
            raise ValueError("batch state/goal shape/device/finite mismatch")
        raw_state = state.detach().float() * scale + mean
        raw_goal = goal.detach().float()[:, :2] * scale[:2] + mean[:2]
        anc = None if anchors is None else anchors.detach().clone()

        def decode(u, current):
            normalized = (current.float() - mean) / scale
            path = intent_inverse(decoder(u, normalized), anc)
            return path.float() * scale[:2] + mean[:2]

        def factory(sampled, cond):
            if len(sampled) != len(state) or len(cond) != len(state):
                raise ValueError("quality factory batch mismatch")

            def quality(u):
                assert_frozen(service, "quality teacher")
                for module in frozen_modules:
                    assert_frozen(module, "decoder/reference")
                if len(u) != len(raw_state) or u.device != raw_state.device:
                    raise ValueError("quality latent batch/device mismatch")
                # No no_grad here. Explicit fp32 also protects cdist under AMP.
                with torch.autocast(device_type=u.device.type, enabled=False):
                    report = service.same_plan(u.float(), raw_state, raw_goal, decode, horizon=horizon)
                if not torch.isfinite(report.cost).all():
                    raise ValueError("nonfinite quality cost")
                if not bool(report.valid.any()):
                    raise ValueError("all candidates invalid; refusing quality supervision")
                return report

            return quality

        return replace(result, quality_factory=factory)

    result.batch_factory = bind
    return result


def build_from_env(*, environ=None, **kwargs):
    """Explicit opt-in caller; absent env fails closed, never chooses a default."""
    env = os.environ if environ is None else environ
    path = env.get("LACOT_REFINE_QUALITY_CKPT", "").strip()
    if not path:
        raise ValueError("LACOT_REFINE_QUALITY_CKPT is required")
    if kwargs.get("diagnostic_cpu"):
        raise ValueError("environment builder cannot enable diagnostic_cpu")
    return build_training_services(checkpoint=path, **kwargs)
