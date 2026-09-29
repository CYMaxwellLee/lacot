"""Frozen differentiable oracle services, offline calibration and qualification reads.

Extends the proposal-v2 quality(u)->[B]/same-plan exposure interfaces. Decoder
callbacks MUST wrap the caller's existing _dec/_intent_inv; this module never
invents another latent decoder. NLL is support, never an outcome/success label.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import copy
import hashlib
import json
from typing import Callable
import numpy as np
import torch
from torch import nn
from lacot.refine_models import (HORIZONS, VERSION, validate_record, assert_disjoint,
                                  subspaces, freeze, model_fingerprint, subset)


@dataclass
class QualityReport:
    cost: torch.Tensor
    components: dict[str, torch.Tensor]
    valid: torch.Tensor
    uncertainty: torch.Tensor
    support: torch.Tensor
    teacher_fingerprint: str

    def __post_init__(self):
        b = self.cost.shape
        if len(b) != 1 or self.cost.dtype != torch.float32 or self.valid.dtype != torch.bool:
            raise ValueError("QualityReport cost fp32[B], valid bool[B] required")
        for value in [self.valid, self.uncertainty, self.support, *self.components.values()]:
            if value.shape != b or value.device != self.cost.device:
                raise ValueError("QualityReport shape/device mismatch")
        if not self.components or not self.teacher_fingerprint:
            raise ValueError("QualityReport components/fingerprint required")

    def json(self):
        return {"cost": self.cost.detach().cpu().tolist(),
                "components": {k: v.detach().cpu().tolist() for k, v in self.components.items()},
                "valid": self.valid.detach().cpu().tolist(),
                "uncertainty": self.uncertainty.detach().cpu().tolist(),
                "support": self.support.detach().cpu().tolist(),
                "teacher_fingerprint": self.teacher_fingerprint}


def paired_ci(improvement, episodes, seed=0, draws=1000):
    """Episode-cluster bootstrap: overlapping windows aren't independent draws."""
    improvement = np.asarray(improvement, float); episodes = np.asarray(episodes)
    groups = [improvement[episodes == e] for e in np.unique(episodes)]
    if len(groups) < 2:
        return [None, None]
    sums = np.array([g.sum() for g in groups]); counts = np.array([len(g) for g in groups])
    idx = np.random.default_rng(seed).integers(len(groups), size=(draws, len(groups)))
    values = sums[idx].sum(1)/counts[idx].sum(1)
    return np.quantile(values, [.025, .975]).tolist()


def features(world, state, actions):
    return torch.cat([world.state_norm(state), world.action_norm(actions).flatten(1)], 1).float()


def neighbor_distance(query, bank):
    # Bound temporary distance matrices; gradients through query are preserved.
    return torch.cat([torch.cdist(q, bank.float()).amin(1)/query.shape[1]**.5
                      for q in query.float().split(128)])


def consequence_components(world, prediction, actions, goal, support):
    mean = prediction.float().mean(0)
    norm = world.state_norm.scale.float()
    distance = ((mean[..., :2]-goal[:, None].float())/norm[:2]).square().mean(-1)
    normalized = prediction.float()/norm
    components = {"arrival": distance[:, -1], "hold": distance[:, -min(4, distance.shape[1]):].mean(1),
                  "behavior_support": torch.nn.functional.softplus(support.float()),
                  "action_smoothness": ((actions[:, 1:].float()-actions[:, :-1].float())/world.action_norm.scale).square().mean((1, 2))}
    uncertainty = normalized.var(0, unbiased=False).mean((1, 2))
    components["ensemble_disagreement"] = uncertainty
    if mean.shape[-1] == 29:
        # Diagnostic magnitudes only; no claimed physical fall/freeze classifier.
        components["gait_displacement"] = ((mean[:, -1, [2, *range(7, 29)]]-mean[:, 0, [2, *range(7, 29)]])/norm[[2, *range(7, 29)]]).square().mean(1)
        components["quaternion_norm_error"] = (mean[..., 3:7].square().sum(-1)-1).square().mean(1)
    return components, uncertainty


@dataclass
class Calibration:
    version: str
    domain: str
    dataset_hash: str
    split_hash: str
    representation_version: str
    teacher_fingerprint: str
    source_kind: str
    source_split: str
    calibration_episodes: list[int]
    train_episodes: list[int]
    support_kind: str
    thresholds: dict
    scales: dict
    pi_radius: dict
    counts: dict
    reference_fingerprint: str | None
    controller_disagreement_p95: float

    def json(self):
        return asdict(self)


def calibration_fingerprint(calibrator, thresholds, scales, radii, controller_threshold):
    bank_hash = hashlib.sha256()
    for h, bank in sorted(calibrator.banks.items()):
        bank_hash.update(str(h).encode())
        bank_hash.update(bank.detach().cpu().contiguous().numpy().tobytes())
    reference_weights = (model_fingerprint({"reference": calibrator.reference_nll}, {})
                         if calibrator.reference_nll is not None else None)
    return model_fingerprint(calibrator.models, calibrator.train["metadata"],
        {"reference": calibrator.reference_fingerprint, "reference_weights": reference_weights,
         "version": VERSION, "support_bank_hash": bank_hash.hexdigest(),
         "thresholds": thresholds, "scales": scales, "pi_radius": radii,
         "controller_disagreement_p95": controller_threshold})


class OfflineCalibrator:
    """Single-use fit; support bank only from train, all cutoffs only calibration.

    Optional reference_nll(state, physical_actions) returns per-dimension NLL[B].
    It must be a frozen reference, identified by a content fingerprint.
    """
    def __init__(self, models, train, *, reference_nll=None, reference_fingerprint=None):
        validate_record(train, split="train", physical=True)
        if (reference_nll is None) != (reference_fingerprint is None):
            raise ValueError("reference NLL and fingerprint must be supplied together")
        if reference_nll is not None and not isinstance(reference_nll, nn.Module):
            raise ValueError("reference NLL must be a frozen nn.Module")
        self.models = models
        self.world = freeze(models["world"])
        self.train = train
        self.reference_nll = freeze(reference_nll) if reference_nll is not None else None
        self.reference_fingerprint = reference_fingerprint
        self.banks = {}
        for h in HORIZONS:
            ids = torch.where(train["mask"][:, :h].all(1))[0][:512]
            if not len(ids):
                continue
            self.banks[h] = features(self.world, train["state"][ids], train["actions"][ids, :h]).detach()
        self.artifact = None
        self._readings_digest = None

    def support(self, state, actions):
        if self.reference_nll is not None:
            result = self.reference_nll(state, actions).float()
            if result.shape != (len(state),):
                raise ValueError("reference must return per-dimension NLL[B]")
            return result
        return neighbor_distance(features(self.world, state, actions), self.banks[actions.shape[1]])

    @torch.no_grad()
    def fit(self, record):
        if self.artifact is not None:
            raise ValueError("calibration is locked; create a new version to recalibrate")
        validate_record(record, split="calibration", physical=True)
        assert_disjoint({"train": self.train, "calibration": record})
        thresholds = {}; scales = {}; radii = {}; counts = {}
        for h in HORIZONS:
            valid_rows = record["mask"][:, :h].all(1)
            counts[str(h)] = {"complete": int(valid_rows.sum()), "short_masked": int((~valid_rows).sum())}
            if not valid_rows.any() or h not in self.banks:
                counts[str(h)]["reason"] = "no complete train/calibration horizon"
                continue
            r = subset(record, valid_rows); a = r["actions"][:, :h]
            pred = self.world(r["state"], a)
            support = self.support(r["state"], a)
            comp, disagreement = consequence_components(self.world, pred, a, r["path"][:, -1], support)
            thresholds[str(h)] = {"support_p95": float(torch.quantile(support, .95)),
                                   "disagreement_p95": float(torch.quantile(disagreement, .95))}
            scales[str(h)] = {k: max(float(v.abs().mean()), 1e-6) for k, v in comp.items()}
            residual = ((pred.mean(0)-r["future"][:, :h])/self.world.state_norm.scale).abs()
            # Split conformal marginal 90% intervals, per physical timestep/dimension.
            q = min(1., np.ceil((len(residual)+1)*.9)/len(residual))
            radii[str(h)] = torch.quantile(residual, q, dim=0, interpolation="higher").tolist()
        action_predictions = self.models["controller"](record["state"], record["path"])
        controller_disagreement = (action_predictions/self.world.action_norm.scale).var(0, unbiased=False).mean((1, 2))
        controller_threshold = float(torch.quantile(controller_disagreement, .95))
        fingerprint = calibration_fingerprint(self, thresholds, scales, radii, controller_threshold)
        self.artifact = Calibration("oracle-calibration-v1", record["metadata"]["domain"],
            record["metadata"]["dataset_hash"], record["metadata"]["split_hash"],
            VERSION, fingerprint, record["metadata"]["source_kind"],
            "calibration", sorted(set(record["episode_id"].tolist())),
            sorted(set(self.train["episode_id"].tolist())),
            "reference_nll_per_dimension" if self.reference_nll is not None else "normalized_state_action_neighbor",
            thresholds, scales, radii, counts, self.reference_fingerprint, controller_threshold)
        return self.artifact

    @torch.no_grad()
    def evaluate(self, record, *, seed=0):
        validate_record(record, split="test", physical=True)
        if self.artifact is None:
            raise ValueError("calibrate before held-out evaluation")
        if (record["metadata"]["dataset_hash"] != self.artifact.dataset_hash or
            record["metadata"]["domain"] != self.artifact.domain or
            record["metadata"]["split_hash"] != self.artifact.split_hash):
            raise ValueError("held-out dataset/domain/split hash mismatch")
        ids = set(record["episode_id"].tolist())
        if ids & set(self.artifact.train_episodes + self.artifact.calibration_episodes):
            raise ValueError("held-out episode overlap")
        spaces = subspaces(self.artifact.domain, record["state"].shape[1]); output = {}
        for h in HORIZONS:
            mask = record["mask"][:, :h].all(1)
            if str(h) not in self.artifact.thresholds or not mask.any():
                output[str(h)] = {"available": False, "reason": "no complete calibration/test horizon"}
                continue
            r = subset(record, mask); a = r["actions"][:, :h]
            pred = self.world(r["state"], a).mean(0)
            obs_only = self.models["obs_only"](r["state"], a).mean(0)
            permutation = torch.randperm(len(a), generator=torch.Generator().manual_seed(seed+h))
            shuffled = self.world(r["state"], a[permutation]).mean(0)
            target = r["future"][:, :h]; norm = self.world.state_norm.scale
            errors = {"world": ((pred-target)/norm).square(), "obs_only": ((obs_only-target)/norm).square(),
                      "shuffled_action": ((shuffled-target)/norm).square(),
                      "constant": ((r["state"][:, None]-target)/norm).square()}
            radius = torch.tensor(self.artifact.pi_radius[str(h)], device=pred.device)
            covered = ((pred-target)/norm).abs() <= radius
            metrics = {}
            for name, dims in spaces.items():
                per_window = {k: v[:, :, dims].mean((1, 2)) for k, v in errors.items()}
                metric = {k+"_nmse": float(v.mean()) for k, v in per_window.items()}
                metric["paired_improvement_ci95"] = paired_ci((per_window["obs_only"]-per_window["world"]).cpu(), r["episode_id"].cpu(), seed)
                metric["shuffled_degradation_ci95"] = paired_ci((per_window["shuffled_action"]-per_window["world"]).cpu(), r["episode_id"].cpu(), seed)
                metric["pi90_coverage"] = float(covered[:, :, dims].float().mean())
                metric["pi90_width_normalized"] = float(2*radius[:, dims].mean())
                metric["pi90_width_raw"] = float((2*radius*norm)[:, dims].mean())
                metrics[name] = metric
            support = self.support(r["state"], a)
            _, uncertainty = consequence_components(self.world, self.world(r["state"], a), a, r["path"][:, -1], support)
            threshold = self.artifact.thresholds[str(h)]
            valid = (support <= threshold["support_p95"]) & (uncertainty <= threshold["disagreement_p95"])
            output[str(h)] = {"available": True, "subspaces": metrics, "data_validity": float(valid.float().mean()),
                              "n_total": len(record["state"]), "n_complete": len(a), "n_short_masked": int((~mask).sum()),
                              "reject_support": int((support > threshold["support_p95"]).sum()),
                              "reject_uncertainty": int((uncertainty > threshold["disagreement_p95"]).sum())}
        action_predictions = self.models["controller"](record["state"], record["path"])
        actions = action_predictions.mean(0)
        nmse = ((actions-record["actions"][:, :4])/self.world.action_norm.scale).square().mean()
        controller_disagreement = (action_predictions/self.world.action_norm.scale).var(0, unbiased=False).mean((1, 2))
        readings = {"horizons": output, "controller_action_nmse": float(nmse),
                "controller_disagreement_validity": float((controller_disagreement <= self.artifact.controller_disagreement_p95).float().mean()),
                "state_information": {"observation_dimensions": record["state"].shape[-1],
                                      "future_velocity_or_endpoint_time_as_input": False,
                                      "action_identifiability": "must_be_established_by_heldout_error"},
                "controller_target": "same_window_original_CHUNK_actions",
                "controller_ant_refine_supported": False,
                "generated_candidates": {"validity": None, "nominal_plus_alternative_coverage": None,
                                         "reason": "requires actual frozen flow/executor candidates; toy does not qualify"},
                "teacher_fingerprint": self.artifact.teacher_fingerprint,
                "split_hash": self.artifact.split_hash,
                "source_kind": self.artifact.source_kind,
                "known_limitation": "W_phi predicts all four states from all four actions in a chunk; s[t+1] can depend on a[t+1:t+4]. Read primary chunk-end/segment metrics; causal masking awaits a separate decision after experiment 1 qualifies.",
                "thresholds_locked_from": "calibration", "scientific_verdict": "not_run"}
        if self._readings_digest is None:
            self._readings_digest = readings_digest(readings)
        return readings


def readings_digest(readings):
    return hashlib.sha256(json.dumps(readings, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def candidate_coverage(valid, nominal_index=0):
    """Model-side candidates only; no outcome fields. Includes all states."""
    if valid.dtype != torch.bool or valid.ndim != 2 or valid.shape[1] < 2:
        raise ValueError("need valid[B,N>=2]")
    nominal = valid[:, nominal_index]
    alternatives = valid.sum(1)-nominal.int()
    return {"validity": float(valid.float().mean()),
            "nominal_plus_alternative_coverage": float((nominal & (alternatives >= 1)).float().mean()),
            "n_states": len(valid), "n_candidates": valid.shape[1]}


@torch.no_grad()
def check_decoder_pairing(decoder, sampled, state, *, min_shuffle_gap=1e-8):
    """Offline adapter preflight, outside the differentiable hot path.

    Use actual held-out flow plans; a constant decoder must fail before enabling
    quality. This check alone does not qualify generated-plan consequences.
    """
    if len(sampled) < 2:
        raise ValueError("decoder pairing preflight needs at least two plans")
    decoded = decoder(sampled, state)
    shuffled = decoder(sampled.roll(1, 0), state)
    if decoded.shape != shuffled.shape or decoded.ndim != 3 or decoded.shape[-1] != 2:
        raise ValueError("decoder must preserve fixed-point XY shape")
    gap = float((decoded.float()-shuffled.float()).square().mean())
    if not np.isfinite(gap) or gap <= min_shuffle_gap:
        raise ValueError("constant/action-blind decoder pairing witness failed")
    return {"decoder_shuffle_gap": gap, "n_plans": len(sampled), "qualified": False}


def qualification_reasons(readings, horizon, *, require_controller=False):
    """Lead's preregistered revision: PI/data validity are health readings, not gates.

    Old: NMSE, paired CI, PI 85-95%, data validity >=90%, A NMSE <=.1.
    New: NMSE <=.9*obs-only, paired CI lower >0, A NMSE <=.1;
    generated-candidate validity/coverage remain required for training authorization.
    """
    reasons = []
    if readings["source_kind"] != "offline_dataset":
        reasons.append("toy records cannot qualify production")
    h = readings["horizons"].get(str(horizon), {})
    if not h.get("available"):
        return reasons+["horizon unavailable"]
    for name, m in h["subspaces"].items():
        if m["world_nmse"] > .9*m["obs_only_nmse"]:
            reasons.append(f"{name}: NMSE vs obs-only")
        if m["paired_improvement_ci95"][0] is None or m["paired_improvement_ci95"][0] <= 0:
            reasons.append(f"{name}: paired CI")
    generated = readings["generated_candidates"]
    for key, threshold in [("validity", .8), ("nominal_plus_alternative_coverage", .8)]:
        if generated.get(key) is None or generated[key] < threshold:
            reasons.append(f"generated {key} missing/below threshold")
    if readings["controller_action_nmse"] > .1:
        reasons.append("A action NMSE >.1")
    return reasons


class QualityService(nn.Module):
    """Quality report on supplied physical actions, or decoder->A->W same-plan.

    diagnostic=True permits wiring checks but NEVER authorize_training().
    Caller supplies existing decoder and actual executor; callbacks receive
    predicted full state afresh at EVERY chunk. Complete tuple validity is an
    explicit boolean caller input, not inferred from an arbitrary embedding.
    """
    def __init__(self, calibrator, *, readings=None, diagnostic=False, expected_fingerprint=None):
        super().__init__()
        if calibrator.artifact is None:
            raise ValueError("missing offline calibration")
        self.world = freeze(calibrator.models["world"])
        self.controller = freeze(calibrator.models["controller"])
        self.calibrator = calibrator; self.calibration = calibrator.artifact
        self.readings = readings; self.diagnostic = diagnostic
        for field in ("source_kind", "dataset_hash", "domain", "representation_version", "split_hash"):
            if getattr(self.calibration, field) != calibrator.train["metadata"][field]:
                raise ValueError("calibration provenance/version mismatch")
        if self.calibration.source_split != "calibration" or self.calibration.representation_version != VERSION:
            raise ValueError("calibration split/version mismatch")
        if expected_fingerprint is not None and expected_fingerprint != self.calibration.teacher_fingerprint:
            raise ValueError("teacher fingerprint mismatch")
        actual = calibration_fingerprint(calibrator, self.calibration.thresholds, self.calibration.scales,
                                         self.calibration.pi_radius, self.calibration.controller_disagreement_p95)
        if actual != self.calibration.teacher_fingerprint:
            raise ValueError("teacher changed after calibration")
        self.eval()

    def train(self, mode=True):
        # Outer model.train() must not toggle frozen teacher stochastic layers.
        super().train(False)
        return self

    def authorize_training(self, horizon, *, same_plan=False):
        if self.diagnostic or self.readings is None:
            raise ValueError("diagnostic/unqualified oracle cannot enable training")
        if self.readings["teacher_fingerprint"] != self.calibration.teacher_fingerprint:
            raise ValueError("qualification fingerprint mismatch")
        if self.readings.get("split_hash") != self.calibration.split_hash:
            raise ValueError("qualification split hash mismatch")
        if readings_digest(self.readings) != self.calibrator._readings_digest:
            raise ValueError("qualification readings were not produced by verified evaluation")
        if self.calibration.source_kind != "offline_dataset" or self.calibration.representation_version != VERSION:
            raise ValueError("toy or incompatible calibration cannot enable production training")
        if same_plan and self.calibration.domain != "pointmaze":
            raise ValueError("ant R>0 unsupported")
        reasons = qualification_reasons(self.readings, horizon, require_controller=same_plan)
        if reasons:
            raise ValueError("oracle qualification failed: "+"; ".join(reasons))

    @torch.no_grad()
    def measure_generated_candidates(self, state, actions, goal, *, parent_model_hash, candidate_ids, nominal_index=0):
        """Measure actual candidate tensors, record provenance, and seal new readings.

        This accepts no caller-provided validity matrix or qualification readings.
        Candidate generation itself remains the later F5 caller's responsibility.
        """
        if self.readings is None or readings_digest(self.readings) != self.calibrator._readings_digest:
            raise ValueError("evaluate held-out data before generated candidates")
        if not isinstance(parent_model_hash, str) or len(parent_model_hash) != 64 or any(c not in "0123456789abcdef" for c in parent_model_hash):
            raise ValueError("candidate parent_model_hash must be sha256")
        if actions.ndim != 4 or state.ndim != 2 or goal.shape != (len(state), 2):
            raise ValueError("candidate tensors require state[B,D], actions[B,N,H,A], goal[B,2]")
        b, n, h, adim = actions.shape
        if n < 2 or not 0 <= nominal_index < n or len(candidate_ids) != n or len(set(candidate_ids)) != n or any(not isinstance(x, str) or not x for x in candidate_ids):
            raise ValueError("candidate ids must be distinct nonempty strings")
        valid = self.report(state.repeat_interleave(n, 0), actions.reshape(b*n, h, adim),
                            goal.repeat_interleave(n, 0)).valid.reshape(b, n)
        measured = candidate_coverage(valid, nominal_index)
        measured.update(parent_model_hash=parent_model_hash, candidate_ids=list(candidate_ids))
        updated = copy.deepcopy(self.readings)
        updated["generated_candidates"] = measured
        self.calibrator._readings_digest = readings_digest(updated)
        self.readings = updated
        return copy.deepcopy(measured)

    def report(self, state, actions, goal, *, tuple_valid=None, prediction=None):
        h = actions.shape[1]
        if str(h) not in self.calibration.thresholds:
            raise ValueError("horizon not calibrated; arrival/hold beyond horizon unknown")
        if goal.shape != (len(state), 2):
            raise ValueError("goal must be raw XY[B,2]")
        if tuple_valid is None:
            tuple_valid = torch.ones(len(state), dtype=torch.bool, device=state.device)
        if tuple_valid.shape != (len(state),) or tuple_valid.dtype != torch.bool:
            raise ValueError("tuple_valid must be bool[B]")
        pred = self.world(state, actions) if prediction is None else prediction
        support = self.calibrator.support(state, actions)
        components, uncertainty = consequence_components(self.world, pred, actions, goal, support)
        thresholds = self.calibration.thresholds[str(h)]
        components["support_ok"] = support <= thresholds["support_p95"]
        components["uncertainty_ok"] = uncertainty <= thresholds["disagreement_p95"]
        components["tuple_ok"] = tuple_valid
        valid = components["support_ok"] & components["uncertainty_ok"] & tuple_valid
        # Equal locked weights, offline calibration scales. Invalid cost is NOT zeroed.
        cost = sum(components[k].float()/self.calibration.scales[str(h)][k]
                   for k in ("arrival", "hold", "behavior_support"))
        return QualityReport(cost.float(), components, valid, uncertainty.float(), support.float(), self.calibration.teacher_fingerprint)

    def rollout(self, state, candidate, goal, executor: Callable, *, horizon, tuple_valid):
        """Closed-loop MODEL rollout; executor(state,candidate,chunk_index)->actions.

        Each ensemble member re-decodes with its own predicted full state.
        Neither this callback nor the service may execute an environment.
        """
        if horizon not in HORIZONS:
            raise ValueError("unsupported physical horizon")
        member_predictions = []; action_sequences = []
        for member in range(3):
            current = state; future = []; actions = []
            for t in range(horizon//4):
                a = executor(current, candidate, t)
                if a.shape != (len(state), 4, self.world.adim):
                    raise ValueError("executor must return original CHUNK actions")
                p = self.world(current, a)[member]
                future.append(p); actions.append(a); current = p[:, -1]
            member_predictions.append(torch.cat(future, 1)); action_sequences.append(torch.cat(actions, 1))
        prediction = torch.stack(member_predictions)
        reports = [self.report(state, a, goal, tuple_valid=tuple_valid, prediction=prediction) for a in action_sequences]
        result = reports[0]
        result.valid = torch.stack([r.valid for r in reports]).all(0)
        result.support = torch.stack([r.support for r in reports]).amax(0)
        result.cost = torch.stack([r.cost for r in reports]).mean(0)
        result.components["behavior_support"] = torch.stack([r.components["behavior_support"] for r in reports]).mean(0)
        result.components["support_ok"] = torch.stack([r.components["support_ok"] for r in reports]).all(0)
        return result

    def same_plan(self, u, state, goal, decoder: Callable, *, horizon=4):
        if self.calibration.domain != "pointmaze":
            raise ValueError("ant R>0 unsupported; use actual frozen executor selector interface")
        # The decoder callback preserves the production _dec/_intent_inv semantics.
        controller_uncertainties = []
        def executor(current, plan, chunk_index):
            path = decoder(plan, current)
            predictions = self.controller(current, path)
            controller_uncertainties.append((predictions.float()/self.world.action_norm.scale).var(0, unbiased=False).mean((1, 2)))
            return predictions.mean(0)
        report = self.rollout(state, u, goal, executor, horizon=horizon,
                              tuple_valid=torch.ones(len(state), dtype=torch.bool, device=state.device))
        uncertainty = torch.stack(controller_uncertainties).amax(0)
        report.components["controller_disagreement"] = uncertainty
        report.components["controller_ok"] = uncertainty <= self.calibration.controller_disagreement_p95
        report.valid = report.valid & report.components["controller_ok"]
        return report

    def quality(self, state, goal, decoder, *, horizon=4):
        """Proposal-v2 closure API: quality(u)->Tensor[B], no actions_data target."""
        self.authorize_training(horizon, same_plan=True)
        def call(u):
            report = self.same_plan(u, state, goal, decoder, horizon=horizon)
            if not bool(report.valid.any()):
                raise ValueError("all candidates invalid; refusing quality supervision")
            # Invalid examples cannot masquerade as zero quality or an easy subset.
            return torch.where(report.valid, report.cost, torch.full_like(report.cost, float("inf")))
        return call


def exposure_loss(head, detached_u, state, decoder, service, *, require_qualified=True):
    """Same-plan detached A targets; gradients only to head/its captured cond."""
    if require_qualified:
        service.authorize_training(4, same_plan=True)
    plan = detached_u.detach()
    with torch.no_grad():
        path = decoder(plan, state.detach())
        predictions = service.controller(state.detach(), path)
        target = predictions.mean(0)
        report = service.report(state.detach(), target, path[:, -1])
        uncertainty = (predictions.float()/service.world.action_norm.scale).var(0, unbiased=False).mean((1, 2))
        report.valid = report.valid & (uncertainty <= service.calibration.controller_disagreement_p95)
    prediction = head(plan)
    if prediction.shape != target.shape:
        raise ValueError("head/controller CHUNK shape mismatch")
    if not bool(report.valid.any()):
        return prediction.float().sum()*0, {"valid": False, "n_valid": 0, "n_total": len(plan), "reason": "no valid same-plan action labels"}
    error = ((prediction.float()-target.float())/service.world.action_norm.scale).square().mean((1, 2))
    return error[report.valid].mean(), {"valid": True, "n_valid": int(report.valid.sum()), "n_total": len(plan), "reason": None}


@torch.no_grad()
def exposure_gauge(head, state, plans, decoder, service):
    """Fixed caller-supplied held-out plans; consumes no RNG, restores all modes."""
    modes = [(m, m.training) for m in head.modules()]
    head.eval()
    result = {}
    try:
        for name in ("anchor", "r0", "r1", "r3"):
            if name not in plans:
                result[name] = {"nmse": None, "valid": False, "reason": "missing plan", "n_valid": 0, "n_total": len(state)}
                continue
            loss, info = exposure_loss(head, plans[name], state, decoder, service, require_qualified=False)
            result[name] = dict(info, nmse=float(loss) if info["valid"] else None)
        anchor = result["anchor"]["nmse"]
        for name in ("r0", "r1", "r3"):
            error = result[name]["nmse"]
            result[name]["gap"] = error-anchor if error is not None and anchor is not None else None
    finally:
        for module, mode in modes:
            module.training = mode
    return result
