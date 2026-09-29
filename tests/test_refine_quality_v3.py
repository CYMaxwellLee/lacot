"""CPU-only v3 contract/wiring tests. No pytest dependency or environment import."""
import copy
import hashlib
import json
import shutil
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from torch import nn

import lacot.refine_models as refine_models
from lacot.refine_models import (META_FIELDS, VERSION, load_offline_npz,
    sha256_file, toy_arrays, build_records, train_models, validate_record, validate_metadata,
    assert_disjoint, subset, action_witness, split_fingerprint, split_episodes)
from lacot.refine_quality import (OfflineCalibrator, QualityService, QualityReport,
    exposure_loss, exposure_gauge, candidate_coverage, qualification_reasons, check_decoder_pairing,
    calibration_fingerprint)
from experiments.refine_v3.train_offline import save_checkpoint, load_checkpoint


class OracleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.arrays = toy_arrays("pointmaze", seed=12, episodes=30)
        cls.records = build_records(*cls.arrays, domain="pointmaze", dataset_hash="a"*64,
                                   split_seed=12, t_cap=32, max_windows=512)
        cls.models, cls.curves = train_models(cls.records["train"], steps=140, hidden=48, seed=12)
        cls.cal = OfflineCalibrator(cls.models, cls.records["train"])
        cls.cal.fit(cls.records["calibration"])
        cls.readings = cls.cal.evaluate(cls.records["test"], seed=12)
        cls.service = QualityService(cls.cal, readings=cls.readings, diagnostic=True)

    def test_eval_injection_rejected_at_training_api(self):
        # Valid full record except for the injected field: cannot pass by missing schema.
        for field in ("reward", "success", "oracle_index"):
            for location in ("top", "metadata"):
                r = copy.deepcopy(self.records["train"])
                (r if location == "top" else r["metadata"])[field] = torch.ones(2)
                with self.subTest(field=field, location=location), self.assertRaisesRegex(ValueError, "evaluation fields"):
                    train_models(r, steps=1)
        r = copy.deepcopy(self.records["train"]); r["metadata"]["eval_only"] = True
        with self.assertRaisesRegex(ValueError, "eval_only"):
            train_models(r, steps=1)

    def test_all_metadata_missing_fields_rejected(self):
        for field in META_FIELDS:
            r = copy.deepcopy(self.records["train"]); del r["metadata"][field]
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "metadata schema"):
                train_models(r, steps=1)

    def test_sources_and_generated_parent(self):
        for source in ("environment_rollout", "historical_experiment", "evaluation"):
            r = copy.deepcopy(self.records["train"]); r["metadata"]["source_kind"] = source
            with self.assertRaisesRegex(ValueError, "whitelisted"):
                train_models(r, steps=1)
        r = copy.deepcopy(self.records["train"]); r["metadata"]["source_kind"] = "model_generated"
        with self.assertRaisesRegex(ValueError, "parent_model_hash"):
            train_models(r, steps=1)
        r["metadata"]["parent_model_hash"] = "b"*64; r["metadata"]["candidate_id"] = "sample-1"
        validate_metadata(r["metadata"])
        with self.assertRaisesRegex(ValueError, "original offline pairs"):
            train_models(r, steps=1)

    def test_loader_whitelist_and_npz_eval_injection(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/"raw.npz"
            obs, act, term = self.arrays
            np.savez(p, observations=obs, actions=act, terminals=term)
            manifest = Path(tmp)/"manifest.json"
            def allow_current(domain="pointmaze", obs_dim=obs.shape[1]):
                manifest.write_text(json.dumps({"datasets": {sha256_file(p): {
                    "domain": domain, "filename": p.name, "obs_dim": obs_dim, "act_dim": act.shape[1]}}}))
            allow_current()
            with patch.object(refine_models, "DATA_MANIFEST", manifest):
                got = load_offline_npz(p, expected_domain="pointmaze")
                np.testing.assert_array_equal(got.observations, obs)
                with self.assertRaises(ValueError):
                    got.observations[0, 0] = 999
                with self.assertRaises(ValueError):
                    got.observations.setflags(write=True)
                forged = refine_models.VerifiedArrays(obs, act, term, got.dataset_hash, got.domain, got._token)
                with self.assertRaisesRegex(ValueError, "verified loader"):
                    build_records(forged, split_seed=12, t_cap=32)
                shutil.copyfile(p, Path(tmp)/"other.npz")
                with self.assertRaisesRegex(ValueError, "filename"):
                    load_offline_npz(Path(tmp)/"other.npz", expected_domain="pointmaze")
                with self.assertRaisesRegex(ValueError, "domain"):
                    load_offline_npz(p, expected_domain="ant")
                allow_current(obs_dim=99)
                with self.assertRaisesRegex(ValueError, "dimensions"):
                    load_offline_npz(p, expected_domain="pointmaze")
                allow_current()
                np.savez(p, observations=obs+1, actions=act, terminals=term)
                with self.assertRaisesRegex(ValueError, "manifest whitelist"):
                    load_offline_npz(p, expected_domain="pointmaze")
                np.savez(p, observations=obs, actions=act, terminals=term)
                for field in ("reward", "success", "oracle_index"):
                    np.savez(p, observations=obs, actions=act, terminals=term, **{field: np.ones(len(obs))})
                    allow_current()
                    with self.assertRaisesRegex(ValueError, "evaluation fields"):
                        load_offline_npz(p, expected_domain="pointmaze")
                np.savez(p, observations=obs, actions=act, terminals=term, hidden_payload=np.ones(2))
                allow_current()
                with self.assertRaisesRegex(ValueError, "whitelist"):
                    load_offline_npz(p, expected_domain="pointmaze")

    def test_episode_split_and_original_action_pairing(self):
        obs, actions, terminals = self.arrays
        assert_disjoint(self.records)
        rows_by_split = []
        for rec in self.records.values():
            occupied = set()
            for i, row in enumerate(rec["row"].tolist()):
                n = int(rec["mask"][i].sum())
                occupied.update(range(row, max(row+n, int(rec["path_end"][i]))+1))
                np.testing.assert_array_equal(rec["actions"][i, :4], actions[row:row+4])
                self.assertEqual(n % 4, 0)
                for j in range(0, n, 4):
                    np.testing.assert_array_equal(rec["actions"][i, j:j+4], actions[row+j:row+j+4])
                np.testing.assert_array_equal(rec["future"][i, :n], obs[row+1:row+n+1])
                self.assertFalse(terminals[row:row+n].any())
                end = int(rec["path_end"][i])
                t = np.linspace(row, end, rec["path"].shape[1]); lo = np.floor(t).astype(int); hi = np.minimum(lo+1, end)
                expected = obs[lo, :2]*(1-(t-lo)[:, None])+obs[hi, :2]*(t-lo)[:, None]
                np.testing.assert_allclose(rec["path"][i], expected, rtol=1e-6, atol=1e-6)
            rows_by_split.append(occupied)
        for i in range(3):
            for j in range(i):
                self.assertFalse(rows_by_split[i] & rows_by_split[j])
        bad = copy.deepcopy(self.records)
        bad["test"]["episode_id"][0] = bad["train"]["episode_id"][0]
        with self.assertRaisesRegex(ValueError, "overlap"):
            assert_disjoint(bad)

    def test_normalization_train_only(self):
        train = self.records["train"]
        norm = self.models["world"].state_norm
        torch.testing.assert_close(norm.mean, train["state"].mean(0))
        torch.testing.assert_close(norm.scale, train["state"].std(0, unbiased=False).clamp_min(1e-4))
        actions = train["actions"][train["mask"]]
        torch.testing.assert_close(self.models["world"].action_norm.scale, actions.std(0, unbiased=False).clamp_min(1e-4))
        held = copy.deepcopy(self.records["test"]); held["state"] += 10000
        before = norm.mean.clone()
        self.cal.evaluate(held)
        torch.testing.assert_close(norm.mean, before)
        for split in ("calibration", "test"):
            with self.assertRaisesRegex(ValueError, "train split"):
                train_models(self.records[split], steps=1)

    def test_shuffled_actions_significantly_worse(self):
        # Used unchanged for the trained obs-only negative witness in red-trained.log.
        cal = OfflineCalibrator(self.models, self.records["train"])
        cal.fit(self.records["calibration"])
        readings = cal.evaluate(self.records["test"], seed=12)
        for h, info in readings["horizons"].items():
            for name, metric in info["subspaces"].items():
                with self.subTest(h=h, subspace=name):
                    self.assertGreater(metric["shuffled_degradation_ci95"][0], 0.1)
                    self.assertGreater(metric["shuffled_action_nmse"], 2*metric["world_nmse"])

    def test_action_blind_witness_is_detected(self):
        r = self.records["test"]
        with torch.no_grad():
            pred = self.models["world"](r["state"], r["actions"][:, :4]).mean(0)
            shuffled = self.models["world"](r["state"], r["actions"].roll(1, 0)[:, :4]).mean(0)
        self.assertGreater(action_witness(pred, shuffled, r["future"][:, :4]), .1)

    def test_model_seed_does_not_change_split(self):
        obs, act, term = self.arrays
        other = build_records(obs, act, term, domain="pointmaze", dataset_hash="a"*64,
                              split_seed=12, t_cap=32, max_windows=512)
        self.assertEqual(other["train"]["metadata"]["split_hash"], self.records["train"]["metadata"]["split_hash"])
        _, _, ids = split_episodes(term, 12)
        self.assertEqual(split_fingerprint("a"*64, 12, ids), other["train"]["metadata"]["split_hash"])
        self.assertEqual({k: set(v["episode_id"].tolist()) for k, v in other.items()},
                         {k: set(v["episode_id"].tolist()) for k, v in self.records.items()})
        seed0, _ = train_models(self.records["train"], steps=1, hidden=16, seed=0)
        seed1, _ = train_models(other["train"], steps=1, hidden=16, seed=1)
        self.assertFalse(torch.equal(next(seed0["world"].parameters()), next(seed1["world"].parameters())))

    def test_revised_qualification_gates(self):
        readings = copy.deepcopy(self.readings)
        readings["source_kind"] = "offline_dataset"
        readings["controller_action_nmse"] = .05
        readings["generated_candidates"] = {"validity": 1., "nominal_plus_alternative_coverage": 1.}
        h = readings["horizons"]["4"]
        h["data_validity"] = 0.
        for metric in h["subspaces"].values():
            metric["world_nmse"] = 0.
            metric["paired_improvement_ci95"] = [.1, .2]
            metric["pi90_coverage"] = 0.
        self.assertEqual(qualification_reasons(readings, 4), [])
        first = next(iter(h["subspaces"].values()))
        first["world_nmse"] = first["obs_only_nmse"]
        self.assertTrue(any("NMSE" in x for x in qualification_reasons(readings, 4)))
        first["world_nmse"] = 0.
        first["paired_improvement_ci95"] = [0., .2]
        self.assertTrue(any("paired CI" in x for x in qualification_reasons(readings, 4)))
        first["paired_improvement_ci95"] = [.1, .2]
        readings["controller_action_nmse"] = .11
        self.assertTrue(any("A action" in x for x in qualification_reasons(readings, 4)))

    def test_offline_label_injection_rejected(self):
        obs, act, term = self.arrays
        with self.assertRaisesRegex(ValueError, "verified loader"):
            build_records(obs, act, term, domain="pointmaze", dataset_hash="a"*64,
                          source_kind="offline_dataset", split_seed=12)
        forged = copy.deepcopy(self.records["train"])
        forged["metadata"]["source_kind"] = "offline_dataset"
        with self.assertRaisesRegex(ValueError, "verified loader"):
            validate_metadata(forged["metadata"])
        with self.assertRaisesRegex(ValueError, "verified loader"):
            train_models(forged, steps=1)

    def test_hand_filled_qualification_readings_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/"raw.npz"
            obs, act, term = self.arrays
            np.savez(p, observations=obs, actions=act, terminals=term)
            manifest = Path(tmp)/"manifest.json"
            manifest.write_text(json.dumps({"datasets": {sha256_file(p): {
                "domain": "pointmaze", "filename": p.name, "obs_dim": obs.shape[1], "act_dim": act.shape[1]}}}))
            with patch.object(refine_models, "DATA_MANIFEST", manifest):
                verified = load_offline_npz(p, expected_domain="pointmaze")
            records = build_records(verified, split_seed=12, t_cap=32, max_windows=128)
            models, _ = train_models(records["train"], steps=1, hidden=16, seed=12)
            cal = OfflineCalibrator(models, records["train"])
            cal.fit(records["calibration"])
            readings = cal.evaluate(records["test"], seed=12)
            readings["generated_candidates"] = {"validity": 1., "nominal_plus_alternative_coverage": 1.}
            readings["controller_action_nmse"] = 0.
            for h in readings["horizons"].values():
                if h.get("available"):
                    for metric in h["subspaces"].values():
                        metric["world_nmse"] = 0.
                        metric["paired_improvement_ci95"] = [1., 2.]
            self.assertEqual(qualification_reasons(readings, 4), [])
            service = QualityService(cal, readings=readings)
            with self.assertRaisesRegex(ValueError, "readings were not produced"):
                service.authorize_training(4)

    def test_generated_candidate_reading_uses_model_report(self):
        cal = OfflineCalibrator(self.models, self.records["train"])
        cal.fit(self.records["calibration"])
        readings = cal.evaluate(self.records["test"], seed=12)
        service = QualityService(cal, readings=readings)
        r = self.records["test"]
        state = r["state"][:3]
        goal = r["path"][:3, -1]
        actions = r["actions"][:3, None, :4].repeat(1, 2, 1, 1)
        measured = service.measure_generated_candidates(
            state, actions, goal, parent_model_hash="b"*64, candidate_ids=["nominal", "alternative"])
        direct = service.report(state.repeat_interleave(2, 0), actions.reshape(6, 4, actions.shape[-1]),
                                goal.repeat_interleave(2, 0)).valid.reshape(3, 2)
        self.assertEqual(measured["validity"], candidate_coverage(direct)["validity"])
        self.assertEqual(measured["candidate_ids"], ["nominal", "alternative"])
        self.assertEqual(measured["parent_model_hash"], "b"*64)
        service.readings["generated_candidates"]["validity"] = -1.
        with self.assertRaisesRegex(ValueError, "readings were not produced"):
            service.authorize_training(4)

    def test_decoder_pairing_and_short_horizon_refusal(self):
        r = self.records["test"]
        check_decoder_pairing(lambda u, s: u, r["path"][:8], r["state"][:8])
        with self.assertRaisesRegex(ValueError, "decoder pairing witness"):
            check_decoder_pairing(lambda u, s: torch.zeros_like(u), r["path"][:8], r["state"][:8])
        train = copy.deepcopy(self.records["train"]); cal = copy.deepcopy(self.records["calibration"])
        for rec in (train, cal):
            rec["mask"][:, 4:] = False
        short = OfflineCalibrator(self.models, train)
        artifact = short.fit(cal)
        self.assertEqual(set(artifact.thresholds), {"4"})
        service = QualityService(short, diagnostic=True)
        with self.assertRaisesRegex(ValueError, "not calibrated"):
            service.report(r["state"][:8], r["actions"][:8, :16], r["path"][:8, -1])

    def test_calibration_only_thresholds_and_locked(self):
        with self.assertRaisesRegex(ValueError, "calibration split"):
            OfflineCalibrator(self.models, self.records["train"]).fit(self.records["test"])
        with self.assertRaisesRegex(ValueError, "locked"):
            self.cal.fit(self.records["calibration"])
        r = self.records["calibration"]; w = self.models["world"]
        for h in (4, 16, 32):
            good = r["mask"][:, :h].all(1)
            a = r["actions"][good, :h]; s = r["state"][good]
            with torch.no_grad():
                support = self.cal.support(s, a)
                disagree = (w(s, a)/w.state_norm.scale).var(0, unbiased=False).mean((1, 2))
            self.assertAlmostEqual(self.cal.artifact.thresholds[str(h)]["support_p95"], float(torch.quantile(support, .95)), places=6)
            self.assertAlmostEqual(self.cal.artifact.thresholds[str(h)]["disagreement_p95"], float(torch.quantile(disagree, .95)), places=6)
        locked = copy.deepcopy(self.cal.artifact.json())
        r = copy.deepcopy(self.records["test"]); r["actions"] *= 100
        self.cal.evaluate(r)
        self.assertEqual(locked, self.cal.artifact.json())

    def test_report_schema_and_invalid_not_zero(self):
        r = self.records["test"]
        report = self.service.report(r["state"][:5], r["actions"][:5, :4], r["path"][:5, -1], tuple_valid=torch.zeros(5, dtype=torch.bool))
        self.assertEqual(set(report.json()), {"cost", "components", "valid", "uncertainty", "support", "teacher_fingerprint"})
        self.assertFalse(report.valid.any())
        self.assertTrue((report.cost > 0).all())
        self.assertEqual(report.cost.dtype, torch.float32)
        with self.assertRaisesRegex(ValueError, "shape/device"):
            QualityReport(report.cost, report.components, report.valid[:1], report.uncertainty, report.support, report.teacher_fingerprint)
        with self.assertRaisesRegex(ValueError, "horizon"):
            self.service.report(r["state"][:5], r["actions"][:5, :8], r["path"][:5, -1])

    def test_forward_and_backward_same_plan_bf16(self):
        r = self.records["test"]; state = r["state"][:6].clone().requires_grad_(); u = r["path"][:6].clone().requires_grad_()
        for amp in (False, True):
            with torch.autocast("cpu", dtype=torch.bfloat16, enabled=amp):
                report = self.service.same_plan(u, state, r["path"][:6, -1], lambda plan, s: plan + .01*s[:, None, :2])
                loss = report.cost.mean()
            self.assertEqual(loss.dtype, torch.float32)
            gu, gs = torch.autograd.grad(loss, (u, state))
            self.assertTrue(torch.isfinite(gu).all() and gu.abs().sum() > 0)
            self.assertTrue(torch.isfinite(gs).all() and gs.abs().sum() > 0)
        self.service.train()
        self.assertFalse(self.service.world.training)
        self.assertTrue(all(p.grad is None for m in self.models.values() for p in m.parameters()))

    def test_executor_rereads_predicted_state_every_chunk(self):
        r = self.records["test"]; state = r["state"][:3]; visits = []
        candidate = r["actions"][:3, :4].clone().requires_grad_()
        def executor(current, plan, index):
            visits.append((index, current.detach().clone()))
            return plan + .01*current[:, None, :2]
        report = self.service.rollout(state, candidate, r["path"][:3, -1], executor, horizon=16, tuple_valid=torch.ones(3, dtype=torch.bool))
        self.assertEqual(len(visits), 12)
        for member in range(3):
            torch.testing.assert_close(visits[member*4][1], state)
            self.assertFalse(torch.equal(visits[member*4+1][1], state))
        gradient = torch.autograd.grad(report.cost.sum(), candidate)[0]
        self.assertGreater(float(gradient.abs().sum()), 0)

    def test_training_refused_without_qualification(self):
        with self.assertRaisesRegex(ValueError, "unqualified"):
            self.service.authorize_training(4, same_plan=True)
        reasons = qualification_reasons(self.readings, 4, require_controller=True)
        self.assertTrue(any("toy" in s for s in reasons))
        self.assertTrue(any("generated" in s for s in reasons))
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            QualityService(self.cal, expected_fingerprint="wrong")
        readings = copy.deepcopy(self.readings); readings["teacher_fingerprint"] = "wrong"
        service = QualityService(self.cal, readings=readings)
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            service.authorize_training(4)
        readings = copy.deepcopy(self.readings); readings["split_hash"] = "wrong"
        service = QualityService(self.cal, readings=readings)
        with self.assertRaisesRegex(ValueError, "split hash"):
            service.authorize_training(4)

    def test_calibration_fingerprint_detects_changed_scale(self):
        changed = copy.copy(self.cal)
        changed.artifact = copy.deepcopy(self.cal.artifact)
        changed.artifact.thresholds["4"]["support_p95"] += .01
        with self.assertRaisesRegex(ValueError, "changed after calibration"):
            QualityService(changed, diagnostic=True)
        changed.artifact = copy.deepcopy(self.cal.artifact)
        changed.artifact.source_kind = "offline_dataset"
        with self.assertRaisesRegex(ValueError, "provenance/version mismatch"):
            QualityService(changed, diagnostic=True)
        changed = copy.copy(self.cal)
        changed.train = {"metadata": dict(self.cal.train["metadata"])}
        changed.train["metadata"]["split_hash"] = "0"*64
        artifact = self.cal.artifact
        mutated = calibration_fingerprint(changed, artifact.thresholds, artifact.scales,
                                          artifact.pi_radius, artifact.controller_disagreement_p95)
        self.assertNotEqual(mutated, artifact.teacher_fingerprint)

    def test_all_invalid_exposure_has_explicit_reason(self):
        r = self.records["test"]; u = r["path"][:4]; state = r["state"][:4]
        head = nn.Sequential(nn.Flatten(), nn.Linear(64, 8), nn.Unflatten(1, (4, 2)))
        original = self.service.report
        def invalid(*args, **kwargs):
            report = original(*args, **kwargs); report.valid = torch.zeros_like(report.valid); return report
        self.service.report = invalid
        try:
            loss, info = exposure_loss(head, u, state, lambda u, s: u, self.service, require_qualified=False)
            self.assertFalse(info["valid"]); self.assertEqual(info["n_valid"], 0)
            self.assertIsNotNone(info["reason"]); self.assertEqual(float(loss), 0)
        finally:
            self.service.report = original

    def test_same_plan_exposure_and_gauge(self):
        r = self.records["test"]; state = r["state"][:8]; u = r["path"][:8].clone().requires_grad_()
        head = nn.Sequential(nn.Flatten(), nn.Linear(64, 8), nn.Unflatten(1, (4, 2)))
        seen = []
        def decoder(plan, s):
            seen.append(plan.clone()); return plan
        # Thresholds are immutable in production; stub only the validity report for this
        # independent gradient-ownership test, avoiding accidental no-valid zero loss.
        original = self.service.report
        def all_valid(*args, **kwargs):
            result = original(*args, **kwargs); result.valid = torch.ones_like(result.valid); return result
        self.service.report = all_valid
        try:
            loss, info = exposure_loss(head, u, state, decoder, self.service, require_qualified=False)
            self.assertTrue(info["valid"])
            loss.backward()
            self.assertIsNone(u.grad)
            self.assertGreater(float(head[1].weight.grad.abs().sum()), 0)
            torch.testing.assert_close(seen[0], u)
            rng = torch.get_rng_state().clone(); head.train()
            gauge = exposure_gauge(head, state, {"anchor": u, "r0": u, "r1": u+.1}, decoder, self.service)
            self.assertTrue(head.training); torch.testing.assert_close(rng, torch.get_rng_state())
            self.assertIsNone(gauge["r3"]["nmse"]); self.assertIsNone(gauge["r3"]["gap"])
            self.assertAlmostEqual(gauge["r0"]["gap"], 0)
        finally:
            self.service.report = original

    def test_reference_nll_calibration(self):
        class Reference(nn.Module):
            def forward(self, state, actions):
                return actions.float().square().mean((1, 2))
        cal = OfflineCalibrator(self.models, self.records["train"], reference_nll=Reference(), reference_fingerprint="c"*64)
        artifact = cal.fit(self.records["calibration"])
        self.assertEqual(artifact.support_kind, "reference_nll_per_dimension")
        r = self.records["calibration"]
        expected = r["actions"][:, :4].square().mean((1, 2)).quantile(.95)
        self.assertAlmostEqual(artifact.thresholds["4"]["support_p95"], float(expected), places=6)

    def test_checkpoint_roundtrip_and_candidate_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/"oracle.pt"
            save_checkpoint(p, self.models, self.cal, self.readings, hidden=48)
            models, cal, readings = load_checkpoint(p)
            self.assertEqual(cal.artifact.json(), self.cal.artifact.json())
            r = self.records["test"]
            with torch.no_grad():
                torch.testing.assert_close(models["world"](r["state"][:4], r["actions"][:4, :4]), self.models["world"](r["state"][:4], r["actions"][:4, :4]))
        cover = candidate_coverage(torch.tensor([[True, True], [True, False], [False, True]]))
        self.assertAlmostEqual(cover["nominal_plus_alternative_coverage"], 1/3)
        self.assertAlmostEqual(cover["validity"], 2/3)


if __name__ == "__main__":
    unittest.main()
