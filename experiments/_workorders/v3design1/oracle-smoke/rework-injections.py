"""Reversible in-memory negative controls for the two rework injection tests."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(root))
sys.path.insert(0, str(root / "tests"))
import test_refine_quality_v3 as case


def run(name):
    suite = unittest.defaultTestLoader.loadTestsFromName("OracleTests." + name, case)
    return unittest.TextTestRunner(verbosity=2, stream=sys.stdout).run(suite)


original_build = case.build_records
original_forward = case.refine_models.ConsequenceEnsemble.forward


def self_labeling_build(*args, **kwargs):
    # Old failure mode: caller's offline_dataset label was accepted for toy arrays.
    if kwargs.get("source_kind") == "offline_dataset":
        kwargs["source_kind"] = "toy_offline"
        result = original_build(*args, **kwargs)
        for record in result.values():
            record["metadata"]["source_kind"] = "offline_dataset"
        return result
    return original_build(*args, **kwargs)


def unsealed_authorize(self, horizon, *, same_plan=False):
    # Old authorization order: source/fingerprint checks, no readings digest.
    if self.readings["teacher_fingerprint"] != self.calibration.teacher_fingerprint:
        raise ValueError("qualification fingerprint mismatch")
    if self.calibration.source_kind != "offline_dataset":
        raise ValueError("toy or incompatible calibration cannot enable production training")
    reasons = case.qualification_reasons(self.readings, horizon, require_controller=same_plan)
    if reasons:
        raise ValueError("oracle qualification failed: " + "; ".join(reasons))


def action_blind_forward(self, state, actions):
    return original_forward(self, state, actions * 0)


with patch.object(case, "build_records", self_labeling_build):
    g1_red = run("test_offline_label_injection_rejected")
with patch.object(case.QualityService, "authorize_training", unsealed_authorize):
    g2_red = run("test_hand_filled_qualification_readings_rejected")
with patch.object(case.refine_models.ConsequenceEnsemble, "forward", action_blind_forward):
    witness_red = run("test_action_blind_witness_is_detected")
g1_green = run("test_offline_label_injection_rejected")
g2_green = run("test_hand_filled_qualification_readings_rejected")
witness_green = run("test_action_blind_witness_is_detected")
expected = (not g1_red.wasSuccessful() and not g2_red.wasSuccessful() and not witness_red.wasSuccessful()
            and g1_green.wasSuccessful() and g2_green.wasSuccessful())
expected = expected and witness_green.wasSuccessful()
print("EXPECTED_THREE_RED_THREE_GREEN:", expected)
raise SystemExit(0 if expected else 1)
