"""CPU-only launch regression; --launcher selects an unchanged baseline copy."""
import argparse
from contextlib import ExitStack
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


# Keep test runs from creating bytecode files in the repository.
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
LAUNCHER = Path(__file__).with_name("run.py")
GUARD_ERROR = "result arm/sigma differs from selected plan"


def load_launcher():
    spec = importlib.util.spec_from_file_location("arm_sigma_launcher", LAUNCHER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ArmSigmaGuardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.launcher = load_launcher()

    def exercise(self, mode, smoke, index, mismatch=None):
        module = self.launcher
        # Retain artifacts in /tmp: no files are deleted by this test.
        base = Path(tempfile.mkdtemp(prefix="ucontrast2-arm-sigma-"))
        arm = "A" if index == 0 else "B"
        planned_sigma = 0.0 if index == 0 else 0.05
        outdir = base / arm
        args = SimpleNamespace(mode=mode, smoke=smoke, index=index,
                               outdir=str(base), calibration_root=str(base / "cal"))
        questions = [(task, episode) for task in range(1, 6)
                     for episode in range(10)]
        ckpts = {33: "fake-s33.pt", 35: "fake-s35.pt"}

        def write_json(path, obj):
            with Path(path).open("x") as stream:
                json.dump(obj, stream, allow_nan=False)

        def read_json(path):
            with Path(path).open() as stream:
                return json.load(stream)

        def reserve(base_path, selected_arm):
            self.assertEqual(Path(base_path), base)
            self.assertEqual(selected_arm, arm)
            outdir.mkdir()
            return outdir

        def fake_rollout(command, *, cwd, env, check):
            self.assertEqual(cwd, module.ROOT)
            self.assertTrue(check)
            is_n64_main = mode == "n64" and not smoke
            source = ("experiments/_workorders/ucontrast2/rollout_n64.py"
                      if is_n64_main else "experiments/scratch_lacot_rollout.py")
            self.assertEqual(command, [sys.executable, "-u", source])
            self.assertEqual(env["LACOT_ORACLE_ARM"], arm)
            sigma = float(env["LACOT_ORACLE_SIGMA"])
            self.assertEqual(sigma, planned_sigma)
            draws = int(env["LACOT_ORACLE_DRAWS"])
            pairs = (questions if is_n64_main else
                     [(task, episode) for task in range(1, 6)
                      for episode in (range(40, 45) if smoke else range(40))])
            tasks = [dict(task=task, episode=episode) for task, episode in pairs]
            oracle = {str(k): 0.0 for k in range(1, draws + 1)}
            quality = [0.0] * draws
            # All later launch gates are satisfied, so an omitted guard writes PASS.
            old.independent_oracle.return_value = (oracle, quality, 0)
            data = dict(arm="A" if mismatch == "arm" else arm,
                        sigma=0.10 if mismatch == "sigma" else sigma,
                        ckpt=env["LACOT_LOAD_CKPT"], draws_per_task=draws,
                        tasks=tasks, oracle_at_k=oracle, per_draw_quality=quality,
                        pooled_oracle=0.0)
            if is_n64_main:
                data["sampled_questions"] = json.loads(
                    env["LACOT_UCONTRAST2_QUESTIONS_JSON"])
                data["question_seed"] = module.QUESTION_SEED
            # launch explicitly checks these final three keys in this order.
            data.update(n_tasks=len(tasks), n_draws=len(tasks) * draws, n_success=0)
            write_json(env["LACOT_ORACLE_OUT"], data)

        old = SimpleNamespace(
            CKPTS=ckpts, SWEEP_SALT=123,
            checkpoint=Mock(return_value={"sha256": "fake-checkpoint"}),
            code_hashes=Mock(return_value={}), sha=Mock(return_value="fake-sha"),
            read=Mock(side_effect=read_json), write=Mock(side_effect=write_json),
            independent_oracle=Mock())
        with ExitStack() as stack:
            stack.enter_context(patch.dict(os.environ, {
                "SLURM_JOB_ID": "mock-allocation", "CUDA_VISIBLE_DEVICES": "mock"
            }, clear=True))
            stack.enter_context(patch.object(module, "old", old))
            stack.enter_context(patch.object(module, "validate_inputs",
                                           return_value=({}, "fake-provenance", "fake-flagoff")))
            stack.enter_context(patch.object(module, "s35_selection",
                                           return_value={"sigma": 0.05}))
            stack.enter_context(patch.object(module, "reserve_outdir", side_effect=reserve))
            stack.enter_context(patch.object(module, "registered_questions",
                                           return_value=questions))
            stack.enter_context(patch.object(module.rollout_n64, "source_hash",
                                           return_value="fake-derived-sha"))
            run = stack.enter_context(patch.object(module.subprocess, "run",
                                                   side_effect=fake_rollout))
            error = None
            try:
                module.launch(args)
            except ValueError as exc:
                error = exc
            run.assert_called_once()
            self.assertEqual(len(list(outdir.glob("*.preflight.json"))), 1)
            receipts = list(outdir.glob("*.verified.json"))
            if mismatch:
                # Check absence even when the baseline fails to raise.
                self.assertFalse(bool(receipts), "mismatched arm/sigma wrote verified.json")
                self.assertIsNotNone(error, "mismatched arm/sigma did not raise")
                self.assertIn(GUARD_ERROR, str(error).lower())
                old.independent_oracle.assert_not_called()
            else:
                self.assertIsNone(error, f"correct arm/sigma raised: {error}")
                old.independent_oracle.assert_called_once()
                self.assertEqual(len(receipts), 1)
                self.assertEqual(read_json(receipts[0])["status"], "PASS")

    def test_n64_wrong_arm(self):
        for smoke in (False, True):
            with self.subTest(smoke=smoke):
                self.exercise("n64", smoke, 1, "arm")

    def test_n64_wrong_sigma(self):
        for smoke in (False, True):
            with self.subTest(smoke=smoke):
                self.exercise("n64", smoke, 1, "sigma")

    def test_s35_wrong_arm(self):
        for smoke in (False, True):
            with self.subTest(smoke=smoke):
                self.exercise("s35", smoke, 1, "arm")

    def test_s35_wrong_sigma(self):
        for smoke in (False, True):
            with self.subTest(smoke=smoke):
                self.exercise("s35", smoke, 1, "sigma")

    def test_n64_correct_arm_sigma(self):
        self.correct_cases("n64")

    def test_s35_correct_arm_sigma(self):
        self.correct_cases("s35")

    def correct_cases(self, mode):
        for smoke in (False, True):
            for index in (0, 1):
                with self.subTest(smoke=smoke, index=index):
                    self.exercise(mode, smoke, index)


class EvidenceResult(unittest.TextTestResult):
    """Keep stdout evidence compact while retaining each original failure message."""

    def getDescription(self, test):
        return test.id().removeprefix("__main__.ArmSigmaGuardTests.")

    def printErrors(self):
        if self.dots or self.showAll:
            self.stream.writeln()
        for test, traceback_text in self.errors + self.failures:
            self.stream.writeln(
                f"{self.getDescription(test)}: {traceback_text.strip().splitlines()[-1]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launcher", type=Path, default=LAUNCHER)
    options, unittest_args = parser.parse_known_args()
    LAUNCHER = options.launcher.resolve()
    unittest.main(argv=[sys.argv[0], *unittest_args],
                  testRunner=unittest.TextTestRunner(stream=sys.stdout, verbosity=1,
                                                     resultclass=EvidenceResult))
