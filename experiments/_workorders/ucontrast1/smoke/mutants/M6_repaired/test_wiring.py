"""CPU-only integration of actual policy_chunk/rollout AST with a toy env/head.

These checks are NOT the s13 calibration or the real-decoder sigma-zero smoke.
"""
import ast
import copy
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
import torch
from collector import Collector, summarize, write_result
from run import (ROOT, SWEEP_EPISODE_OFFSET, SWEEP_SALT, calibration_contract,
                 checkpoint, flagoff_contract, independent_oracle)

torch.set_num_threads(1)
SOURCE = ROOT / "experiments/scratch_lacot_rollout.py"


class ToyEnv:
    def __init__(self):
        self.history = []
        self.action_space = self

    def seed(self, value):
        self.action_seed = value

    def reset(self, seed, options):
        # ogbench reset jitter uses legacy global NumPy, not reset(seed=).
        self.obs = np.array([.38, .28] if seed % 2 else [0., 0.], np.float32)
        self.obs += np.random.uniform(-.01, .01, 2).astype(np.float32)
        self.goal = np.array([.4, .3], np.float32)
        self.goal += np.random.uniform(-.01, .01, 2).astype(np.float32)
        self.history.append(("reset", seed))
        return self.obs.copy(), {"goal": self.goal.copy()}

    def step(self, action):
        self.obs += action * .25
        reached = bool(np.linalg.norm(self.obs - self.goal) < .12)
        self.history.append((action.tobytes(), self.obs.tobytes(), reached))
        return self.obs.copy(), 0., reached, False, {"success": reached}


def harness(collector=None, source=None):
    """Load function bodies, never import the heavyweight/training main script."""
    tree = ast.parse(source if source is not None else SOURCE.read_text())
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef)
             and n.name in ("policy_chunk", "rollout")]
    env = ToyEnv()
    def state(x):
        return torch.as_tensor(x, dtype=torch.float32)[None]
    def head(cond, u):
        act = torch.tanh(.5 * (cond[:, 2:] - cond[:, :2]) + .3 * u[:, 0])
        return act[:, None].expand(-1, 2, -1)
    ns = dict(torch=torch, np=np, env=env, N_TASKS=2, SEEDS=3, MAXH=8,
              FINISH_ON=False, FINISH_R=0., U_SOURCE="flow", GRAD_REFINE=False,
              ORACLE_COLLECTOR=collector, DIAG_DUMP=True, DIAG_ROWS=[],
              _intent_anchor_eval=lambda o, g: None, normstate=state,
              _intent_cond_arm=lambda a: None, condvec=lambda s, g, i: torch.cat((s, g), -1),
              _bon_plan=lambda n, *args: torch.randn(n, 2, 2),
              _apply_refine=lambda c, u, r: u, _q=lambda u: u, ahead=head,
              _reset_grad_cache=lambda: None, _reseed_shuf=lambda s: None)
    exec(compile(ast.Module(body=funcs, type_ignores=[]), str(SOURCE), "exec"), ns)
    return ns


def run_arm(arm, sigma, draws=8, offset=0, salt=0):
    c = Collector(arm, sigma, draws, offset, salt)
    h = harness(c)
    for d in range(draws):
        h["rollout"](1, True, f"cpu toy {arm} {d}", oracle_draw=d)
    return c, summarize(c.rows, draws, arm, sigma)


def run_oracle_loop(source=None):
    """Execute the production draw loop AST, including its R1 cross-check."""
    c = Collector("A", 0., 2)
    text = source if source is not None else SOURCE.read_text()
    h = harness(c, text)
    tree = ast.parse(text)
    loop = next(n for n in ast.walk(tree) if isinstance(n, ast.For)
                and isinstance(n.iter, ast.Call)
                and isinstance(n.iter.func, ast.Name) and n.iter.func.id == "range"
                and any(isinstance(x, ast.Attribute) and x.attr == "draws"
                        for x in ast.walk(n.iter)))
    h.update(ORACLE_ARM="A")
    exec(compile(ast.Module(body=[loop], type_ignores=[]), str(SOURCE), "exec"), h)
    return c


class Wiring(unittest.TestCase):
    def test_sigma_zero_full_trajectory_and_counts(self):
        c, s = run_arm("B", 0.)
        self.assertTrue(s["negative_mutant_passed"])
        self.assertEqual((s["n_tasks"], s["n_draws"]), (6, 48))
        self.assertEqual(s["oracle_at_k"]["1"], s["oracle_at_k"]["8"])
        self.assertTrue(all(r["u_samples"] == r["u_calls"] for r in c.rows))

    def test_a_fresh_b_replayed_flow_and_paired_first_u(self):
        a, sa = run_arm("A", 0.)
        b, sb = run_arm("B", .1)
        self.assertTrue(all(r["u_samples"] == r["u_calls"] for r in a.rows))
        for ta, tb in zip(sa["tasks"], sb["tasks"]):
            self.assertEqual(ta["draws"][0]["first_u_sha256"], tb["draws"][0]["first_u_sha256"])
            self.assertGreater(len({r["first_u_sha256"] for r in ta["draws"]}), 1)
            self.assertEqual(len({r["first_u_sha256"] for r in tb["draws"]}), 1)
            self.assertTrue(all(r["u_samples"] == r["u_calls"] for r in tb["draws"]))
            for ra, rb in zip(ta["draws"], tb["draws"]):
                for key in ("env_seed", "stream_seed", "initial_sha256", "goal_sha256"):
                    self.assertEqual(ra[key], rb[key])

    def test_sigma_stream_common_and_does_not_consume_flow_rng(self):
        cs = [Collector("B", sigma, 8) for sigma in (.05, .1, .2)]
        for c in cs:
            c.begin(1, 0, 2, np.zeros(2), np.ones(2))
        before = torch.random.get_rng_state().clone()
        samples = [c.action(np.zeros(2, np.float32)) / c.sigma for c in cs]
        np.testing.assert_allclose(samples[0], samples[1])
        np.testing.assert_allclose(samples[1], samples[2])
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))

    def test_leak_and_resample_mutants_rejected(self):
        c, _ = run_arm("B", 0.)
        for field, value in (("trajectory_sha256", "corrupt"), ("u_samples", 1),
                             ("first_u_sha256", "changed"), ("env_seed", 999),
                             ("success", not c.rows[-1]["success"])):
            rows = copy.deepcopy(c.rows)
            rows[-1][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                summarize(rows, 8, "B", 0.)

    def test_missing_duplicate_and_monotone_bits(self):
        c, s = run_arm("A", 0.)
        for rows in (c.rows[:-1], c.rows + [c.rows[0]]):
            with self.assertRaises(ValueError):
                summarize(rows, 8, "A", 0.)
        vals = list(s["oracle_at_k"].values())
        self.assertEqual(vals, sorted(vals))

    def test_flag_off_matches_original_rollout_bitwise(self):
        old = subprocess.check_output(["git", "show", "HEAD:experiments/scratch_lacot_rollout.py"],
                                      cwd=ROOT, text=True)
        a, b = harness(source=old), harness()
        np.random.seed(123)
        ra = a["rollout"](1, True, "unchanged")
        np.random.seed(123)
        rb = b["rollout"](1, True, "unchanged")
        self.assertEqual(ra, rb)
        self.assertEqual(a["env"].history, b["env"].history)
        self.assertEqual(a["DIAG_ROWS"], b["DIAG_ROWS"])

    def test_b_sigma_zero_matches_a_draw_zero_bitwise(self):
        a, _ = run_arm("A", 0., 1)
        b, _ = run_arm("B", 0., 1)
        self.assertEqual([r["trajectory_sha256"] for r in a.rows],
                         [r["trajectory_sha256"] for r in b.rows])
        self.assertEqual([r["u_sequence_sha256"] for r in a.rows],
                         [r["u_sequence_sha256"] for r in b.rows])

    def test_output_counts_and_exclusive_write(self):
        c, _ = run_arm("B", 0.)
        # Isolated CPU fixture artifact, never mixed into real smoke outputs.
        temp = tempfile.mkdtemp(prefix="ucontrast-cpu-")  # retain fixtures; no deletes
        p = Path(temp) / "result.json"
        r = write_result(p, dict(ckpt="TOY", tag="CPU_ONLY"), c)
        self.assertEqual(list(r)[-3:], ["n_tasks", "n_draws", "n_success"])
        self.assertTrue(r["training_use_prohibited"])
        with self.assertRaises(FileExistsError):
            write_result(p, dict(ckpt="TOY", tag="CPU_ONLY"), c)

    def test_frozen_checkpoint_and_new_anchor(self):
        p = json.loads((ROOT / "experiments/_workorders/ucontrast1/provenance.template.json").read_text())
        self.assertEqual(calibration_contract(p)["expected_r1"], .35)
        self.assertEqual(checkpoint(33, p)["sha256"], p["checkpoints"]["33"]["sha256"])

    def test_frozen_checkpoint_rejects_wrong_path_or_hash(self):
        p = json.loads((ROOT / "experiments/_workorders/ucontrast1/provenance.template.json").read_text())
        p["checkpoints"]["33"]["sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            checkpoint(33, p)
        p["checkpoints"]["33"]["sha256"] = checkpoint(33, json.loads(
            (ROOT / "experiments/_workorders/ucontrast1/provenance.template.json").read_text()))["sha256"]
        p["checkpoints"]["33"]["path"] = "wrong.pt"
        with self.assertRaises(ValueError):
            checkpoint(33, p)

    def test_gpu_flagoff_proof_missing_blocks(self):
        missing = Path(tempfile.mkdtemp(prefix="ucontrast-no-gpu-proof-")) / "proof.json"
        with self.assertRaises(FileNotFoundError):
            flagoff_contract(missing, 33)

    def test_sweep_task_and_seed_triples_disjoint(self):
        c, _ = run_arm("B", .1, 2, SWEEP_EPISODE_OFFSET, SWEEP_SALT)
        main = {(t, e, 7 * t + e + 1_000_003 * d)
                for t in (1, 2) for e in range(40) for d in range(2)}
        self.assertFalse(main.intersection((r["task"], r["episode"], r["stream_seed"])
                                           for r in c.rows))
        self.assertTrue(all(r["env_seed"] == c.reset_seed(r["task"], r["episode"])
                            for r in c.rows))

    def test_reset_global_jitter_is_paired(self):
        a, _ = run_arm("A", 0., 3)
        b, _ = run_arm("B", .1, 3)
        self.assertEqual([(r["initial_sha256"], r["goal_sha256"]) for r in a.rows],
                         [(r["initial_sha256"], r["goal_sha256"]) for r in b.rows])

    def test_positive_sigma_changes_real_step_trajectory(self):
        zero, _ = run_arm("B", 0., 3)
        noisy, _ = run_arm("B", .1, 3)
        self.assertTrue(any(a["trajectory_sha256"] != b["trajectory_sha256"]
                            for a, b in zip(zero.rows, noisy.rows)))

    def test_independent_oracle_all_draws(self):
        _, summary = run_arm("A", 0., 8)
        oracle, quality, n_success = independent_oracle(summary["tasks"], 8)
        self.assertEqual(oracle, summary["oracle_at_k"])
        self.assertEqual(quality, summary["per_draw_quality"])
        self.assertEqual(n_success, summary["n_success"])

    def test_r1_rollout_rate_equals_collector_per_draw(self):
        c = run_oracle_loop()
        self.assertEqual(len(c.rows), 2 * 2 * 3)

    def test_oracle_independent_formula_catches_draw0_only(self):
        c, summary = run_arm("A", 0., 3)
        rows = copy.deepcopy(c.rows)
        # Force a valid task whose draw 1 alone succeeds; the independent
        # oracle must count it at @2 and @3.
        key = (rows[0]["task"], rows[0]["episode"])
        for r in rows:
            if (r["task"], r["episode"]) == key:
                r["success"] = r["draw"] == 1
        altered = summarize(rows, 3, "A", 0.)
        oracle, _, _ = independent_oracle(altered["tasks"], 3)
        self.assertEqual(oracle["2"], altered["oracle_at_k"]["2"])
        self.assertGreater(oracle["2"], oracle["1"])


if __name__ == "__main__":
    unittest.main()
