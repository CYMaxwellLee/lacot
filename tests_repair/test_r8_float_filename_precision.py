#!/usr/bin/env python3
"""Nearby floating-point settings must name distinct result files."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

from test_r8_default_names import filename

ROOT = Path(__file__).resolve().parents[1]


def route_tag(lr):
    path = ROOT / "experiments/route_dict/run_cell.py"
    tree = ast.parse(path.read_text())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    node = next(n for n in main.body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "tag" for t in n.targets))
    args = SimpleNamespace(k=8, steps=1, batch=1, lr=lr, beta=0.25, latent_dim=16,
                           hidden=512, decay=0.99, dead_steps=500, seed=30008,
                           split_seed=42, val_frac=0.1, arc_length=7.5, n_resample=16,
                           n_cluster_examples=5)
    scope = {"args": args}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), scope)
    return scope["tag"]


def scratch_extra(**kwargs):
    path = ROOT / "experiments/scratch_lacot_rollout.py"
    tree = ast.parse(path.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_tag_extra")
    scope = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), scope)
    return scope["_tag_extra"](**kwargs)


class FloatNames(unittest.TestCase):
    def test_route_nearby_lr(self):
        self.assertNotEqual(route_tag(0.30000001), route_tag(0.30000002))

    def test_value_nearby_weight(self):
        v = dict(ENV_NAME="pointmaze-medium-stitch-v0", T_FIX=64, K=4, EPS=0.5,
                 EVAL_PAIRS=200000, STEPS1=1500, STEPS_V=5000, W_MSE=1.0,
                 W_RANK=1.0, W_NEG=1.0, USE_SG=1, ORACLE=1, SAMPLE="interp",
                 SG_QUERY=1, SEED=0, W_CTR2=0.3, TRAIN_PAIRS=200000, SHUFFLE_CTRL=1)
        self.assertNotEqual(filename("exp_value_u.py", "out", {**v, "W_CTR2": 0.30000001}),
                            filename("exp_value_u.py", "out", {**v, "W_CTR2": 0.30000002}))

    def test_rounds_nearby_ema(self):
        v = dict(SEED=0, STEPS1=1200, STEPS2=3000, PROBE_STEPS=1200,
                 EMA_M=0.996, MAX_ROUNDS=12)
        self.assertNotEqual(filename("exp_refine_rounds.py", "out", {**v, "EMA_M": 0.30000001}),
                            filename("exp_refine_rounds.py", "out", {**v, "EMA_M": 0.30000002}))

    def test_scratch_defaults_and_nearby_floats(self):
        self.assertEqual(scratch_extra(), "")
        for key in ("TEACHER_MIX", "INTENT_DROP", "INTENT_GUID_W"):
            with self.subTest(key=key):
                self.assertNotEqual(scratch_extra(**{key: 0.30000001}),
                                    scratch_extra(**{key: 0.30000002}))
        self.assertNotEqual(scratch_extra(), scratch_extra(INTENT_ZERO=1))

        path = ROOT / "experiments/scratch_lacot_rollout.py"
        tree = ast.parse(path.read_text())
        call = next(n.value for n in tree.body if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == "_extra" for t in n.targets))
        self.assertTrue({"INTENT_ZERO", "INTENT_GUID_W"}.issubset(
            {keyword.arg for keyword in call.keywords}))


if __name__ == "__main__":
    unittest.main()
