#!/usr/bin/env python3
"""Default experiment filenames retain their historical indexes."""
import ast
import os
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1] / "experiments"


def filename(script, target, values):
    tree = ast.parse((ROOT / script).read_text())
    node = next(n for n in tree.body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == target for t in n.targets)
                and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Attribute)
                and n.value.func.attr == "join")
    namespace = {"os": os, "__file__": str(ROOT / script), "OUT_DIR": "/tmp", **values}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(ROOT / script), "exec"), namespace)
    return Path(namespace[target]).name


class DefaultNames(unittest.TestCase):
    def test_value_u(self):
        v = dict(ENV_NAME="pointmaze-medium-stitch-v0", T_FIX=64, K=4, EPS=0.5,
                 EVAL_PAIRS=200000, STEPS1=1500, STEPS_V=5000, W_MSE=1.0,
                 W_RANK=1.0, W_NEG=1.0, USE_SG=1, ORACLE=1, SAMPLE="interp",
                 SG_QUERY=1, SEED=0, W_CTR2=0.3, TRAIN_PAIRS=200000, SHUFFLE_CTRL=1)
        self.assertEqual(filename("exp_value_u.py", "out", v),
                         "value_u_pointmaze-medium-stitch-v0_T64_K4_e0.5_p200000_s11500_sv5000"
                         "_wm1_wr1_wn1_sg1_o1_interp_q1_s0.json")

    def test_oracle(self):
        class Model:
            head_kind = "continuous"
        v = dict(K=4, SEED=0, STEPS1=1500, STEPS2=4000, MAXH=1000,
                 EPISODES=20, model=Model())
        self.assertEqual(filename("exp_oracle_true.py", "_out", v),
                         "results_oracle_K4_seed0_continuous.json")

    def test_rounds(self):
        v = dict(SEED=0, STEPS1=1200, STEPS2=3000, PROBE_STEPS=1200,
                 EMA_M=0.996, MAX_ROUNDS=12)
        self.assertEqual(filename("exp_refine_rounds.py", "out", v), "rounds_seed0.json")

    def test_nondefault_settings_change_names(self):
        value = dict(ENV_NAME="pointmaze-medium-stitch-v0", T_FIX=64, K=4, EPS=0.5,
                     EVAL_PAIRS=200000, STEPS1=1500, STEPS_V=5000, W_MSE=1.0,
                     W_RANK=1.0, W_NEG=1.0, USE_SG=1, ORACLE=1, SAMPLE="interp",
                     SG_QUERY=1, SEED=0, W_CTR2=0.3, TRAIN_PAIRS=200000, SHUFFLE_CTRL=1)
        original = filename("exp_value_u.py", "out", value)
        self.assertNotEqual(original, filename("exp_value_u.py", "out", {**value, "W_CTR2": 0.4}))
        self.assertNotEqual(original, filename("exp_value_u.py", "out", {**value, "TRAIN_PAIRS": 100}))
        self.assertNotEqual(original, filename("exp_value_u.py", "out", {**value, "SHUFFLE_CTRL": 0}))

        class Model:
            head_kind = "continuous"
        oracle = dict(K=4, SEED=0, STEPS1=1500, STEPS2=4000, MAXH=1000,
                      EPISODES=20, model=Model())
        original = filename("exp_oracle_true.py", "_out", oracle)
        for key, value in (("STEPS1", 1), ("STEPS2", 1), ("MAXH", 500), ("EPISODES", 2)):
            self.assertNotEqual(original, filename("exp_oracle_true.py", "_out", {**oracle, key: value}))

        rounds = dict(SEED=0, STEPS1=1200, STEPS2=3000, PROBE_STEPS=1200,
                      EMA_M=0.996, MAX_ROUNDS=12)
        original = filename("exp_refine_rounds.py", "out", rounds)
        for key, value in (("STEPS1", 1), ("STEPS2", 1), ("PROBE_STEPS", 1),
                           ("EMA_M", 0.9), ("MAX_ROUNDS", 4)):
            self.assertNotEqual(original, filename("exp_refine_rounds.py", "out", {**rounds, key: value}))


if __name__ == "__main__":
    unittest.main()
