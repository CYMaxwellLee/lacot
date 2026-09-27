"""CPU regression for code information beyond the starting observation."""
import contextlib
import io
import json
from pathlib import Path
import runpy
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch


SCRIPT = Path(__file__).resolve().parents[1] / "experiments" / "bcodec_shat_probe" / "shat_probe.py"


class FakeVQ:
    def __init__(self):
        self.groups = [types.SimpleNamespace(embed=torch.tensor([[-1.0], [1.0]]))]

    def __call__(self, encoded, training=False):
        return None, (encoded[:, :1] > 0).long(), None


class FakeDict:
    G = 1

    def __init__(self):
        self.vq = FakeVQ()

    def parameters(self):
        return []

    def eval(self):
        return self

    def encode(self, states, actions):
        return states


class ShatProbeObsOnlyTest(unittest.TestCase):
    def run_case(self, extra_code_information):
        rng = np.random.default_rng(12 if extra_code_information else 13)
        n_train, n_val = 256, 128
        n = n_train + n_val
        obs = rng.choice([-1.0, 1.0], size=(n, 1)).astype(np.float32)
        code = (rng.choice([-1.0, 1.0], size=(n, 1)).astype(np.float32)
                if extra_code_information else obs.copy())
        noise = rng.normal(0, 0.2, size=(n, 1)).astype(np.float32)
        target = (code if extra_code_information else obs) + noise
        data = dict(a_seg=np.zeros((n, 1), dtype=np.float32), s_seg=code,
                    s_next=target, obs_start=obs,
                    train_idx=np.arange(n_train), val_idx=np.arange(n_train, n))
        common = types.ModuleType("bcodec_common")
        common.DATASET_NAME = "fake"
        common.load_bcodec_gk_ckpt = lambda path: (
            FakeDict(), dict(G=1, K=2, D=1, obs_dim=1, act_dim=1, seg_len=1, lam_s=0),
            torch.zeros(1), torch.ones(1))
        common.load_bcodec_segments = lambda *args, **kwargs: data
        common._norm_seg = lambda values, *args: values

        with tempfile.TemporaryDirectory() as tmp:
            ref = Path(tmp) / "ref.json"
            ref.write_text(json.dumps({"metrics": {"G": 1, "K": 2, "recon_s_val": 0.04}}))
            argv = [str(SCRIPT), "--ckpt", "fake", "--lam1-ref-json", str(ref),
                    "--data-path", "fake", "--tag", "case", "--out-dir", tmp,
                    "--hidden", "16", "--steps", "300", "--batch", "64",
                    "--lr", "0.01", "--seed", "3"]
            torch.set_num_threads(1)
            previous = sys.modules.get("bcodec_common")
            sys.modules["bcodec_common"] = common
            try:
                with patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()):
                    runpy.run_path(str(SCRIPT), run_name="__main__")
            finally:
                if previous is None:
                    del sys.modules["bcodec_common"]
                else:
                    sys.modules["bcodec_common"] = previous
            return json.loads((Path(tmp) / "case_summary.json").read_text())["results"]

    def test_state_determined_code_does_not_add_information(self):
        result = self.run_case(False)
        self.assertGreater(result["ratio_bad_to_real"], 2.0)
        self.assertIn("err_obs_only", result)
        self.assertFalse(result["code_adds_information"])
        self.assertAlmostEqual(result["ratio_full_vs_obsonly"],
                               result["err_obs_only"] / max(result["err_real_val"], 1e-12))

    def test_independent_code_adds_information(self):
        result = self.run_case(True)
        self.assertIn("err_obs_only", result)
        self.assertTrue(result["code_adds_information"])
        self.assertLess(result["err_real_val"], result["err_obs_only"] * 0.9)


if __name__ == "__main__":
    unittest.main()
