"""Regression tests for a decoder that ignores both code and state."""
import contextlib
import io
from pathlib import Path
import runpy
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1] / "experiments" / "bcodec_v0"
SCRIPTS = ("bcodec_shortcut_check.py", "bcodec_gk_shortcut_check.py")


class ConstantDecoder:
    def __init__(self, grouped):
        self.grouped = grouped

    def encode(self, state, action):
        return torch.zeros((len(state), 1))

    def vq(self, encoded, training=False):
        shape = (len(encoded), 2) if self.grouped else (len(encoded),)
        return None, torch.zeros(shape, dtype=torch.long), None

    def decode_codes(self, codes, state):
        return torch.zeros((len(state), 1)), None


class ShortcutRatioTest(unittest.TestCase):
    def test_constant_decoder_is_suspect(self):
        for script in SCRIPTS:
            with self.subTest(script=script), tempfile.TemporaryDirectory() as out_dir:
                grouped = "gk" in script
                saved = []
                common = types.ModuleType("bcodec_common")
                common.DATASET_NAME = "fake"
                cfg = dict(K=2, G=2, seg_len=1, obs_dim=1, split_seed=0, val_frac=0.5)
                loader = lambda path: (ConstantDecoder(grouped), cfg, torch.zeros(1), torch.ones(1))
                common.load_bcodec_ckpt = loader
                common.load_bcodec_gk_ckpt = loader
                common.load_bcodec_segments = lambda *args, **kwargs: dict(
                    val_idx=np.arange(3), s_seg=np.zeros((3, 1), dtype=np.float32),
                    a_seg=np.zeros((3, 1), dtype=np.float32),
                    obs_start=np.arange(3, dtype=np.float32).reshape(3, 1))
                common._norm_seg = lambda values, *args: values
                common.save_json = lambda result, path: saved.append(result)
                argv = [script, "--ckpt", "fake", "--seed", "3", "--tag", "test", "--out-dir", out_dir]
                with patch.dict(sys.modules, {"bcodec_common": common}), patch.object(sys, "argv", argv):
                    with contextlib.redirect_stdout(io.StringIO()):
                        runpy.run_path(str(ROOT / script), run_name="__main__")
                self.assertEqual(len(saved), 1)
                self.assertEqual(saved[0]["metric_a"]["median"], 0)
                self.assertEqual(saved[0]["metric_b"]["median"], 0)
                self.assertIn("SUSPECT", saved[0]["verdict"])


if __name__ == "__main__":
    unittest.main()
