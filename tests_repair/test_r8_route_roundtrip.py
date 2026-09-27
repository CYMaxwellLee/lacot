#!/usr/bin/env python3
"""Route result names must survive the writer, shell count, and aggregator."""
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HIP_VISIBLE_DEVICES"] = ""
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "route_dict"))
import aggregate
import run_cell


class FakeModel:
    def eval(self):
        return self

    def decode_code(self, ids):
        return torch.zeros(len(ids), 32)


class RouteRoundtrip(unittest.TestCase):
    def test_writer_aggregator_and_shell_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "data").mkdir()
            (root / "data" / "antmaze-medium-stitch-v0.npz").touch()
            data = dict(segs=np.zeros((100, 32)), train_idx=np.arange(80),
                        val_idx=np.arange(80, 100), seg_dim=32,
                        extraction=dict(grid_ok=True, n_degenerate_rotation=0,
                                        n_segments_per_episode_mean=1,
                                        n_segments_per_episode_min=1,
                                        n_segments_per_episode_max=1))
            metrics = dict(recon_mse_train=0.1, recon_mse_val=0.2,
                           baseline_mse_val=1.0, val_improvement_over_baseline_pct=80,
                           perplexity=4.0, n_active_codes=8, top1_usage_frac=0.2,
                           top3_usage_frac=0.5, collapsed=False, restarts_total=0,
                           total_steps_used=1, usage_counts=[1] * 8,
                           usage_frac=[0.125] * 8)
            results = dict(metrics=metrics, timing=dict(train_seconds=0),
                           plateau=dict(extended=False))
            argv = ["run_cell.py", "--k", "8", "--steps", "1", "--lr", "0.30000001",
                    "--data-dir", str(root / "data"), "--out-dir", str(root)]
            with mock.patch.object(sys, "argv", argv), \
                 mock.patch.object(run_cell.rc, "load_route_data", return_value=data), \
                 mock.patch.object(run_cell.rc, "train_and_eval", return_value=(FakeModel(), results, np.zeros(100, dtype=int))), \
                 mock.patch.object(run_cell.rc, "draw_route_words"), \
                 mock.patch.object(run_cell.rc, "draw_code_examples"), \
                 contextlib.redirect_stdout(io.StringIO()):
                run_cell.main()
            files = list((root / "results").glob("K*.json"))
            self.assertEqual(len(files), 1)
            self.assertIn("0.30000001", files[0].name)
            with mock.patch.object(aggregate, "RESULTS_DIR", str(root / "results")), \
                 contextlib.redirect_stdout(io.StringIO()):
                aggregate.main()
            summary = json.loads((root / "results" / "summary.json").read_text())
            self.assertEqual([c["K"] for c in summary["cells"]], [8])
            self.assertEqual(summary["missing_cells"], ["K16", "K32"])
            line = next(x for x in (ROOT / "experiments/route_dict/run_all.sh").read_text().splitlines()
                        if "jsons found:" in x)
            counted = subprocess.run(["bash", "-c", line], cwd=root, text=True,
                                     capture_output=True, check=True)
            self.assertIn("jsons found: 1", counted.stdout)


if __name__ == "__main__":
    unittest.main()
