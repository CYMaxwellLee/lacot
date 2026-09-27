#!/usr/bin/env python3
"""Regression coverage for incomplete-grid recommendations and A/B rendering."""
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HIP_VISIBLE_DEVICES"] = ""

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.gk_scan import aggregate as gk_aggregate
from experiments.gk_scan import render_ab as gk_render
from experiments.humanoid_dict import aggregate as hd_aggregate
from experiments.humanoid_dict import render_ab as hd_render


GS = [1, 2, 4, 8, 16, 32]
KS = [8, 16, 32, 64]


def _result():
    return {
        "metrics": {
            "baseline_mse_val": 1.0,
            "recon_mse_val": 0.2,
            "recon_mse_train": 0.2,
            "val_improvement_over_baseline_pct": 80.0,
            "any_group_collapsed": False,
            "total_steps_used": 1,
            "restarts_total": 0,
        },
        "factor_validity": {
            "min_delta_mse": 0.1,
            "mean_delta_mse": 0.1,
            "max_delta_mse": 0.1,
            "mean_abs_offdiag_corr": None,
            "mean_top_dim_energy_frac": 1.0,
        },
        "plateau": {"extended": False},
        "timing": {"train_seconds": 0.1},
    }


def _write_grid(results_dir, full):
    results_dir.mkdir(parents=True)
    cells = [(G, K) for G in GS for K in KS] if full else [(1, 8)]
    for G, K in cells:
        (results_dir / f"G{G}_K{K}.json").write_text(json.dumps(_result()))


def _run_aggregate(aggregate, results_dir, full=False):
    draw = mock.patch.object(aggregate.gc if aggregate is gk_aggregate else aggregate.hc,
                             "draw_knee_curve", return_value=None)
    _write_grid(results_dir, full=full)
    with mock.patch.object(aggregate, "RESULTS_DIR", str(results_dir)), draw:
        with contextlib.redirect_stdout(io.StringIO()):
            aggregate.main()
    return json.loads((results_dir / "summary.json").read_text())


def _render_with_summary(render, tmpdir, summary):
    results_dir = Path(tmpdir) / "results"
    results_dir.mkdir()
    (results_dir / "summary.json").write_text(json.dumps(summary))
    output = io.StringIO()
    with mock.patch.object(render, "HERE", tmpdir), mock.patch.object(sys, "argv", ["render_ab.py"]):
        with contextlib.redirect_stdout(output):
            render.main()
    return output.getvalue()


class IncompleteGridRecommendationTests(unittest.TestCase):
    def test_aggregators_with_missing_cells_do_not_recommend(self):
        for aggregate in (gk_aggregate, hd_aggregate):
            with self.subTest(aggregate=aggregate.__name__), tempfile.TemporaryDirectory() as tmpdir:
                summary = _run_aggregate(aggregate, Path(tmpdir) / "results", full=False)
                self.assertEqual(len(summary["missing_cells"]), 23)
                self.assertIsNone(summary["recommendation"]["cell"])
                self.assertTrue(summary["incomplete"])

    def test_aggregators_with_full_grid_still_recommend(self):
        for aggregate in (gk_aggregate, hd_aggregate):
            with self.subTest(aggregate=aggregate.__name__), tempfile.TemporaryDirectory() as tmpdir:
                summary = _run_aggregate(aggregate, Path(tmpdir) / "results", full=True)
                self.assertEqual(summary["missing_cells"], [])
                self.assertFalse(summary["incomplete"])
                self.assertIsNotNone(summary["recommendation"]["cell"])

    def test_renderers_reject_incomplete_summary_even_if_it_has_a_cell(self):
        missing = [f"G{G}_K{K}" for G in GS for K in KS][1:]
        summary = {
            "missing_cells": missing,
            "incomplete": True,
            "recommendation": {"cell": {"G": 1, "K": 8}, "fallback": False},
        }
        for render in (gk_render, hd_render):
            with self.subTest(render=render.__name__), tempfile.TemporaryDirectory() as tmpdir:
                with self.assertRaisesRegex(ValueError, "不完整.*23"):
                    _render_with_summary(render, tmpdir, summary)

    def test_renderers_reject_null_recommendation_with_a_message(self):
        summary = {
            "missing_cells": [],
            "incomplete": False,
            "recommendation": {"cell": None, "fallback": False},
        }
        for render in (gk_render, hd_render):
            with self.subTest(render=render.__name__), tempfile.TemporaryDirectory() as tmpdir:
                with self.assertRaisesRegex(ValueError, "沒有推薦格"):
                    _render_with_summary(render, tmpdir, summary)

    def test_renderers_accept_complete_summary_and_continue_to_checkpoint(self):
        summary = {
            "missing_cells": [],
            "incomplete": False,
            "recommendation": {"cell": {"G": 1, "K": 8}, "fallback": False},
        }
        for render in (gk_render, hd_render):
            with self.subTest(render=render.__name__), tempfile.TemporaryDirectory() as tmpdir:
                with self.assertRaises(SystemExit) as raised:
                    _render_with_summary(render, tmpdir, summary)
                self.assertEqual(raised.exception.code, 1)


if __name__ == "__main__":
    unittest.main()
