"""GK drift report must not award a win when any measured chunk is non-finite."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

import numpy as np


SOURCE = Path(__file__).resolve().parents[1] / "experiments/bcodec_v0/bcodec_gk_drift_eval.py"
spec = importlib.util.spec_from_file_location("bcodec_gk_drift_eval_repair", SOURCE)
gk = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gk)


def run_main(xy, recon_a, recon_s):
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        ruler = root / "ruler.json"
        ruler.write_text("{}")
        tasks = [dict(episode=i, s0=0, e0=4, M=4, n_chunks=1) for i in range(len(xy))]
        samples = iter(zip(xy, recon_a, recon_s))

        def fake_reset(*args, **kwargs):
            x, a, s = next(samples)
            return dict(xy_err=np.array([x]), recon_a_mse=np.array([a]),
                        recon_s_mse=np.array([s]), xy_err_garbage_k0=0.1)

        argv = [str(SOURCE), "--ckpt", "unused", "--ruler", str(ruler),
                "--out-dir", str(root), "--n-traj", str(len(tasks)), "--tag", "repair"]
        output = io.StringIO()
        with patch.object(sys, "argv", argv), \
             patch.object(gk.bc, "load_bcodec_gk_ckpt", return_value=(object(), dict(seg_len=4, G=2, K=16, D=4, lam_s=1), None, None)), \
             patch.object(gk.wv, "load_npz", return_value={}), \
             patch.object(gk.wv, "make_env", return_value=type("Env", (), {"unwrapped": object()})()), \
             patch.object(gk.trk32, "build_tasks", return_value=(tasks, [])), \
             patch.object(gk, "run_reset_chunks", side_effect=fake_reset), \
             contextlib.redirect_stdout(output):
            gk.main()
        return json.loads((root / "repair_summary.json").read_text()), output.getvalue()


def test_nonfinite_measurements_are_suspect():
    report, output = run_main([0.01, 0.02, np.inf], [0.001, np.nan, 0.002], [0.003, np.inf, 0.004])
    assert "SUSPECT" in report["verdict"], f"non-finite chunks incorrectly won: {report['verdict']}"
    for name in ("e_indep_baseline", "recon_a_baseline", "recon_s_baseline"):
        assert report[name]["n_total"] == 3
        assert report[name]["n_finite"] == 2
        keys = ("xy_p25", "xy_p50", "xy_p75", "xy_mean") if name == "e_indep_baseline" else ("p25", "p50", "p75", "mean")
        assert np.isfinite([report[name][key] for key in keys]).all()
    assert "⚠️" in output


def test_all_finite_measurements_can_win():
    report, _ = run_main([0.01, 0.02, 0.025], [0.001] * 3, [0.003] * 3)
    assert report["verdict"].startswith("全勝")
    assert report["e_indep_baseline"]["n_finite"] == report["e_indep_baseline"]["n_total"] == 3


if __name__ == "__main__":
    test_nonfinite_measurements_are_suspect()
    test_all_finite_measurements_can_win()
    print("PASS: GK main filters non-finite measurements and guards verdict")
