"""Run the decode probe's real startup path without entering its training loop."""
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments" / "exp_decode_probe.py"
PYTHON = "/home/cymaxwelllee/Projects/lacot/.venv/bin/python3"


def main():
    with tempfile.TemporaryDirectory() as tmp:
        observations = np.arange(24, dtype=np.float32).reshape(12, 2)
        np.savez(Path(tmp) / "tiny.npz", observations=observations,
                 actions=np.zeros_like(observations),
                 terminals=np.array([False] * 5 + [True] + [False] * 5 + [True]))
        startup = '''from pathlib import Path
import numpy as np
source = Path(__import__("sys").argv[1])
prefix = source.read_text(encoding="utf-8").split("\\ndef sota_mlp", 1)[0]
assert prefix != source.read_text(encoding="utf-8"), "startup boundary missing"
scope = {"__file__": str(source)}
exec(compile(prefix, str(source), "exec"), scope)
rng = np.random.default_rng(0)
if scope["HOLDOUT_MOD"] == 1:
    try:
        scope["make_batch"](rng, held=False, bs=1)
    except ValueError as exc:
        assert "held=False" in str(exc) and "eligible" in str(exc).lower(), str(exc)
    else:
        raise AssertionError("empty training split was accepted")
else:
    assert scope["make_batch"](rng, held=False, bs=1)[0].shape == (1, 6, 2)
    assert scope["make_batch"](rng, held=True, bs=1)[0].shape == (1, 6, 2)
'''
        for holdout in (2, 1):
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OGBENCH_DATA_DIR": tmp,
                   "LACOT_ENV": "tiny", "LACOT_DP_MODE": "sg",
                   "LACOT_DP_HOLDOUT": str(holdout), "LACOT_CHUNK": "4",
                   "LACOT_TCAP": "6"}
            try:
                result = subprocess.run([PYTHON, "-c", startup, str(SCRIPT)], cwd=ROOT,
                                        env=env, text=True, capture_output=True, timeout=5)
            except subprocess.TimeoutExpired as exc:
                raise AssertionError(f"HOLDOUT={holdout} hung in make_batch") from exc
            assert result.returncode == 0, f"HOLDOUT={holdout}: {result.stdout}\n{result.stderr}"


if __name__ == "__main__":
    main()
