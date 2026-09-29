"""S2 prosecutor mutation driver. Never touches the inspected files.

Usage: driver.py <mutant_dir>
  <mutant_dir>/test_wiring.py  (copy of delivered test, maybe text-patched)
  <mutant_dir>/collector.py    (copy of delivered collector, maybe patched)
  env MUT_SOURCE=<path>        (optional patched copy of the main rollout file)
"""
import importlib.util
import os
import sys
import unittest
from pathlib import Path

WORK = "/home/cymaxwelllee/Projects/lacot/experiments/_workorders/ucontrast1"
mdir = sys.argv[1]
sys.path[:0] = [mdir, WORK]          # mutant collector first; real run.py from WORK
spec = importlib.util.spec_from_file_location("test_wiring", os.path.join(mdir, "test_wiring.py"))
mod = importlib.util.module_from_spec(spec)
sys.modules["test_wiring"] = mod
spec.loader.exec_module(mod)
import collector as _c               # noqa: E402  (prove which collector was bound)
assert os.path.dirname(os.path.abspath(_c.__file__)) == os.path.abspath(mdir), _c.__file__
if os.environ.get("MUT_SOURCE"):
    mod.SOURCE = Path(os.environ["MUT_SOURCE"])
suite = unittest.defaultTestLoader.loadTestsFromTestCase(mod.Wiring)
with open(os.devnull, "w") as devnull:
    stdout = sys.stdout
    sys.stdout = devnull             # silence rollout prints
    try:
        res = unittest.TextTestRunner(verbosity=0, stream=devnull).run(suite)
    finally:
        sys.stdout = stdout
bad = [("FAIL", t.id().split(".")[-1], (m.strip().splitlines() or [""])[-1][:110]) for t, m in res.failures]
bad += [("ERROR", t.id().split(".")[-1], (m.strip().splitlines() or [""])[-1][:110]) for t, m in res.errors]
print(f"ran={res.testsRun} failures={len(res.failures)} errors={len(res.errors)}")
for row in bad:
    print("  ", *row)

raise SystemExit(0 if res.wasSuccessful() else 1)
