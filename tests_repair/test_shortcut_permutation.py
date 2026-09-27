"""Regression tests for the cross-state derangement in both shortcut checks."""
import ast
import multiprocessing
from pathlib import Path
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1] / "experiments" / "bcodec_v0"
SCRIPTS = ("bcodec_shortcut_check.py", "bcodec_gk_shortcut_check.py")


def run_permutation(script, n, queue, stuck=False):
    tree = ast.parse((ROOT / script).read_text())
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    start = next(i for i, node in enumerate(main.body)
                 if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "perm"
                                                          for t in node.targets))
    end = next(i for i in range(start, len(main.body))
               if isinstance(main.body[i], ast.Assign) and any(isinstance(t, ast.Name) and t.id == "other_obs_t_norm"
                                                                for t in main.body[i].targets))
    block = ast.fix_missing_locations(ast.Module(body=main.body[start:end + 1], type_ignores=[]))
    rng = type("StuckRng", (), {"permutation": lambda self, size: np.arange(size)})() if stuck else np.random.default_rng(3)
    scope = {"np": np, "rng": rng, "n": n,
             "obs_t_norm": np.arange(n)}
    try:
        exec(compile(block, str(ROOT / script), "exec"), scope)
        queue.put(("ok", scope["perm"].tolist()))
    except Exception as exc:
        queue.put((type(exc).__name__, str(exc)))


class ShortcutPermutationTest(unittest.TestCase):
    def run_case(self, script, n, stuck=False):
        queue = multiprocessing.Queue()
        process = multiprocessing.Process(target=run_permutation, args=(script, n, queue, stuck))
        process.start()
        process.join(3)
        if process.is_alive():
            process.terminate()
            process.join()
            self.fail(f"{script} hung for n={n}")
        self.assertEqual(process.exitcode, 0)
        return queue.get(timeout=1)

    def test_derangement(self):
        for script in SCRIPTS:
            with self.subTest(script=script):
                result, value = self.run_case(script, 3)
                self.assertEqual(result, "ok", (script, result, value))
                self.assertEqual(sorted(value), list(range(3)))
                self.assertTrue(all(i != x for i, x in enumerate(value)))

    def test_single_sample_raises(self):
        for script in SCRIPTS:
            with self.subTest(script=script):
                result, value = self.run_case(script, 1)
                self.assertEqual(result, "ValueError", (script, result, value))

    def test_retry_limit(self):
        for script in SCRIPTS:
            with self.subTest(script=script):
                result, value = self.run_case(script, 3, stuck=True)
                self.assertEqual(result, "RuntimeError", (script, result, value))


if __name__ == "__main__":
    unittest.main()
