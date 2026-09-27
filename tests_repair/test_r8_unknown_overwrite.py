#!/usr/bin/env python3
"""Historical files without identity fields cannot be silently overwritten."""
import ast
import json
import os
from pathlib import Path
import tempfile
import unittest

SOURCE = Path(__file__).resolve().parents[1] / "experiments/scratch_lacot_rollout.py"


def check_existing(dst, out):
    tree = ast.parse(SOURCE.read_text())
    guard = next(n for n in tree.body if isinstance(n, ast.If)
                 and isinstance(n.test, ast.Call)
                 and isinstance(n.test.func, ast.Attribute)
                 and n.test.func.attr == "exists"
                 and any(isinstance(x, ast.Name) and x.id == "_old" for x in ast.walk(n)))
    exec(compile(ast.Module(body=[guard], type_ignores=[]), str(SOURCE), "exec"),
         {"os": os, "json": json, "dst": str(dst), "out": out})


class UnknownOverwrite(unittest.TestCase):
    def test_missing_identity_is_unknown(self):
        new = dict(guid_w=0.0, intent_zero=0, intent_drop=0.0, bon_n=0)
        with tempfile.TemporaryDirectory() as tmp:
            dst = Path(tmp) / "rollout.json"
            for key in new:
                with self.subTest(key=key):
                    old = dict(new)
                    old.pop(key)
                    dst.write_text(json.dumps(old))
                    with self.assertRaisesRegex(AssertionError, "拒絕覆蓋"):
                        check_existing(dst, new)
            dst.write_text(json.dumps(new))
            check_existing(dst, new)


if __name__ == "__main__":
    unittest.main()
