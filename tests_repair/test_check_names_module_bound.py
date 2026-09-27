"""CPU-only regression: parse source without importing or executing experiments."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest


CHECKER = Path(__file__).resolve().parents[1] / "experiments/_check_names.py"
READ_ONLY_REPO = Path("/home/cymaxwelllee/Projects/lacot")


def scan(paths):
    return subprocess.run(
        [sys.executable, "-B", str(CHECKER), *map(str, paths)],
        capture_output=True, text=True,
    )


class ModuleBindingsTest(unittest.TestCase):
    def check_source(self, source, missing=()):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "case.py"
            path.write_text(textwrap.dedent(source), encoding="utf-8")
            result = scan([path])
        self.assertEqual(result.stderr, "")
        self.assertEqual(result.returncode, int(bool(missing)), result.stdout)
        errors = [line for line in result.stdout.splitlines() if "用到 `" in line]
        self.assertEqual(len(errors), len(missing), result.stdout)
        for name in missing:
            self.assertIn(f"用到 `{name}`", result.stdout)

    def test_false_branch(self):
        self.check_source("""
            if False:
                hidden = 1
            print(hidden)
        """, ("hidden",))

    def test_uncalled_function(self):
        self.check_source("""
            def unused():
                hidden = 1
            print(hidden)
        """, ("hidden",))

    def test_if_intersection_and_constants(self):
        self.check_source("""
            import os
            if os.getenv("FLAG"):
                shared = 1
                one_side = 1
            else:
                shared = 2
            if False:
                unreachable = 1
            else:
                live = 1
            if True:
                always = 1
            else:
                dead_else = 1
            print(shared, live, always, one_side, unreachable, dead_else)
        """, ("one_side", "unreachable", "dead_else"))

    def test_zero_iteration_loops(self):
        # 9/27 lead 裁（pragmatic）：模組層 for/while 主體綁定【收】——實務必至少執行一次，
        # 不收＝全倉 26 處假警報。理論上零次執行的漏報風險由人工承擔（lint 工具取捨）。
        self.check_source("""
            for index in []:
                for_value = 1
            while False:
                while_value = 1
            print(index, for_value, while_value)
        """)

    def test_with_no_grad_ed_rand(self):
        self.check_source("""
            import torch
            with torch.no_grad():
                ED_RAND = 0.5
            if ED_RAND > 0:
                print(ED_RAND)
            print(ED_RAND)
        """)

    def test_try_except_raise_datasets(self):
        self.check_source("""
            import ogbench
            ENV_NAME, OGB_DATA = "test", "/tmp"
            try:
                train_ds, val_ds = ogbench.make_env_and_datasets(
                    ENV_NAME, dataset_dir=OGB_DATA, dataset_only=True)
            except Exception as e:
                print(f"download failed: {type(e).__name__}: {e}")
                raise
            for split_name, ds in (("train", train_ds), ("val", val_ds)):
                print(split_name, ds)
        """)

    def test_with_and_try_respect_nested_scopes(self):
        self.check_source("""
            from contextlib import nullcontext
            with nullcontext() as ctx:
                visible = 1
                if False:
                    hidden_if = 1
                def unused():
                    hidden_function = 1
            try:
                value = 1
            except Exception as error:
                handler_only = 1
                raise
            else:
                success = 1
            finally:
                finished = 1
            print(ctx, visible, unused, value, success, finished)
            print(hidden_if, hidden_function, handler_only, error)
        """, ("hidden_if", "hidden_function", "handler_only", "error"))

    def test_real_reported_examples(self):
        paths = [READ_ONLY_REPO / "experiments" / name for name in (
            "exp_value_u.py", "probe_antmaze_a0.py",
        )]
        result = scan(paths)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_read_only_full_repository(self):
        self.assertTrue(READ_ONLY_REPO.is_dir())
        paths = []
        for directory, dirs, files in os.walk(READ_ONLY_REPO):
            # Installed dependencies and Git metadata are not repository sources.
            dirs[:] = [d for d in dirs if d not in {".venv", ".git", "__pycache__"}]
            paths.extend(Path(directory) / f for f in files if f.endswith(".py"))
        self.assertTrue(paths)
        result = scan(sorted(paths))
        # 9/27 lead 裁：已知條件型 pattern（條件塊建立＋同條件 guard 使用）為 checker 的
        # 固有極限，白名單管理；⛔ 白名單外出現任何 ⛔ 即 FAIL。
        allowed = {"scratch_lacot_rollout.py", "exp_refine_probe.py",
                   "probe_theory_checks.py", "ref_c5ca359.py"}
        flagged = {line.split("/")[-1].rstrip() for line in result.stdout.splitlines()
                   if line.startswith("\u26d4") or line.startswith("⛔")}
        unexpected = flagged - allowed
        self.assertFalse(unexpected, f"白名單外的報告: {unexpected}\n" + result.stdout)
        self.assertEqual(result.stderr, "")
        print(f"Read-only repository scan: {len(paths)} files, flagged(whitelisted)={sorted(flagged)}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
