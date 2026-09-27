#!/usr/bin/env python3
"""Paired comparisons require a verifiable task identity on both sides."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lacot.dev_eval import paired_diff


class PairedTaskIdTests(unittest.TestCase):
    def test_missing_task_id_on_both_sides_raises(self):
        a = [{"idx": 0, "success": True}]
        b = [{"idx": 0, "success": False}]
        with self.assertRaisesRegex(AssertionError, "task_id"):
            paired_diff(a, b, boot=1)

    def test_missing_task_id_on_either_side_raises(self):
        row = {"idx": 0, "success": True, "task_id": 0}
        missing = {"idx": 0, "success": False}
        for a, b in ((row, missing), (missing, row)):
            with self.subTest(a=a, b=b), self.assertRaisesRegex(AssertionError, "task_id"):
                paired_diff([a], [b], boot=1)

    def test_matching_task_id_is_allowed(self):
        a = [{"idx": 0, "task_id": 0, "success": True}]
        b = [{"idx": 0, "task_id": 0, "success": False}]
        self.assertEqual(paired_diff(a, b, boot=1)["n_a_only"], 1)

    def test_mismatched_task_id_raises(self):
        a = [{"idx": 0, "task_id": 0, "success": True}]
        b = [{"idx": 0, "task_id": 1, "success": False}]
        with self.assertRaisesRegex(AssertionError, "task_id"):
            paired_diff(a, b, boot=1)


if __name__ == "__main__":
    unittest.main()
