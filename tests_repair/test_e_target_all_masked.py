import unittest
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lacot.e_target import CrossAttention


class AllMaskedAttentionTest(unittest.TestCase):
    def test_rejects_sample_with_all_frames_masked(self):
        attention = CrossAttention(d_model=4, num_heads=2)
        query = torch.randn(2, 1, 4)
        context = torch.randn(2, 3, 4)
        mask = torch.tensor([[True, True, True], [False, True, False]])

        with self.assertRaisesRegex(ValueError, "all context frames"):
            attention(query, context, key_padding_mask=mask)


if __name__ == "__main__":
    unittest.main()
