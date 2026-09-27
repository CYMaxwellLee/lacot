import unittest
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lacot.vq import TokenVQ


class EmptyVQBatchTest(unittest.TestCase):
    def test_empty_training_batch_returns_zero_without_changing_ema_state(self):
        vq = TokenVQ(n_codes=4, dim=3, dead_steps=2)
        vq.train()
        empty = torch.empty(0, 2, 3, requires_grad=True)
        before = {name: value.clone() for name, value in vq.named_buffers()}

        quantized, loss, stats = vq(empty)

        self.assertEqual(quantized.shape, empty.shape)
        self.assertEqual(loss.item(), 0.0)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(stats["used"], 0)
        self.assertEqual(stats["commit"], 0.0)
        for name, value in vq.named_buffers():
            self.assertTrue(torch.equal(value, before[name]), name)


if __name__ == "__main__":
    unittest.main()
