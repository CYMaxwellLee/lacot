import unittest
import sys
from types import SimpleNamespace
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lacot.model import LaCoTActor, LaCoTActorState


class FakeFlow:
    def nll(self, u, cond):
        return u.new_ones(())

    def sample(self, batch_size, cond):
        return cond.new_zeros(batch_size, 1, 1)


class FakeActionHead:
    def __call__(self, inputs):
        return inputs

    def nll(self, predictions, actions):
        return actions.new_ones(actions.shape[0])


class ZeroRoundLossTest(unittest.TestCase):
    def test_both_model_losses_are_finite_when_rounds_is_zero(self):
        cond = torch.zeros(2, 2)
        u_target = torch.zeros(2, 1, 1)
        actions = torch.zeros(2, 1, 1)

        for model_class in (LaCoTActor, LaCoTActorState):
            model = SimpleNamespace(
                k=1,
                d_model=1,
                flow=FakeFlow(),
                action_head=FakeActionHead(),
                refine_rounds=lambda cond, u0, rounds: [u0],
            )
            with self.subTest(model_class=model_class.__name__):
                total, losses = model_class.losses_given(
                    model, cond, u_target, actions, rounds=0
                )

                self.assertTrue(torch.isfinite(total))
                self.assertEqual(losses["l_cons"], 0.0)
                self.assertEqual(losses["l_act_refine"], 0.0)


if __name__ == "__main__":
    unittest.main()
