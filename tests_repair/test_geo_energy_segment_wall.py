"""A segment crossing a wall must cost more than a clear detour."""
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lacot.refine_grad import GeoEnergy


class GeoEnergySegmentWallTest(unittest.TestCase):
    def setUp(self):
        x, y = np.meshgrid(np.arange(0, 4.001, 0.125),
                           np.arange(0, 2.001, 0.125), indexing="ij")
        free = (x <= 0.5) | (x >= 3.5) | (y <= 0.5)
        obs = np.stack((x[free], y[free]), axis=1)
        self.geo = GeoEnergy(obs, mu=[0, 0], sd=[1, 1], res=8)
        self.start = torch.tensor([[0.25, 1.5]])
        self.goal = torch.tensor([[3.75, 1.5]])
        self.straight = torch.tensor([[[0.25, 1.5], [3.75, 1.5]]],
                                     requires_grad=True)
        self.detour = torch.tensor([[[0.25, 1.5], [0.25, 0.25],
                                     [3.75, 0.25], [3.75, 1.5]]])

    def test_wall_between_free_endpoints_is_penalized(self):
        geo = self.geo
        self.assertLess(float(geo.wall_depth(self.straight).max()), 1e-6)
        self.assertGreater(float(geo.wall_depth(self.straight[:, :1] * 0.5
                                                + self.straight[:, 1:] * 0.5).item()), 0.875)
        direct_energy, direct_terms = geo(self.straight, self.start, self.goal,
                                          per_term=True)
        detour_energy, _ = geo(self.detour, self.start, self.goal, per_term=True)
        self.assertGreater(float(direct_terms["wall"].item()), 0.0)
        self.assertGreater(float(direct_energy.item()), float(detour_energy.item()))
        direct_terms["wall"].sum().backward()
        self.assertTrue(torch.isfinite(self.straight.grad).all())
        self.assertGreater(float(self.straight.grad.abs().sum()), 0.0)
        geo.wall_interp_k = 0
        self.assertEqual(float(geo(self.straight, self.start, self.goal,
                                   per_term=True)[1]["wall"].item()), 0.0)
        geo.wall_interp_k = 1
        self.assertGreater(float(geo(self.straight, self.start, self.goal,
                                     per_term=True)[1]["wall"].item()), 0.0)

    def test_clear_path_energy_matches_original_point_sampling(self):
        geo = self.geo
        energy, terms = geo(self.detour, self.start, self.goal, per_term=True)
        self.assertEqual(tuple(energy.shape), (1,))
        old_wall = geo.wall_depth(self.detour).mean(1)
        old_energy = (geo.w["wall"] * old_wall + geo.w["length"] * terms["length"])
        self.assertLess(float(old_wall.max()), 1e-6)
        self.assertTrue(torch.allclose(energy, old_energy, atol=1e-5, rtol=0))


if __name__ == "__main__":
    unittest.main()
