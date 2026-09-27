#!/usr/bin/env python3
"""Two-way wiring checks through the same decoder, snapshot and env path as C1."""
import unittest

import numpy as np
import torch

import run_c1 as c1


class WiringTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(19)
        cls.model = c1.bc.BehaviorCodecFactorizedVQVAE(8, 29, 4, 2, 4, D=4, hidden=32)
        cls.model.eval()
        with torch.no_grad():
            for group in cls.model.vq.groups:
                group.embed.mul_(15)
        cls.mu = np.zeros(29, dtype=np.float32)
        cls.sd = np.ones(29, dtype=np.float32)
        cls.raw = c1.wv.load_npz(c1.os.environ.get("OGBENCH_DATA_DIR", "/home/cymaxwelllee/.ogbench/data"),
                                 c1.DATASET, "val")
        cls.env = c1.wv.make_env(c1.os.environ.get("OGBENCH_DATA_DIR", "/home/cymaxwelllee/.ogbench/data"),
                                 c1.DATASET)
        c1.reset_to_dataset(cls.env, cls.raw, 0)
        cls.snap = c1.snapshot(cls.env)
        cls.codes = [(0, 0), (1, 1), (2, 2), (3, 3)]

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def cells(self):
        return [c1.execute_code(self.env, self.snap, self.model, code, self.mu, self.sd)
                for code in self.codes]

    def test_shat_is_observation_only_and_wrong_mapping_is_detected(self):
        good = self.cells()
        wrong = [dict(c) for c in good]
        # Deliberately miswire ŝ while preserving its prediction multiset.
        for i in range(4):
            wrong[i] = c1.execute_code(self.env, self.snap, self.model, self.codes[i],
                                       self.mu, self.sd, shat_override=good[(i + 1) % 4]["shat_norm"])
            np.testing.assert_array_equal(good[i]["actions"], wrong[i]["actions"])
            np.testing.assert_array_equal(good[i]["actual"], wrong[i]["actual"])
        a = c1.start_stats(good, self.raw["observations"][4], self.sd, [1, 2, 3, 0])
        b = c1.start_stats(wrong, self.raw["observations"][4], self.sd, [1, 2, 3, 0])
        with self.assertRaises(AssertionError):  # wrong ŝ/code pairing really FAILS
            np.testing.assert_allclose(a["gait"]["numerator"], b["gait"]["numerator"], rtol=0, atol=1e-10)
        np.testing.assert_allclose(a["gait"]["numerator"],
                                   c1.start_stats(good, self.raw["observations"][4], self.sd,
                                                  [1, 2, 3, 0])["gait"]["numerator"])

    def test_wrong_payload_changes_actions_and_edelta(self):
        good = self.cells()
        wrong = [dict(c) for c in good]
        # Feed the wrong full tuple into the actual decoder/env, not merely metadata.
        wrong[1] = c1.execute_code(self.env, self.snap, self.model, (0, 0), self.mu, self.sd)
        with self.assertRaises(AssertionError):  # wrong action payload really FAILS
            np.testing.assert_array_equal(good[1]["actions"], wrong[1]["actions"])
        a = c1.start_stats(good, self.raw["observations"][4], self.sd, [1, 2, 3, 0])
        b = c1.start_stats(wrong, self.raw["observations"][4], self.sd, [1, 2, 3, 0])
        with self.assertRaises(AssertionError):  # wrong payload's E_delta really FAILS
            np.testing.assert_allclose(a["gait"]["numerator"], b["gait"]["numerator"], rtol=0, atol=1e-10)
        repaired = c1.execute_code(self.env, self.snap, self.model, self.codes[1], self.mu, self.sd)
        np.testing.assert_array_equal(good[1]["actions"], repaired["actions"])
        np.testing.assert_array_equal(good[1]["actual"], repaired["actual"])

    def test_pair_denominator_and_whole_tuple_donors(self):
        base = np.zeros(29, dtype=np.float64)
        base[3] = 1.0  # wxyz identity quaternion
        cells = []
        for i in range(4):
            actual = base.copy()
            actual[0] = i
            cells.append(dict(obs_start=base.tolist(), shat=[base.tolist()] * 4,
                              actual=[actual.tolist()] * 4))
        stats = c1.start_stats(cells, base, self.sd, [1, 2, 3, 0])
        # Six code pairs: (0-1)^2+(0-2)^2+...+(2-3)^2 = 20.
        self.assertEqual(stats["xy"]["denominator"], 20.0)
        self.assertEqual(stats["xy"]["numerator"], 20.0)  # constant predictor E_delta=1
        self.assertEqual(stats["xy"]["permutation_numerator"], 20.0)

        raw = self.raw
        pool = c1.wv.cut_segments(raw, 4)
        codes = [(int(i % 4), int((i + 1) % 4)) for i in range(len(pool))]
        eps, ends = c1.wv.episode_bounds(raw["terminals"])
        episode_of = np.empty(len(raw["actions"]), dtype=np.int32)
        for ei, (a, b) in enumerate(zip(eps, ends)):
            episode_of[a:b + 1] = ei
        pool_obs = raw["observations"][pool]
        pool_actions = raw["actions"][pool[:, None] + np.arange(4)].reshape(len(pool), -1)
        donors = c1.choose_donors(pool, codes, pool_obs, pool_actions, episode_of,
                                   int(pool[0]), pool_obs[0], codes[0], self.mu, self.sd)
        self.assertEqual(len(donors), 3)
        for donor in donors:
            self.assertEqual(len(donor["code"]), 2)
            self.assertEqual(donor["code"], codes[np.searchsorted(pool, donor["segment"])])
            self.assertNotEqual(episode_of[donor["segment"]], episode_of[pool[0]])


if __name__ == "__main__":
    unittest.main()
