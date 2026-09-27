"""Regression for trajectory-level G×K validation splitting."""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.gk_scan.gk_common import load_segments_with_obs


class TrajectorySplitTest(unittest.TestCase):
    def test_regular_episode_grid_keeps_trajectories_together(self):
        total_steps = 10 * 201
        terminals = np.zeros(total_steps, dtype=bool)
        terminals[200::201] = True
        with tempfile.TemporaryDirectory() as tmp:
            data_path = Path(tmp) / "regular_episodes.npz"
            np.savez(data_path, actions=np.zeros((total_steps, 8), dtype=np.float32),
                     observations=np.zeros((total_steps, 29), dtype=np.float32),
                     terminals=terminals)
            data = load_segments_with_obs(data_path, split_seed=7, val_frac=0.3)

        self.assertFalse(data["fallback_segmentation_used"])
        episode_ids = data["seg_starts"] // 201
        self.assertFalse(set(episode_ids[data["train_idx"]]) &
                         set(episode_ids[data["val_idx"]]))

    def test_trajectory_holdout_and_legacy_segment_indices(self):
        lengths = [8, 12, 16, 8, 12, 16, 8, 12, 16, 8]
        total_steps = sum(lengths)
        terminals = np.zeros(total_steps, dtype=bool)
        terminals[np.cumsum(lengths) - 1] = True
        actions = np.zeros((total_steps, 8), dtype=np.float32)
        observations = np.zeros((total_steps, 29), dtype=np.float32)

        with tempfile.TemporaryDirectory() as tmp:
            data_path = Path(tmp) / "episodes.npz"
            np.savez(data_path, actions=actions, observations=observations,
                     terminals=terminals)

            default = load_segments_with_obs(data_path, split_seed=7, val_frac=0.3)
            trajectory = load_segments_with_obs(
                data_path, split_seed=7, val_frac=0.3, split_by="trajectory"
            )
            segment = load_segments_with_obs(
                data_path, split_seed=7, val_frac=0.3, split_by="segment"
            )

        episode_ends = np.flatnonzero(terminals)
        episode_ids = np.searchsorted(episode_ends, trajectory["seg_starts"])
        train_episodes = set(episode_ids[trajectory["train_idx"]])
        val_episodes = set(episode_ids[trajectory["val_idx"]])
        self.assertEqual(len(val_episodes), int(len(lengths) * 0.3))
        self.assertFalse(train_episodes & val_episodes)
        np.testing.assert_array_equal(default["train_idx"], trajectory["train_idx"])
        np.testing.assert_array_equal(default["val_idx"], trajectory["val_idx"])

        permutation = np.random.default_rng(7).permutation(len(segment["seg_starts"]))
        n_val = int(len(permutation) * 0.3)
        np.testing.assert_array_equal(segment["val_idx"], permutation[:n_val])
        np.testing.assert_array_equal(segment["train_idx"], permutation[n_val:])


if __name__ == "__main__":
    unittest.main()
