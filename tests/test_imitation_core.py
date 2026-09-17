import tempfile
import unittest
from pathlib import Path

import numpy as np

from rb2301_ca1.imitation_core import (
    DemonstrationBatch,
    allocate_episode_counts,
    concatenate_demonstrations,
    episode_split_indices,
    load_demonstrations,
    save_demonstrations,
)


def make_batch(episode_offset=0):
    return DemonstrationBatch(
        observations=np.ones((4, 42), dtype=np.float32),
        expert_actions=np.zeros((4, 3), dtype=np.float32),
        executed_actions=np.zeros((4, 3), dtype=np.float32),
        episode_ids=np.asarray([episode_offset, episode_offset, episode_offset + 1, episode_offset + 1]),
        layout_seeds=np.asarray([10, 10, 11, 11]),
        interventions=np.asarray([False, False, True, False]),
    )


class ImitationDatasetTests(unittest.TestCase):
    def test_episode_allocation_preserves_total_and_balances_workers(self):
        counts = allocate_episode_counts(503, 4)
        self.assertEqual(counts, (126, 126, 126, 125))
        self.assertEqual(sum(counts), 503)
        self.assertLessEqual(max(counts) - min(counts), 1)

    def test_round_trip_uses_non_pickle_npz(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "demonstrations.npz"
            save_demonstrations(path, make_batch())
            loaded = load_demonstrations(path)
            self.assertTrue(np.array_equal(loaded.observations, make_batch().observations))
            self.assertEqual(loaded.observations.dtype, np.float32)

    def test_concatenation_reindexes_episode_ids(self):
        combined = concatenate_demonstrations([make_batch(), make_batch()])
        self.assertEqual(len(np.unique(combined.episode_ids)), 4)
        self.assertEqual(len(combined.observations), 8)

    def test_validation_split_keeps_whole_episodes_together(self):
        episodes = np.repeat(np.arange(10), 3)
        train, validation = episode_split_indices(episodes, 0.2, seed=2301)
        train_episodes = set(episodes[train])
        validation_episodes = set(episodes[validation])
        self.assertFalse(train_episodes & validation_episodes)
        self.assertEqual(len(validation_episodes), 2)


if __name__ == "__main__":
    unittest.main()
