import unittest

import numpy as np

from rb2301_ca1.expert_policy import ConeExpertPolicy, cone_clearance


class ConeExpertTests(unittest.TestCase):
    def observation(self):
        return np.ones(42, dtype=np.float32)

    def test_clear_world_moves_forward(self):
        action = ConeExpertPolicy().predict(self.observation())
        self.assertTrue(np.allclose(action, [1.0, 0.0, 0.0]))

    def test_front_block_selects_left(self):
        observation = self.observation()
        observation[[0, 1, 2, 34, 35]] = 0.02
        action = ConeExpertPolicy().predict(observation)
        self.assertTrue(np.allclose(action, [0.0, 1.0, 0.0]))

    def test_side_commitment_is_retained(self):
        expert = ConeExpertPolicy()
        observation = self.observation()
        observation[[0, 1, 2, 34, 35]] = 0.02
        self.assertGreater(expert.predict(observation)[1], 0.0)
        observation[[8, 9, 10]] = 0.02
        observation[[26, 27, 28]] = 1.0
        action = expert.predict(observation)
        self.assertLess(action[1], 0.0)

    def test_ninety_degree_cones_cover_diagonal_rays(self):
        scan = np.ones(36, dtype=np.float32)
        scan[4] = 0.2
        self.assertAlmostEqual(cone_clearance(scan, 0.0, 90.0), 0.2)


if __name__ == "__main__":
    unittest.main()
