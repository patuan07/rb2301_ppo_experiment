import unittest

import numpy as np

from rb2301_ca1.rl_core import (
    action_to_velocity,
    build_observation,
    evaluate_transition,
    observation_bounds,
    preprocess_scan,
    sample_scan_metres,
    smooth_normalized_action,
)


class ScanTests(unittest.TestCase):
    def test_scan_is_exactly_36_normalized_float32_values(self):
        scan = np.linspace(0.0, 12.0, 721, dtype=np.float32)
        observation = preprocess_scan(scan)
        self.assertEqual(observation.shape, (36,))
        self.assertEqual(observation.dtype, np.float32)
        self.assertTrue(np.all(observation >= 0.0))
        self.assertTrue(np.all(observation <= 1.0))

    def test_invalid_values_are_handled_safely(self):
        scan = np.full(720, np.inf, dtype=np.float32)
        scan[0] = np.nan
        metres = sample_scan_metres(scan)
        self.assertEqual(float(metres[0]), 0.0)
        self.assertTrue(np.all(metres[1:] == 10.0))


class ActionTests(unittest.TestCase):
    def test_three_continuous_axes_are_scaled_independently(self):
        velocity = action_to_velocity([0.5, -0.25, 1.0])
        self.assertTrue(np.allclose(velocity, (0.2, -0.1, 0.8)))

    def test_invalid_action_fails(self):
        with self.assertRaises(ValueError):
            action_to_velocity([0.0, 1.0])

    def test_out_of_range_action_is_safely_clipped(self):
        velocity = action_to_velocity([2.0, -2.0, 3.0])
        self.assertTrue(np.allclose(velocity, (0.4, -0.4, 0.8)))

    def test_action_filter_limits_command_jump(self):
        filtered = smooth_normalized_action([1.0, -1.0, 0.5], [0.0, 0.0, 0.0], 0.25)
        self.assertTrue(np.allclose(filtered, [0.25, -0.25, 0.125]))

    def test_action_filter_brakes_immediately_before_turning(self):
        filtered = smooth_normalized_action([0.0, 1.0, 0.0], [1.0, 0.0, 0.0], 0.15)
        self.assertTrue(np.allclose(filtered, [0.0, 0.15, 0.0]))

    def test_action_filter_does_not_instantly_reverse_an_axis(self):
        filtered = smooth_normalized_action([-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], 0.15)
        self.assertTrue(np.allclose(filtered, [0.0, 0.0, 0.0]))


class ObservationTests(unittest.TestCase):
    def test_canonical_observation_bounds_match_42_value_interface(self):
        low, high = observation_bounds(36)
        self.assertEqual(low.shape, (42,))
        self.assertEqual(high.shape, (42,))
        np.testing.assert_array_equal(low[:36], np.zeros(36, dtype=np.float32))
        np.testing.assert_array_equal(low[36:39], [-1.0, -1.0, 0.0])
        np.testing.assert_array_equal(low[39:], [-1.0, -1.0, -1.0])
        np.testing.assert_array_equal(high, np.ones(42, dtype=np.float32))

    def test_observation_contains_scan_goal_and_previous_action(self):
        observation = build_observation(
            np.full(721, 5.0, dtype=np.float32),
            x=0.0,
            y=0.0,
            yaw=np.pi / 2.0,
            goal_x=7.2,
            goal_y=0.0,
            previous_action=[0.1, -0.2, 0.3],
        )
        self.assertEqual(observation.shape, (42,))
        # With the robot facing world +y, the world +x goal lies to its right.
        self.assertTrue(np.allclose(observation[36:38], [0.0, -1.0], atol=1e-6))
        self.assertTrue(np.allclose(observation[-3:], [0.1, -0.2, 0.3]))


class RewardTests(unittest.TestCase):
    def make_transition(self, **changes):
        arguments = dict(
            previous_x=1.0,
            current_x=1.1,
            current_y=0.0,
            minimum_scan=1.0,
            step_count=10,
            goal_x=7.2,
            corridor_half_width=2.15,
            collision_distance=0.2,
            max_episode_steps=300,
        )
        arguments.update(changes)
        return evaluate_transition(**arguments)

    def test_progress_reward(self):
        transition = self.make_transition()
        self.assertAlmostEqual(transition.reward, 0.4975)
        self.assertFalse(transition.terminated)
        self.assertFalse(transition.truncated)

    def test_collision_has_priority_over_success(self):
        transition = self.make_transition(current_x=7.3, minimum_scan=0.1)
        self.assertTrue(transition.terminated)
        self.assertEqual(transition.reason, "collision")
        self.assertLess(transition.reward, 0.0)

    def test_success(self):
        transition = self.make_transition(previous_x=7.1, current_x=7.2)
        self.assertTrue(transition.terminated)
        self.assertEqual(transition.reason, "success")

    def test_time_limit_is_truncation(self):
        transition = self.make_transition(step_count=300)
        self.assertTrue(transition.truncated)
        self.assertEqual(transition.reason, "time_limit")

    def test_leaving_corridor_cannot_be_profitable(self):
        transition = self.make_transition(current_x=7.0, current_y=2.2)
        self.assertTrue(transition.truncated)
        self.assertEqual(transition.reason, "out_of_bounds")
        self.assertLess(transition.reward, 0.0)

    def test_command_changes_and_turning_are_penalized(self):
        smooth = self.make_transition(action_delta_squared=0.0, turn_velocity=0.0)
        abrupt = self.make_transition(action_delta_squared=2.0, turn_velocity=0.8)
        self.assertLess(abrupt.reward, smooth.reward)

    def test_late_collision_reward_is_strongly_negative(self):
        transition = self.make_transition(
            previous_x=7.0,
            current_x=7.1,
            minimum_scan=0.1,
        )
        self.assertLess(transition.reward, -39.0)


if __name__ == "__main__":
    unittest.main()
