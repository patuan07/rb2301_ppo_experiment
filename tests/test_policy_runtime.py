"""Tests for the torch-free deployment runtime.

These run on system Python with no PyTorch, no Stable-Baselines3 and no ROS,
which is exactly the environment the robot has.  The equivalence check against
the real checkpoint lives in ``export_policy.py`` instead, keeping this suite
free of the training dependencies.
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from rb2301_ca1.policy_runtime import (
    LASER_MOUNT_YAW,
    POLICY_MAX_RANGE,
    SIM_ANGLE_INCREMENT,
    SIM_RAY_COUNT,
    ActorMLP,
    self_test,
    sim_ray_bearings,
    sim_scan_from_ranges,
    wrap_to_pi,
)


PROJECT = Path(__file__).resolve().parents[1]
POLICY_DIRECTORY = PROJECT / "src/rb2301_ca1/policy"
ACTOR_PATH = POLICY_DIRECTORY / "actor.npz"
REFERENCE_PATH = POLICY_DIRECTORY / "actor_reference.npz"

SIM_ANGLE_MIN = -3.14


def make_actor(weights, biases, activation="tanh"):
    return ActorMLP(
        weights=tuple(np.asarray(w, dtype=np.float32) for w in weights),
        biases=tuple(np.asarray(b, dtype=np.float32) for b in biases),
        activation=activation,
        observation_dim=weights[0].shape[1],
        action_dim=weights[-1].shape[0],
    )


class ActorTests(unittest.TestCase):
    def test_output_is_bounded_and_finite(self):
        rng = np.random.default_rng(0)
        actor = make_actor(
            [rng.normal(size=(8, 4)), rng.normal(size=(3, 8))],
            [rng.normal(size=8), rng.normal(size=3)],
        )
        action = actor.forward(rng.normal(size=4))
        self.assertEqual(action.shape, (3,))
        self.assertEqual(action.dtype, np.float32)
        self.assertTrue(np.all(np.isfinite(action)))
        self.assertTrue(np.all(action >= -1.0))
        self.assertTrue(np.all(action <= 1.0))

    def test_a_saturated_mean_yields_exactly_one(self):
        """The output layer is linear; only the clip bounds the action.

        Stable-Baselines3 returns ``DiagGaussianDistribution.mode()`` -- the raw
        mean -- for this checkpoint, and the bound comes from clipping against
        the action space.  Applying a tanh to the output as well is the single
        most plausible way to misread the inference path, and it would give
        ``tanh(3.0) = 0.995`` here instead of ``1.0``.
        """
        actor = make_actor(
            [np.zeros((4, 2), dtype=np.float32)],
            [np.full(4, 3.0, dtype=np.float32)],
        )
        action = actor.forward(np.zeros(2, dtype=np.float32))
        np.testing.assert_array_equal(action, np.ones(4, dtype=np.float32))

    def test_a_saturated_negative_mean_yields_exactly_minus_one(self):
        actor = make_actor(
            [np.zeros((4, 2), dtype=np.float32)],
            [np.full(4, -3.0, dtype=np.float32)],
        )
        action = actor.forward(np.zeros(2, dtype=np.float32))
        np.testing.assert_array_equal(action, -np.ones(4, dtype=np.float32))

    def test_float64_evaluation_keeps_double_precision(self):
        identity = np.eye(3, dtype=np.float32)
        actor = make_actor(
            [identity * 0.5, identity],
            [np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32)],
        )
        action = actor.forward(np.ones(3), dtype=np.float64)
        self.assertEqual(action.dtype, np.float64)
        np.testing.assert_allclose(action, np.tanh(0.5), rtol=0.0, atol=1e-15)

    def test_relu_activation_is_supported(self):
        identity = np.eye(2, dtype=np.float32)
        actor = make_actor(
            [identity, identity],
            [np.zeros(2, dtype=np.float32), np.zeros(2, dtype=np.float32)],
            activation="relu",
        )
        action = actor.forward(np.asarray([-2.0, 3.0]))
        np.testing.assert_array_equal(action, np.asarray([0.0, 1.0], dtype=np.float32))

    def test_a_wrong_observation_length_is_rejected(self):
        actor = make_actor(
            [np.zeros((2, 4), dtype=np.float32)],
            [np.zeros(2, dtype=np.float32)],
        )
        with self.assertRaises(ValueError):
            actor.forward(np.zeros(5, dtype=np.float32))

    def test_a_non_finite_observation_is_rejected(self):
        actor = make_actor(
            [np.zeros((2, 4), dtype=np.float32)],
            [np.zeros(2, dtype=np.float32)],
        )
        with self.assertRaises(ValueError):
            actor.forward(np.full(4, np.nan, dtype=np.float32))

    def test_a_missing_actor_file_says_how_to_make_one(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "actor.npz"
            with self.assertRaises(FileNotFoundError) as context:
                ActorMLP.load(missing)
        self.assertIn("export_policy", str(context.exception))

    def test_loading_rejects_an_unsupported_activation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "actor.npz"
            np.savez(
                path,
                activation=np.asarray("sigmoid"),
                obs_dim=np.asarray(2, dtype=np.int64),
                act_dim=np.asarray(2, dtype=np.int64),
                w0=np.zeros((2, 2), dtype=np.float32),
                b0=np.zeros(2, dtype=np.float32),
            )
            with self.assertRaises(ValueError) as context:
                ActorMLP.load(path)
        # The message must name the offending activation, not just fail.
        self.assertIn("sigmoid", str(context.exception))

    def test_loading_rejects_a_layer_chain_that_does_not_line_up(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "actor.npz"
            np.savez(
                path,
                activation=np.asarray("tanh"),
                obs_dim=np.asarray(4, dtype=np.int64),
                act_dim=np.asarray(1, dtype=np.int64),
                w0=np.zeros((3, 4), dtype=np.float32),
                b0=np.zeros(3, dtype=np.float32),
                w1=np.zeros((1, 5), dtype=np.float32),  # should be (1, 3)
                b1=np.zeros(1, dtype=np.float32),
            )
            with self.assertRaises(ValueError):
                ActorMLP.load(path)


class ExportedActorTests(unittest.TestCase):
    """Checks that only run once a checkpoint has been exported."""

    @unittest.skipUnless(ACTOR_PATH.is_file(), "no exported actor in policy/")
    def test_exported_actor_matches_the_42_to_3_architecture(self):
        actor = ActorMLP.load(ACTOR_PATH)
        self.assertEqual(actor.observation_dim, 42)
        self.assertEqual(actor.action_dim, 3)
        self.assertEqual(actor.activation, "tanh")
        self.assertEqual(
            [tuple(weight.shape) for weight in actor.weights],
            [(128, 42), (128, 128), (3, 128)],
        )

    @unittest.skipUnless(
        ACTOR_PATH.is_file() and REFERENCE_PATH.is_file(),
        "no exported reference in policy/",
    )
    def test_self_test_passes_against_the_exported_reference(self):
        result = self_test(ActorMLP.load(ACTOR_PATH), REFERENCE_PATH)
        self.assertGreater(result.rows, 0)
        self.assertLessEqual(result.float64_error, result.float64_tolerance)
        self.assertLessEqual(result.float32_error, result.float32_tolerance)
        self.assertTrue(result.passed)


class ScanResamplingTests(unittest.TestCase):
    """Rays are matched by bearing, not by array index."""

    @staticmethod
    def world(bearing):
        return 2.0 + np.sin(bearing) + 0.5 * np.cos(3.0 * bearing)

    @classmethod
    def make_scan(cls, ray_count, angle_min, angle_increment, mount_yaw):
        """Sample the fixed world pattern through one sensor's convention.

        Ranges are a function of the *world* bearing, so a scan taken with a
        different ``angle_min``, sweep direction or mount rotation describes the
        same room and must resample to the same numbers.
        """

        bearings = mount_yaw + angle_min + np.arange(ray_count) * angle_increment
        return cls.world(bearings).astype(np.float32)

    @classmethod
    def reference(cls):
        """The simulated sensor's own convention: the identity conversion."""

        return sim_scan_from_ranges(
            cls.make_scan(SIM_RAY_COUNT, SIM_ANGLE_MIN, SIM_ANGLE_INCREMENT, LASER_MOUNT_YAW),
            angle_min=SIM_ANGLE_MIN,
            angle_increment=SIM_ANGLE_INCREMENT,
            mount_yaw=LASER_MOUNT_YAW,
        ).ranges

    def test_matching_the_simulated_convention_is_the_identity(self):
        """The training path must be untouched, or the policy sees a new world."""

        scan = self.make_scan(
            SIM_RAY_COUNT, SIM_ANGLE_MIN, SIM_ANGLE_INCREMENT, LASER_MOUNT_YAW
        )
        result = sim_scan_from_ranges(
            scan,
            angle_min=SIM_ANGLE_MIN,
            angle_increment=SIM_ANGLE_INCREMENT,
            mount_yaw=LASER_MOUNT_YAW,
        )
        self.assertEqual(result.ranges.shape, (SIM_RAY_COUNT,))
        self.assertEqual(result.invalid_fraction, 0.0)
        self.assertEqual(result.valid_count, SIM_RAY_COUNT)
        np.testing.assert_allclose(result.ranges, scan, atol=1e-6)

    def test_a_different_angle_min_describes_the_same_room(self):
        reference = self.reference()
        for angle_min in (0.0, -np.pi, 1.0, -2.0):
            with self.subTest(angle_min=angle_min):
                result = sim_scan_from_ranges(
                    self.make_scan(360, angle_min, 2.0 * np.pi / 360, 0.0),
                    angle_min=angle_min,
                    angle_increment=2.0 * np.pi / 360,
                    mount_yaw=0.0,
                )
                np.testing.assert_allclose(result.ranges[::4], reference[::4], atol=0.05)

    def test_a_reversed_sweep_describes_the_same_room(self):
        increment = -2.0 * np.pi / 360
        result = sim_scan_from_ranges(
            self.make_scan(360, 2.0 * np.pi + increment, increment, 0.0),
            angle_min=2.0 * np.pi + increment,
            angle_increment=increment,
            mount_yaw=0.0,
        )
        np.testing.assert_allclose(result.ranges[::4], self.reference()[::4], atol=0.05)

    def test_a_rotated_mount_describes_the_same_room(self):
        result = sim_scan_from_ranges(
            self.make_scan(360, 0.0, 2.0 * np.pi / 360, np.pi),
            angle_min=0.0,
            angle_increment=2.0 * np.pi / 360,
            mount_yaw=np.pi,
        )
        np.testing.assert_allclose(result.ranges[::4], self.reference()[::4], atol=0.05)

    def test_the_forward_ray_is_the_one_that_sees_ahead(self):
        """A wall dead ahead must land on ray 0, not ray 360.

        This is the whole reason the conversion exists: the simulated laser is
        mounted with a pi yaw offset, so an index-based copy of a real scan whose
        ``angle_min`` is ``-pi`` would put the obstacle field behind the robot.
        """

        ray_count = 360
        ranges = np.full(ray_count, 8.0, dtype=np.float32)
        ranges[0] = 0.4  # a wall at world bearing zero, i.e. straight ahead
        result = sim_scan_from_ranges(
            ranges,
            angle_min=0.0,
            angle_increment=2.0 * np.pi / ray_count,
            mount_yaw=0.0,
        )
        self.assertAlmostEqual(float(result.ranges[0]), 0.4, places=5)
        self.assertAlmostEqual(result.minimum_valid_range, 0.4, places=5)


class ScanSanitizationTests(unittest.TestCase):
    @staticmethod
    def convert(ranges, **kwargs):
        return sim_scan_from_ranges(
            ranges,
            angle_min=0.0,
            angle_increment=2.0 * np.pi / 360,
            **kwargs,
        )

    def test_a_zero_sentinel_scan_is_not_a_wall_at_zero_metres(self):
        """Drivers commonly publish 0.0 for "no return".

        Taken literally it normalises to zero, indistinguishable from a wall
        touching the sensor, which would latch the emergency stop and leave the
        robot immobile with no obvious cause.
        """

        result = self.convert(np.zeros(360, dtype=np.float32), range_max=12.0)
        self.assertTrue(np.all(result.ranges == np.float32(POLICY_MAX_RANGE)))
        self.assertAlmostEqual(result.minimum_valid_range, POLICY_MAX_RANGE)
        self.assertFalse(result.is_healthy)

    def test_positive_infinity_is_a_valid_clear_reading(self):
        result = self.convert(np.full(360, np.inf, dtype=np.float32), range_max=12.0)
        self.assertTrue(np.all(result.ranges == np.float32(POLICY_MAX_RANGE)))
        self.assertEqual(result.out_of_range_fraction, 1.0)
        self.assertEqual(result.invalid_fraction, 0.0)
        self.assertTrue(result.is_healthy)

    def test_out_of_range_is_distinguished_from_invalid(self):
        """An open room is normal; a sensor returning nothing is not.

        Counting out-of-range returns as faults would halt the robot in exactly
        the wide spaces it is meant to cross, because a room larger than the
        LiDAR's range is ordinary.  Note the two cases are numerically identical
        -- both become ``max_range`` -- so the decision rests entirely on which
        counter they land in.
        """

        open_room = self.convert(np.full(360, np.inf, dtype=np.float32), range_max=12.0)
        broken = self.convert(np.zeros(360, dtype=np.float32), range_max=12.0)
        np.testing.assert_array_equal(open_room.ranges, broken.ranges)
        self.assertTrue(open_room.is_healthy)
        self.assertFalse(broken.is_healthy)

    def test_returns_at_or_beyond_range_max_count_as_out_of_range(self):
        result = self.convert(np.full(360, 12.0, dtype=np.float32), range_max=12.0)
        self.assertEqual(result.out_of_range_fraction, 1.0)
        self.assertEqual(result.invalid_fraction, 0.0)

    def test_an_infinite_range_max_leaves_finite_readings_alone(self):
        """Gazebo publishes +inf for no-return; a driver may leave max unbounded."""

        ranges = np.full(360, 90.0, dtype=np.float32)
        result = self.convert(ranges, range_max=float("inf"))
        self.assertEqual(result.valid_count, 360)
        self.assertTrue(np.all(result.ranges == np.float32(POLICY_MAX_RANGE)))

    def test_nan_and_negative_infinity_are_invalid(self):
        ranges = np.full(360, 5.0, dtype=np.float32)
        ranges[:180] = np.nan
        ranges[180:270] = -np.inf
        result = self.convert(ranges, range_max=12.0)
        self.assertAlmostEqual(result.invalid_fraction, 0.75)
        self.assertFalse(result.is_healthy)

    def test_a_genuine_short_reading_is_kept(self):
        ranges = np.full(360, 5.0, dtype=np.float32)
        ranges[7] = 0.3
        result = self.convert(ranges, range_max=12.0)
        self.assertAlmostEqual(result.minimum_valid_range, 0.3, places=5)
        self.assertTrue(np.any(np.isclose(result.ranges, 0.3)))
        self.assertTrue(result.is_healthy)

    def test_range_min_is_only_honoured_when_asked_for(self):
        """``min_range`` is not applied by default, matching the training path."""

        ranges = np.full(360, 5.0, dtype=np.float32)
        ranges[0] = 0.05
        kept = self.convert(ranges, range_max=12.0, range_min=0.15)
        self.assertAlmostEqual(kept.minimum_valid_range, 0.05, places=5)
        dropped = self.convert(
            ranges, range_max=12.0, range_min=0.15, treat_range_min_as_invalid=True
        )
        self.assertAlmostEqual(dropped.minimum_valid_range, 5.0, places=5)

    def test_an_empty_scan_is_rejected(self):
        with self.assertRaises(ValueError):
            self.convert(np.asarray([], dtype=np.float32))

    def test_a_single_ray_scan_does_not_crash(self):
        result = self.convert(np.asarray([4.0], dtype=np.float32), range_max=12.0)
        self.assertEqual(result.ranges.shape, (SIM_RAY_COUNT,))


class RayGeometryTests(unittest.TestCase):
    def test_ray_zero_looks_forward_and_the_sweep_closes_a_full_turn(self):
        """The laser is mounted with a pi yaw offset, so bin zero is ahead."""

        bearings = sim_ray_bearings(SIM_RAY_COUNT)
        self.assertEqual(bearings.shape, (SIM_RAY_COUNT,))
        self.assertAlmostEqual(float(bearings[0]), 0.0, places=2)
        self.assertAlmostEqual(
            float(wrap_to_pi(bearings[-1] + SIM_ANGLE_INCREMENT)),
            0.0,
            places=2,
        )

    def test_the_four_cardinal_rays_land_where_the_urdf_says(self):
        bearings = sim_ray_bearings(SIM_RAY_COUNT)
        quarter = SIM_RAY_COUNT // 4
        self.assertAlmostEqual(float(bearings[quarter]), np.pi / 2, places=2)
        self.assertAlmostEqual(float(wrap_to_pi(bearings[2 * quarter])), -np.pi, places=2)
        self.assertAlmostEqual(float(wrap_to_pi(bearings[3 * quarter])), -np.pi / 2, places=2)

    def test_wrap_to_pi_keeps_angles_comparable(self):
        # The range is [-pi, pi), so both +3pi and -3pi land on -pi.
        np.testing.assert_allclose(
            wrap_to_pi(np.asarray([0.0, 3.0 * np.pi, -3.0 * np.pi, 2.0 * np.pi + 0.5])),
            np.asarray([0.0, -np.pi, -np.pi, 0.5]),
            atol=1e-12,
        )


if __name__ == "__main__":
    unittest.main()
