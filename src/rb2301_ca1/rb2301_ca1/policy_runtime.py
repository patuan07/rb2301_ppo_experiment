"""NumPy-only inference for an exported actor, plus LiDAR convention repair.

This module deliberately imports nothing beyond NumPy so that a storage-limited
robot can run the trained policy without installing PyTorch or
Stable-Baselines3.  The actor is a small MLP, so an exported ``.npz`` of its
weights is all that is required; :mod:`rb2301_ca1.export_policy` produces that
file on a development machine.

Two responsibilities live here:

``ActorMLP``
    Reproduce ``PPO.predict(observation, deterministic=True)`` exactly.

``sim_scan_from_ranges``
    Convert a real ``sensor_msgs/LaserScan`` into the angular convention of the
    Gazebo sensor the policy was trained on.  The conversion is by *angle*, not
    by array index, because a real driver is free to choose its own
    ``angle_min``, sweep direction, and ray count.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np


# Geometry of the Gazebo sensor the policy was trained against.  The laser link
# is mounted at rpy "0 0 3.1416" relative to base_link, so a bearing expressed in
# the policy frame is pi greater than the same bearing in the laser-link frame.
SIM_RAY_COUNT = 720
SIM_MIN_ANGLE = -3.14
SIM_MAX_ANGLE = 3.14
SIM_ANGLE_INCREMENT = (SIM_MAX_ANGLE - SIM_MIN_ANGLE) / SIM_RAY_COUNT
LASER_MOUNT_YAW = 3.1416

POLICY_RAY_COUNT = 36
POLICY_MAX_RANGE = 10.0

# Above this fraction of unusable returns the scan is considered unusable and
# the caller should stop rather than drive on a mostly-blind sensor.
SCAN_INVALID_FRACTION_LIMIT = 0.5

DEFAULT_FLOAT32_TOLERANCE = 1e-5
DEFAULT_FLOAT64_TOLERANCE = 1e-9


def sim_ray_bearings(
    ray_count: int = SIM_RAY_COUNT,
    *,
    mount_yaw: float = LASER_MOUNT_YAW,
) -> np.ndarray:
    """Return the policy-frame bearing of every ray in the simulated sweep.

    ``bearings[k]`` is measured counter-clockwise from robot-forward, so bin 0
    looks ahead and bin ``SIM_RAY_COUNT // 2`` looks behind.
    """

    if ray_count <= 0:
        raise ValueError("ray_count must be positive")
    offsets = np.arange(ray_count, dtype=np.float64) * SIM_ANGLE_INCREMENT
    return mount_yaw + SIM_MIN_ANGLE + offsets


def wrap_to_pi(angles: Any) -> np.ndarray:
    """Wrap angles into ``(-pi, pi]`` so differences compare without a branch."""

    values = np.asarray(angles, dtype=np.float64)
    return (values + np.pi) % (2.0 * np.pi) - np.pi


@dataclass(frozen=True)
class ScanConversion:
    """A real scan re-expressed in the simulated sensor's own convention."""

    ranges: np.ndarray
    """``float32`` array of ``SIM_RAY_COUNT`` ranges in metres."""

    invalid_fraction: float
    """Fraction of input returns that measured nothing at all.

    This -- not :attr:`out_of_range_fraction` -- is the sensor-health signal.
    """

    out_of_range_fraction: float
    """Fraction of input returns legitimately reporting nothing within range."""

    minimum_valid_range: float
    """Smallest trustworthy range in the input, or ``POLICY_MAX_RANGE``."""

    valid_count: int
    """Number of input returns that measured something within range."""

    @property
    def is_healthy(self) -> bool:
        """True when enough returns survived to make a decision worth trusting."""

        return self.invalid_fraction <= SCAN_INVALID_FRACTION_LIMIT


def sim_scan_from_ranges(
    ranges: Any,
    *,
    angle_min: float,
    angle_increment: float,
    range_min: float = 0.0,
    range_max: float = float("inf"),
    mount_yaw: float = 0.0,
    max_range: float = POLICY_MAX_RANGE,
    out_size: int = SIM_RAY_COUNT,
    treat_range_min_as_invalid: bool = False,
) -> ScanConversion:
    """Project a real laser sweep onto the simulated sensor's ray bearings.

    The policy was trained on a 720-bin sweep whose beam bearings are fixed by
    the robot's URDF.  A real driver may publish any ``angle_min``, may sweep
    clockwise, and may use a different ray count, so resampling positionally by
    index would silently rotate or mirror the obstacle field while introducing
    no error anywhere.  Every ray is therefore placed by its measured bearing.

    Returns are sanitised first, and two physically different things are kept
    apart.  *Out of range* is a valid measurement meaning "nothing is there":
    Gazebo writes ``+inf``, and many drivers write their own ``range_max``.  That
    is benign and becomes ``max_range``, which is what the policy was trained to
    see.  *Invalid* is a measurement that did not happen -- ``NaN``, ``-inf``, or
    the ``0.0`` that drivers commonly publish for no-return.  Left alone, a
    ``0.0`` normalises to zero and is indistinguishable from a wall at zero
    distance, which would trip the emergency stop on every distant ray and leave
    the robot immobile with no obvious cause, so it too becomes ``max_range``.

    The two are reported separately because only the invalid fraction says
    anything about sensor health.  Counting out-of-range returns as faults would
    halt the robot in exactly the open spaces it is meant to drive through: a
    room wider than the LiDAR's range is normal, a sensor whose returns have all
    collapsed to zero is not.

    Genuine short readings above ``range_min`` are deliberately *kept*: a
    spurious near obstacle stops the robot, whereas a spurious clear reading
    would drive it into one.
    """

    values = np.asarray(ranges, dtype=np.float32).reshape(-1)
    if values.size == 0:
        raise ValueError("LaserScan contains no ranges")
    if out_size <= 0:
        raise ValueError("out_size must be positive")
    if not np.isfinite(angle_increment):
        raise ValueError("angle_increment must be finite")

    sanitised = np.array(values, dtype=np.float32, copy=True)
    finite = np.isfinite(sanitised)
    positive_infinity = np.isposinf(sanitised)

    invalid = (~finite & ~positive_infinity) | (sanitised <= 0.0)
    out_of_range = positive_infinity.copy()
    if np.isfinite(range_max) and range_max > 0.0:
        out_of_range |= finite & (sanitised >= range_max)
    if treat_range_min_as_invalid and range_min > 0.0:
        invalid |= finite & (sanitised <= range_min)

    discarded = invalid | out_of_range
    sanitised[discarded] = np.float32(max_range)
    np.clip(sanitised, 0.0, max_range, out=sanitised)

    valid_mask = ~discarded
    valid_count = int(np.count_nonzero(valid_mask))
    minimum_valid_range = (
        float(np.min(sanitised[valid_mask])) if valid_count else float(max_range)
    )
    total = float(values.size)

    # Bearings of the incoming beams, then of the beams the policy expects.
    measured = mount_yaw + float(angle_min) + np.arange(values.size) * float(
        angle_increment
    )
    targets = sim_ray_bearings(out_size)

    if values.size == 1:
        nearest = np.zeros(out_size, dtype=np.int64)
    else:
        difference = wrap_to_pi(measured[None, :] - targets[:, None])
        nearest = np.argmin(np.abs(difference), axis=1).astype(np.int64)

    return ScanConversion(
        ranges=sanitised[nearest].astype(np.float32, copy=False),
        invalid_fraction=float(np.count_nonzero(invalid)) / total,
        out_of_range_fraction=float(np.count_nonzero(out_of_range)) / total,
        minimum_valid_range=minimum_valid_range,
        valid_count=valid_count,
    )


def _relu(values: np.ndarray) -> np.ndarray:
    return np.maximum(values, 0.0)


_ACTIVATIONS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "tanh": np.tanh,
    "relu": _relu,
}


@dataclass(frozen=True)
class ActorMLP:
    """The exported actor network, evaluated with NumPy only.

    Stable-Baselines3's deterministic action for this checkpoint is the raw mean
    of a diagonal Gaussian, clipped to the action space.  Concretely,
    ``PPO.predict(observation, deterministic=True)`` reaches
    ``ActorCriticPolicy.predict`` -> ``get_distribution(...).get_actions(True)``,
    which for a ``Box`` space with ``use_sde=False`` and no ``use_squash_output``
    returns ``DiagGaussianDistribution.mode()`` -- the network's mean output,
    untouched.  ``log_std`` therefore plays no part in inference, and the clip to
    ``[-1, 1]`` is applied against the action-space bounds by
    ``BaseAlgorithm.predict`` rather than by the policy.  There is no
    observation normalisation to reproduce.
    """

    weights: tuple[np.ndarray, ...]
    biases: tuple[np.ndarray, ...]
    activation: str
    observation_dim: int
    action_dim: int

    @classmethod
    def load(cls, path: str | Path) -> "ActorMLP":
        """Read an exporter-produced ``actor.npz``.

        The number of layers is taken from the file, and the activation is read
        rather than assumed, so a retrained policy with a different
        ``activation_fn`` cannot be deployed as a silently wrong network.
        """

        source = Path(path)
        if not source.is_file():
            raise FileNotFoundError(
                f"Exported actor not found: {source}. Run "
                "'python -m rb2301_ca1.export_policy' on the training machine."
            )

        with np.load(source, allow_pickle=False) as data:
            missing = [key for key in ("activation", "obs_dim", "act_dim") if key not in data]
            if missing:
                raise ValueError(f"{source} is missing metadata keys: {missing}")

            activation = str(data["activation"])
            observation_dim = int(data["obs_dim"])
            action_dim = int(data["act_dim"])

            if activation not in _ACTIVATIONS:
                raise ValueError(
                    f"Unsupported activation {activation!r} in {source}; "
                    f"expected one of {sorted(_ACTIVATIONS)}"
                )

            weights: list[np.ndarray] = []
            biases: list[np.ndarray] = []
            index = 0
            while f"w{index}" in data or f"b{index}" in data:
                if f"w{index}" not in data or f"b{index}" not in data:
                    raise ValueError(f"{source} has an incomplete layer {index}")
                weights.append(np.array(data[f"w{index}"], dtype=np.float32))
                biases.append(np.array(data[f"b{index}"], dtype=np.float32))
                index += 1

        if not weights:
            raise ValueError(f"{source} contains no layers")

        if weights[0].shape[1] != observation_dim:
            raise ValueError(
                f"{source} declares obs_dim={observation_dim} but layer 0 expects "
                f"{weights[0].shape[1]}"
            )
        if weights[-1].shape[0] != action_dim:
            raise ValueError(
                f"{source} declares act_dim={action_dim} but the final layer "
                f"produces {weights[-1].shape[0]}"
            )
        for position, (weight, bias) in enumerate(zip(weights, biases)):
            if weight.shape[0] != bias.shape[0]:
                raise ValueError(f"layer {position} weight/bias disagree in {source}")
            if position and weight.shape[1] != weights[position - 1].shape[0]:
                raise ValueError(f"layer {position} does not match layer {position - 1}")

        return cls(
            weights=tuple(weights),
            biases=tuple(biases),
            activation=activation,
            observation_dim=observation_dim,
            action_dim=action_dim,
        )

    def forward(
        self,
        observation: Any,
        *,
        dtype: Any = np.float32,
    ) -> np.ndarray:
        """Map one observation to a bounded action in ``[-1, 1]``.

        ``dtype`` exists so the self-test can evaluate the identical
        architecture in float64, where agreement with the training machine is an
        exact statement about the network rather than a statement about GEMM
        rounding.
        """

        activation = _ACTIVATIONS[self.activation]
        values = np.asarray(observation, dtype=dtype).reshape(-1)
        if values.shape != (self.observation_dim,):
            raise ValueError(
                f"observation must contain {self.observation_dim} values, "
                f"received {values.shape[0]}"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError("observation must contain only finite values")

        hidden = values
        for weight, bias in zip(self.weights[:-1], self.biases[:-1]):
            hidden = activation(hidden @ weight.astype(dtype).T + bias.astype(dtype))
        output = hidden @ self.weights[-1].astype(dtype).T + self.biases[-1].astype(dtype)
        return np.clip(output, -1.0, 1.0).astype(dtype, copy=False)


@dataclass(frozen=True)
class SelfTestResult:
    """Outcome of comparing the NumPy actor against the exported reference."""

    rows: int
    float32_error: float
    float32_tolerance: float
    float64_error: float
    float64_tolerance: float
    worst_float32_row: int

    @property
    def passed(self) -> bool:
        return (
            self.float32_error <= self.float32_tolerance
            and self.float64_error <= self.float64_tolerance
        )


def self_test(actor: ActorMLP, reference_path: str | Path) -> SelfTestResult:
    """Verify the exported actor against the reference captured at export time.

    Two comparisons, because one is not enough.  The float64 pass proves the
    *architecture* is right -- weight identity, transpose convention, activation,
    and the absence of any squash or normalisation -- since a real disagreement
    there would show up as an order-one difference.  The float32 pass then proves
    the exact code path the robot will run, against a tolerance that was measured
    on the training machine rather than guessed.
    """

    source = Path(reference_path)
    if not source.is_file():
        raise FileNotFoundError(f"Reference file not found: {source}")

    with np.load(source, allow_pickle=False) as data:
        required = {"observations", "actions_float32"}
        missing = sorted(required - set(data))
        if missing:
            raise ValueError(f"{source} is missing keys: {missing}")
        observations = np.array(data["observations"], dtype=np.float32)
        expected_float32 = np.array(data["actions_float32"], dtype=np.float32)
        expected_float64 = (
            np.array(data["actions_float64"], dtype=np.float64)
            if "actions_float64" in data
            else None
        )
        float32_tolerance = float(
            data["float32_tolerance"] if "float32_tolerance" in data else DEFAULT_FLOAT32_TOLERANCE
        )
        float64_tolerance = float(
            data["float64_tolerance"] if "float64_tolerance" in data else DEFAULT_FLOAT64_TOLERANCE
        )

    if observations.ndim != 2 or observations.shape[0] == 0:
        raise ValueError(f"{source} observations must be a non-empty 2-D array")
    if expected_float32.shape != (observations.shape[0], actor.action_dim):
        raise ValueError(
            f"{source} actions must have shape "
            f"({observations.shape[0]}, {actor.action_dim})"
        )

    actual_float32 = np.stack(
        [actor.forward(row) for row in observations]
    ).astype(np.float32)
    deviation32 = np.abs(actual_float32 - expected_float32)
    float32_error = float(np.max(deviation32))
    worst_row = int(np.argmax(np.max(deviation32, axis=1)))

    float64_error = 0.0
    if expected_float64 is not None:
        actual_float64 = np.stack(
            [actor.forward(row, dtype=np.float64) for row in observations]
        ).astype(np.float64)
        float64_error = float(np.max(np.abs(actual_float64 - expected_float64)))

    return SelfTestResult(
        rows=int(observations.shape[0]),
        float32_error=float32_error,
        float32_tolerance=float32_tolerance,
        float64_error=float64_error,
        float64_tolerance=float64_tolerance,
        worst_float32_row=worst_row,
    )
