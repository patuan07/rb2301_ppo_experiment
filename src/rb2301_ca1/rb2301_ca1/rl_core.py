"""ROS-independent utilities for continuous holonomic maze control."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class RewardConfig:
    """Weights used by :func:`evaluate_transition`."""

    progress_scale: float = 5.0
    # At 20 control decisions/s, 0.0025 preserves the old 0.05 reward/s cost.
    step_penalty: float = 0.0025
    # A late collision must remain worse than making maximum forward progress.
    collision_penalty: float = 40.0
    success_bonus: float = 20.0
    out_of_bounds_penalty: float = 40.0
    smoothness_scale: float = 0.10
    turn_scale: float = 0.002
    proximity_scale: float = 0.05
    safety_distance: float = 0.45


@dataclass(frozen=True)
class Transition:
    """Reward and episode flags produced after one environment action."""

    reward: float
    terminated: bool
    truncated: bool
    reason: str
    terms: dict[str, float]


def sample_scan_metres(
    ranges: Any,
    ray_count: int = 36,
    min_range: float = 0.05,
    max_range: float = 10.0,
) -> np.ndarray:
    """Sanitize and uniformly sample a LaserScan as metres.

    ``+inf`` means that no obstacle was detected and becomes ``max_range``.
    NaN and negative infinity are treated as unsafe zero-range readings.  The
    integer indexing deliberately avoids the duplicated final ray sometimes
    produced by a 0-to-2pi scan.
    """

    values = np.asarray(ranges, dtype=np.float32).reshape(-1)
    if values.size == 0:
        raise ValueError("LaserScan contains no ranges")
    if ray_count <= 0:
        raise ValueError("ray_count must be positive")
    if max_range <= min_range:
        raise ValueError("max_range must be greater than min_range")

    indices = (np.arange(ray_count, dtype=np.int64) * values.size) // ray_count
    sampled = values[indices]
    sampled = np.nan_to_num(
        sampled,
        nan=0.0,
        posinf=max_range,
        neginf=0.0,
    )
    return np.clip(sampled, 0.0, max_range).astype(np.float32, copy=False)


def preprocess_scan(
    ranges: Any,
    ray_count: int = 36,
    min_range: float = 0.05,
    max_range: float = 10.0,
) -> np.ndarray:
    """Return a normalized float32 LiDAR observation in the range [0, 1]."""

    metres = sample_scan_metres(ranges, ray_count, min_range, max_range)
    return (metres / np.float32(max_range)).astype(np.float32, copy=False)


def smooth_normalized_action(
    action: Any,
    previous_action: Any,
    smoothing_alpha: float = 0.35,
) -> np.ndarray:
    """Apply a brake-fast, accelerate-smoothly filter in ``[-1, 1]``.

    Increasing magnitude is low-pass filtered. Decreasing magnitude is applied
    immediately, and a sign reversal first brakes that axis to zero. This keeps
    learned motion smooth without delaying an obstacle-avoidance stop.
    """

    if not 0.0 < smoothing_alpha <= 1.0:
        raise ValueError("smoothing_alpha must be in (0, 1]")
    requested = np.asarray(action, dtype=np.float32).reshape(-1)
    previous = np.asarray(previous_action, dtype=np.float32).reshape(-1)
    if requested.shape != (3,) or previous.shape != (3,):
        raise ValueError("action and previous_action must each contain [x, y, yaw]")
    if not np.all(np.isfinite(requested)):
        raise ValueError("action must contain only finite values")
    requested = np.clip(requested, -1.0, 1.0)
    previous = np.clip(previous, -1.0, 1.0)
    filtered = np.empty(3, dtype=np.float32)
    for index, (target, current) in enumerate(zip(requested, previous)):
        if target * current < 0.0:
            # Never cross through zero in one control tick.
            filtered[index] = 0.0
        elif abs(target) <= abs(current):
            # Braking and releasing an axis must not be delayed.
            filtered[index] = target
        else:
            filtered[index] = current + smoothing_alpha * (target - current)
    return filtered


def action_to_velocity(
    action: Any,
    max_x_velocity: float = 0.4,
    max_y_velocity: float = 0.4,
    max_turn_velocity: float = 0.8,
) -> tuple[float, float, float]:
    """Scale a continuous normalized holonomic action to a Twist triple."""

    values = np.asarray(action, dtype=np.float32).reshape(-1)
    if values.shape != (3,):
        raise ValueError("action must contain [x, y, yaw]")
    if not np.all(np.isfinite(values)):
        raise ValueError("action must contain only finite values")
    values = np.clip(values, -1.0, 1.0)
    scales = np.asarray(
        [max_x_velocity, max_y_velocity, max_turn_velocity], dtype=np.float32
    )
    if np.any(scales <= 0.0):
        raise ValueError("velocity limits must be positive")
    velocity = values * scales
    return tuple(float(value) for value in velocity)


def observation_bounds(ray_count: int = 36) -> tuple[np.ndarray, np.ndarray]:
    """Return the canonical bounded space used by ROS and imitation models."""

    if ray_count <= 0:
        raise ValueError("ray_count must be positive")
    low = np.concatenate(
        [
            np.zeros(ray_count, dtype=np.float32),
            np.asarray([-1.0, -1.0, 0.0], dtype=np.float32),
            -np.ones(3, dtype=np.float32),
        ]
    )
    high = np.ones(ray_count + 6, dtype=np.float32)
    return low, high


def build_observation(
    ranges: Any,
    *,
    x: float,
    y: float,
    yaw: float,
    goal_x: float,
    goal_y: float,
    previous_action: Any,
    ray_count: int = 36,
    min_range: float = 0.05,
    max_range: float = 10.0,
    distance_scale: float = 8.0,
) -> np.ndarray:
    """Build a Markov observation for planar goal-directed navigation.

    The scan is followed by the goal direction in the robot frame, normalized
    goal distance, and the last applied normalized action. Supplying goal
    direction is important when yaw is controllable: a single body-frame scan
    alone does not identify the world-frame goal direction.
    """

    scan = preprocess_scan(ranges, ray_count, min_range, max_range)
    dx = float(goal_x - x)
    dy = float(goal_y - y)
    distance = float(np.hypot(dx, dy))
    if distance > 1e-6:
        world_direction = np.asarray([dx / distance, dy / distance], dtype=np.float32)
    else:
        world_direction = np.zeros(2, dtype=np.float32)
    cosine = np.float32(np.cos(yaw))
    sine = np.float32(np.sin(yaw))
    goal_direction_body = np.asarray(
        [
            cosine * world_direction[0] + sine * world_direction[1],
            -sine * world_direction[0] + cosine * world_direction[1],
        ],
        dtype=np.float32,
    )
    previous = np.asarray(previous_action, dtype=np.float32).reshape(-1)
    if previous.shape != (3,):
        raise ValueError("previous_action must contain [x, y, yaw]")
    normalized_distance = np.float32(np.clip(distance / distance_scale, 0.0, 1.0))
    return np.concatenate(
        [scan, goal_direction_body, [normalized_distance], np.clip(previous, -1.0, 1.0)]
    ).astype(np.float32, copy=False)


def evaluate_transition(
    *,
    previous_x: float,
    current_x: float,
    current_y: float,
    minimum_scan: float,
    step_count: int,
    goal_x: float,
    corridor_half_width: float,
    collision_distance: float,
    max_episode_steps: int,
    action_delta_squared: float = 0.0,
    turn_velocity: float = 0.0,
    reward_config: RewardConfig = RewardConfig(),
) -> Transition:
    """Calculate reward and terminal state without depending on ROS."""

    terms = {
        "progress": reward_config.progress_scale * (current_x - previous_x),
        "step": -reward_config.step_penalty,
        "collision": 0.0,
        "success": 0.0,
        "out_of_bounds": 0.0,
        "smoothness": -reward_config.smoothness_scale * action_delta_squared,
        "turn": -reward_config.turn_scale * abs(turn_velocity),
        "proximity": 0.0,
    }

    proximity = max(0.0, reward_config.safety_distance - minimum_scan)
    if reward_config.safety_distance > 0.0:
        terms["proximity"] = -reward_config.proximity_scale * (
            proximity / reward_config.safety_distance
        )

    collision = minimum_scan <= collision_distance
    success = current_x >= goal_x and not collision
    out_of_bounds = abs(current_y) >= corridor_half_width
    timed_out = step_count >= max_episode_steps

    reason = "running"
    terminated = False
    truncated = False

    if collision:
        # Never let a large forward jump make a collision profitable.
        terms["progress"] = min(terms["progress"], 0.0)
        terms["collision"] = -reward_config.collision_penalty
        terminated = True
        reason = "collision"
    elif success:
        terms["success"] = reward_config.success_bonus
        terminated = True
        reason = "success"
    elif out_of_bounds:
        # Likewise, crossing the boundary must not be a shortcut to reward.
        terms["progress"] = min(terms["progress"], 0.0)
        terms["out_of_bounds"] = -reward_config.out_of_bounds_penalty
        truncated = True
        reason = "out_of_bounds"
    elif timed_out:
        truncated = True
        reason = "time_limit"

    return Transition(
        reward=float(sum(terms.values())),
        terminated=terminated,
        truncated=truncated,
        reason=reason,
        terms=terms,
    )
