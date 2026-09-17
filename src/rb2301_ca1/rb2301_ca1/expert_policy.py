"""ROS-independent 90-degree-cone expert used for demonstrations and DAgger."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ConeExpertConfig:
    ray_count: int = 36
    lidar_max_range: float = 10.0
    cone_fov_degrees: float = 90.0
    minimum_clearance: float = 0.30
    slowdown_distance: float = 0.10
    minimum_speed_ratio: float = 0.25


def cone_clearance(
    scan_metres: np.ndarray,
    centre_angle_degrees: float,
    fov_degrees: float,
) -> float:
    """Return minimum clearance in a circular cone over one complete scan."""

    scan = np.asarray(scan_metres, dtype=np.float32).reshape(-1)
    if scan.size == 0:
        return 0.0
    angles = np.arange(scan.size, dtype=np.float32) * (360.0 / scan.size)
    difference = (angles - centre_angle_degrees + 180.0) % 360.0 - 180.0
    values = scan[np.abs(difference) <= fov_degrees / 2.0]
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0
    return float(np.min(values))


class ConeExpertPolicy:
    """Stateful cardinal expert matching the user's assignment solver."""

    def __init__(self, config: ConeExpertConfig = ConeExpertConfig()) -> None:
        self.config = config
        self.last_side = 0
        self.heading_degrees: int | None = None

    def reset(self) -> None:
        self.last_side = 0
        self.heading_degrees = None

    def predict(self, observation: np.ndarray) -> np.ndarray:
        values = np.asarray(observation, dtype=np.float32).reshape(-1)
        if values.size < self.config.ray_count:
            raise ValueError("observation does not contain the configured LiDAR rays")
        scan_metres = np.clip(
            values[: self.config.ray_count],
            0.0,
            1.0,
        ) * self.config.lidar_max_range

        movement_angles = [0, 90, -90, 180]
        if self.last_side < 0:
            movement_angles[1:3] = [-90, 90]

        for angle_degrees in movement_angles:
            clearance = cone_clearance(
                scan_metres,
                angle_degrees,
                self.config.cone_fov_degrees,
            )
            if clearance <= self.config.minimum_clearance:
                continue
            ratio = float(
                np.clip(
                    (clearance - self.config.minimum_clearance)
                    / self.config.slowdown_distance,
                    0.0,
                    1.0,
                )
            )
            ratio = max(ratio, self.config.minimum_speed_ratio)
            radians = math.radians(angle_degrees)
            action = np.asarray(
                [ratio * math.cos(radians), ratio * math.sin(radians), 0.0],
                dtype=np.float32,
            )
            action[np.abs(action) < 1e-7] = 0.0
            self.heading_degrees = angle_degrees
            if angle_degrees == 90:
                self.last_side = 1
            elif angle_degrees == -90:
                self.last_side = -1
            return action

        self.heading_degrees = None
        return np.zeros(3, dtype=np.float32)
