"""Small runtime helpers shared without importing ROS or learning libraries."""

from __future__ import annotations

import secrets


def select_base_ros_domain_id(
    requested: int | None,
    num_envs: int,
) -> int:
    """Choose a contiguous ROS domain block while avoiding common domain 0."""

    if num_envs < 1 or num_envs > 233:
        raise ValueError("--num-envs must be in the range 1..233")
    maximum_base = 233 - num_envs
    if requested is not None:
        if requested < 0 or requested > maximum_base:
            raise ValueError("ROS domain IDs must remain in the range 0..232")
        return requested
    minimum_base = 20 if maximum_base >= 20 else 0
    return minimum_base + secrets.randbelow(maximum_base - minimum_base + 1)
