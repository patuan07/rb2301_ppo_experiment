"""Generate randomized obstacle fields and reject layouts without a safe route.

Unlike the original RL prototype, this generator never carves an obstacle-free
lane. Obstacles are sampled across the complete corridor, then an A* search on
an inflated occupancy grid verifies that a holonomic robot can reach the goal.
Additional difficulty checks reject trivial straight-centre solutions.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, replace

import numpy as np


@dataclass(frozen=True)
class LayoutConfig:
    x_start: float = 0.8
    x_stop: float = 6.8
    x_spacing: float = 0.4
    y_min: float = -2.0
    y_max: float = 2.0
    y_spacing: float = 0.2
    start_x: float = 0.0
    start_y: float = 0.0
    goal_x: float = 7.2
    goal_y: float = 0.0
    corridor_half_width: float = 2.15
    planning_resolution: float = 0.1
    obstacle_clearance: float = 0.30
    boundary_clearance: float = 0.22
    minimum_obstacles: int = 40
    maximum_active_obstacles: int = 48
    obstacle_pool_size: int = 64
    minimum_straight_blockers: int = 3
    straight_path_half_width: float = 0.30
    minimum_path_lateral_excursion: float = 0.50
    minimum_path_stretch: float = 1.025
    maximum_generation_attempts: int = 500


@dataclass(frozen=True)
class MazeLayout:
    active_positions: tuple[tuple[float, float, float], ...]
    all_positions: tuple[tuple[float, float, float], ...]
    path_centres: tuple[tuple[float, float], ...]
    path_length: float
    straight_blockers: int
    generation_attempt: int


DIFFICULTIES = ("easy", "medium", "hard")


def layout_config_for_difficulty(name: str) -> LayoutConfig:
    """Return a curriculum level without introducing a carved lane."""

    normalized = name.lower().strip()
    base = LayoutConfig()
    if normalized == "easy":
        return replace(
            base,
            minimum_obstacles=32,
            maximum_active_obstacles=40,
            minimum_straight_blockers=1,
            minimum_path_lateral_excursion=0.30,
            minimum_path_stretch=1.005,
            obstacle_clearance=0.28,
        )
    if normalized == "medium":
        return base
    if normalized == "hard":
        return replace(
            base,
            minimum_obstacles=48,
            maximum_active_obstacles=58,
            minimum_straight_blockers=4,
            minimum_path_lateral_excursion=0.70,
            minimum_path_stretch=1.05,
            obstacle_clearance=0.32,
            maximum_generation_attempts=1000,
        )
    raise ValueError(f"Unknown maze difficulty {name!r}; choose from {DIFFICULTIES}")


def _validate_config(config: LayoutConfig) -> None:
    if config.planning_resolution <= 0.0:
        raise ValueError("planning_resolution must be positive")
    if config.obstacle_clearance <= 0.0:
        raise ValueError("obstacle_clearance must be positive")
    if config.boundary_clearance < 0.0:
        raise ValueError("boundary_clearance cannot be negative")
    if not 0 < config.minimum_obstacles <= config.maximum_active_obstacles:
        raise ValueError("active obstacle bounds are invalid")
    if config.maximum_active_obstacles > config.obstacle_pool_size:
        raise ValueError("maximum_active_obstacles exceeds obstacle_pool_size")
    if config.maximum_generation_attempts < 1:
        raise ValueError("maximum_generation_attempts must be positive")


def _axis_values(start: float, stop: float, spacing: float) -> np.ndarray:
    return np.arange(start, stop + spacing / 2.0, spacing, dtype=np.float64)


def _planning_axes(config: LayoutConfig) -> tuple[np.ndarray, np.ndarray]:
    x_steps = int(
        math.floor(
            (config.goal_x - config.start_x) / config.planning_resolution
            + 1e-9
        )
    )
    x_values = (
        config.start_x
        + np.arange(x_steps + 1, dtype=np.float64)
        * config.planning_resolution
    )
    permitted_half_width = config.corridor_half_width - config.boundary_clearance
    y_steps = int(
        math.floor(permitted_half_width / config.planning_resolution + 1e-9)
    )
    # Construct from zero so the centre line and both lateral limits are exact
    # mirror images even when the permitted width is not resolution-aligned.
    y_values = (
        np.arange(-y_steps, y_steps + 1, dtype=np.float64)
        * config.planning_resolution
    )
    return x_values, y_values


def _nearest_index(values: np.ndarray, coordinate: float) -> int:
    return int(np.argmin(np.abs(values - coordinate)))


def _occupancy_grid(
    positions: tuple[tuple[float, float, float], ...],
    config: LayoutConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_values, y_values = _planning_axes(config)
    grid_x, grid_y = np.meshgrid(x_values, y_values, indexing="ij")
    blocked = np.zeros(grid_x.shape, dtype=bool)
    clearance_squared = config.obstacle_clearance**2
    for obstacle_x, obstacle_y, _ in positions:
        blocked |= (
            np.square(grid_x - obstacle_x) + np.square(grid_y - obstacle_y)
            <= clearance_squared
        )
    return blocked, x_values, y_values


def find_clear_path(
    positions: tuple[tuple[float, float, float], ...],
    config: LayoutConfig = LayoutConfig(),
) -> tuple[tuple[tuple[float, float], ...], float]:
    """Return an eight-connected A* path and length, or ``((), inf)``."""

    _validate_config(config)
    blocked, x_values, y_values = _occupancy_grid(positions, config)
    start = (
        _nearest_index(x_values, config.start_x),
        _nearest_index(y_values, config.start_y),
    )
    goal = (
        _nearest_index(x_values, config.goal_x),
        _nearest_index(y_values, config.goal_y),
    )
    if blocked[start] or blocked[goal]:
        return (), math.inf

    neighbours = (
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    )
    frontier: list[tuple[float, float, tuple[int, int]]] = []
    heapq.heappush(frontier, (0.0, 0.0, start))
    previous: dict[tuple[int, int], tuple[int, int]] = {}
    cost = {start: 0.0}

    while frontier:
        _, current_cost, current = heapq.heappop(frontier)
        if current_cost > cost.get(current, math.inf):
            continue
        if current == goal:
            break
        for delta_x, delta_y in neighbours:
            candidate = (current[0] + delta_x, current[1] + delta_y)
            if not (
                0 <= candidate[0] < blocked.shape[0]
                and 0 <= candidate[1] < blocked.shape[1]
            ):
                continue
            if blocked[candidate]:
                continue
            if delta_x and delta_y:
                # A diagonal is only safe when the two adjacent cardinal cells
                # are also free. Otherwise A* could squeeze between the corners
                # of two inflated obstacles even though the robot cannot.
                adjacent_x = (current[0] + delta_x, current[1])
                adjacent_y = (current[0], current[1] + delta_y)
                if blocked[adjacent_x] or blocked[adjacent_y]:
                    continue
            step_cost = config.planning_resolution * (
                math.sqrt(2.0) if delta_x and delta_y else 1.0
            )
            candidate_cost = current_cost + step_cost
            if candidate_cost >= cost.get(candidate, math.inf):
                continue
            cost[candidate] = candidate_cost
            previous[candidate] = current
            heuristic = config.planning_resolution * math.hypot(
                goal[0] - candidate[0],
                goal[1] - candidate[1],
            )
            heapq.heappush(
                frontier,
                (candidate_cost + heuristic, candidate_cost, candidate),
            )

    if goal not in cost:
        return (), math.inf

    cells = [goal]
    while cells[-1] != start:
        cells.append(previous[cells[-1]])
    cells.reverse()
    path = tuple(
        (float(x_values[x_index]), float(y_values[y_index]))
        for x_index, y_index in cells
    )
    return path, float(cost[goal])


def _straight_blocker_count(
    positions: tuple[tuple[float, float, float], ...],
    config: LayoutConfig,
) -> int:
    return sum(
        abs(y - config.start_y) <= config.straight_path_half_width
        for _, y, _ in positions
    )


def _layout_is_acceptable(
    positions: tuple[tuple[float, float, float], ...],
    config: LayoutConfig,
) -> tuple[bool, tuple[tuple[float, float], ...], float, int]:
    blocker_count = _straight_blocker_count(positions, config)
    if blocker_count < config.minimum_straight_blockers:
        return False, (), math.inf, blocker_count
    path, path_length = find_clear_path(positions, config)
    if not path:
        return False, (), math.inf, blocker_count
    lateral_excursion = max(abs(y - config.start_y) for _, y in path)
    direct_distance = math.hypot(
        config.goal_x - config.start_x,
        config.goal_y - config.start_y,
    )
    acceptable = (
        lateral_excursion >= config.minimum_path_lateral_excursion
        and path_length >= direct_distance * config.minimum_path_stretch
    )
    return acceptable, path, path_length, blocker_count


def generate_layout(
    seed: int | None = None,
    config: LayoutConfig = LayoutConfig(),
) -> MazeLayout:
    """Sample the full corridor until a connected, nontrivial layout is found."""

    _validate_config(config)
    rng = np.random.default_rng(seed)
    candidates = tuple(
        (float(x), float(y), 0.0)
        for x in _axis_values(config.x_start, config.x_stop, config.x_spacing)
        for y in _axis_values(config.y_min, config.y_max, config.y_spacing)
    )
    if config.maximum_active_obstacles > len(candidates):
        raise ValueError("obstacle count exceeds the number of candidate positions")

    for attempt in range(1, config.maximum_generation_attempts + 1):
        obstacle_count = int(
            rng.integers(
                config.minimum_obstacles,
                config.maximum_active_obstacles + 1,
            )
        )
        selected_indices = rng.choice(
            len(candidates),
            size=obstacle_count,
            replace=False,
        )
        active = tuple(sorted((candidates[int(i)] for i in selected_indices)))
        acceptable, path, path_length, blockers = _layout_is_acceptable(
            active,
            config,
        )
        if not acceptable:
            continue
        parked = tuple(
            (-3.0, 0.0, -2.0)
            for _ in range(config.obstacle_pool_size - len(active))
        )
        return MazeLayout(
            active_positions=active,
            all_positions=active + parked,
            path_centres=path,
            path_length=path_length,
            straight_blockers=blockers,
            generation_attempt=attempt,
        )

    raise RuntimeError(
        "Unable to generate a connected maze satisfying the requested difficulty "
        f"after {config.maximum_generation_attempts} attempts"
    )
