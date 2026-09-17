"""Portable demonstration dataset helpers with no ROS dependency."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class DemonstrationBatch:
    observations: np.ndarray
    expert_actions: np.ndarray
    executed_actions: np.ndarray
    episode_ids: np.ndarray
    layout_seeds: np.ndarray
    interventions: np.ndarray

    def validate(self) -> None:
        sample_count = len(self.observations)
        if self.observations.ndim != 2:
            raise ValueError("observations must be a two-dimensional array")
        if self.expert_actions.shape != (sample_count, 3):
            raise ValueError("expert_actions must have shape (N, 3)")
        if self.executed_actions.shape != (sample_count, 3):
            raise ValueError("executed_actions must have shape (N, 3)")
        for name, values in (
            ("episode_ids", self.episode_ids),
            ("layout_seeds", self.layout_seeds),
            ("interventions", self.interventions),
        ):
            if values.shape != (sample_count,):
                raise ValueError(f"{name} must have shape (N,)")
        if not np.all(np.isfinite(self.observations)):
            raise ValueError("observations contain non-finite values")
        if not np.all(np.isfinite(self.expert_actions)):
            raise ValueError("expert_actions contain non-finite values")
        if not np.all(np.isfinite(self.executed_actions)):
            raise ValueError("executed_actions contain non-finite values")
        if np.any(np.abs(self.expert_actions) > 1.00001):
            raise ValueError("expert actions must lie in [-1, 1]")
        if np.any(np.abs(self.executed_actions) > 1.00001):
            raise ValueError("executed actions must lie in [-1, 1]")


def allocate_episode_counts(total: int, worker_count: int) -> tuple[int, ...]:
    """Split a requested total deterministically across non-empty workers."""

    if total < 1:
        raise ValueError("total must be positive")
    if worker_count < 1:
        raise ValueError("worker_count must be positive")
    if worker_count > total:
        raise ValueError("worker_count cannot exceed total episodes")
    quotient, remainder = divmod(total, worker_count)
    return tuple(
        quotient + int(rank < remainder)
        for rank in range(worker_count)
    )


def save_demonstrations(path: str | Path, batch: DemonstrationBatch) -> None:
    batch.validate()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        observations=np.asarray(batch.observations, dtype=np.float32),
        expert_actions=np.asarray(batch.expert_actions, dtype=np.float32),
        executed_actions=np.asarray(batch.executed_actions, dtype=np.float32),
        episode_ids=np.asarray(batch.episode_ids, dtype=np.int64),
        layout_seeds=np.asarray(batch.layout_seeds, dtype=np.int64),
        interventions=np.asarray(batch.interventions, dtype=bool),
    )


def load_demonstrations(path: str | Path) -> DemonstrationBatch:
    with np.load(Path(path), allow_pickle=False) as archive:
        batch = DemonstrationBatch(
            observations=np.asarray(archive["observations"], dtype=np.float32),
            expert_actions=np.asarray(archive["expert_actions"], dtype=np.float32),
            executed_actions=np.asarray(archive["executed_actions"], dtype=np.float32),
            episode_ids=np.asarray(archive["episode_ids"], dtype=np.int64),
            layout_seeds=np.asarray(archive["layout_seeds"], dtype=np.int64),
            interventions=np.asarray(archive["interventions"], dtype=bool),
        )
    batch.validate()
    return batch


def concatenate_demonstrations(
    batches: list[DemonstrationBatch],
) -> DemonstrationBatch:
    if not batches:
        raise ValueError("at least one demonstration batch is required")
    observations: list[np.ndarray] = []
    expert_actions: list[np.ndarray] = []
    executed_actions: list[np.ndarray] = []
    episode_ids: list[np.ndarray] = []
    layout_seeds: list[np.ndarray] = []
    interventions: list[np.ndarray] = []
    episode_offset = 0
    observation_width = batches[0].observations.shape[1]
    for batch in batches:
        batch.validate()
        if batch.observations.shape[1] != observation_width:
            raise ValueError("all datasets must use the same observation width")
        unique_episodes, normalized_ids = np.unique(
            batch.episode_ids,
            return_inverse=True,
        )
        observations.append(batch.observations)
        expert_actions.append(batch.expert_actions)
        executed_actions.append(batch.executed_actions)
        episode_ids.append(normalized_ids.astype(np.int64) + episode_offset)
        layout_seeds.append(batch.layout_seeds)
        interventions.append(batch.interventions)
        episode_offset += len(unique_episodes)
    result = DemonstrationBatch(
        observations=np.concatenate(observations),
        expert_actions=np.concatenate(expert_actions),
        executed_actions=np.concatenate(executed_actions),
        episode_ids=np.concatenate(episode_ids),
        layout_seeds=np.concatenate(layout_seeds),
        interventions=np.concatenate(interventions),
    )
    result.validate()
    return result


def episode_split_indices(
    episode_ids: np.ndarray,
    validation_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Split entire episodes so related transitions never cross the boundary."""

    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must lie strictly between zero and one")
    identifiers = np.asarray(episode_ids, dtype=np.int64)
    unique = np.unique(identifiers)
    if unique.size < 2:
        raise ValueError("at least two episodes are needed for train/validation split")
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique)
    validation_count = min(
        max(int(round(len(unique) * validation_fraction)), 1),
        len(unique) - 1,
    )
    validation_episodes = shuffled[:validation_count]
    validation_mask = np.isin(identifiers, validation_episodes)
    return np.flatnonzero(~validation_mask), np.flatnonzero(validation_mask)
