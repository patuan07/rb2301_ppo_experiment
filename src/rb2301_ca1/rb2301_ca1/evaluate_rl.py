"""Evaluate a trained continuous PPO or SAC holonomic policy.

Managed evaluation can use several isolated Gazebo workers. Every episode is
assigned its original global seed before work is split, so parallel evaluation
produces the same seed set as serial evaluation while reducing wall-clock time.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
from collections import Counter
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3 import PPO, SAC

from .maze_layout import DIFFICULTIES, layout_config_for_difficulty
from .ros_gym_env import RosLidarMazeEnv, RosMazeEnvConfig
from .runtime_utils import select_base_ros_domain_id


@dataclass(frozen=True)
class EvaluationWorkerSpec:
    """Picklable inputs for one isolated managed evaluation worker."""

    rank: int
    episode_indices: tuple[int, ...]
    seed: int
    model: Path
    algorithm: str
    config: RosMazeEnvConfig
    run_dir: Path
    base_ros_domain_id: int
    worker_recycle_episodes: int
    worker_recovery_attempts: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--algorithm", choices=("ppo", "sac"), default="ppo")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=92301)
    parser.add_argument("--maze-difficulty", choices=DIFFICULTIES, default="medium")
    parser.add_argument(
        "--external-sim",
        action="store_true",
        help="Use an already-running Gazebo instance instead of launching workers",
    )
    parser.add_argument(
        "--base-ros-domain-id",
        type=int,
        default=None,
        help="First ROS domain for managed evaluation workers",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Directory for managed evaluation logs and summary JSON",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Optional explicit path for the machine-readable evaluation summary",
    )
    parser.add_argument(
        "--randomize-obstacles",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--worker-recycle-episodes", type=int, default=200)
    parser.add_argument("--worker-recovery-attempts", type=int, default=2)
    return parser.parse_args()


def _episode_chunks(episodes: int, worker_count: int) -> list[tuple[int, ...]]:
    """Assign episode indices round-robin for balanced parallel workers."""

    return [
        tuple(index for index in range(episodes) if index % worker_count == rank)
        for rank in range(worker_count)
    ]


def _episode_result(
    *,
    episode_index: int,
    seed: int,
    episode_return: float,
    info: dict[str, Any],
) -> dict[str, Any]:
    return {
        "episode": episode_index + 1,
        "seed": seed + episode_index,
        "return": float(episode_return),
        "steps": int(info.get("step_count", 0)),
        "reason": str(info.get("reason", "unknown")),
        "minimum_scan": float(info.get("minimum_scan", float("nan"))),
        "x": float(info.get("x", float("nan"))),
        "y": float(info.get("y", float("nan"))),
        "yaw": float(info.get("yaw", float("nan"))),
    }


def _evaluate_worker(spec: EvaluationWorkerSpec) -> list[dict[str, Any]]:
    """Evaluate one fixed subset of seeds in one isolated Gazebo process."""

    from .managed_env import make_managed_env

    environment = None
    try:
        environment = make_managed_env(
            rank=spec.rank,
            base_ros_domain_id=spec.base_ros_domain_id,
            config=spec.config,
            run_dir=spec.run_dir,
            recycle_episodes=spec.worker_recycle_episodes,
            recovery_attempts=spec.worker_recovery_attempts,
        )()
        algorithm_class = PPO if spec.algorithm == "ppo" else SAC
        model = algorithm_class.load(
            str(spec.model),
            env=environment,
            device="cpu",
        )
        results: list[dict[str, Any]] = []
        for episode_index in spec.episode_indices:
            observation, _ = environment.reset(seed=spec.seed + episode_index)
            episode_return = 0.0
            terminated = truncated = False
            info: dict[str, Any] = {"reason": "unknown"}
            while not (terminated or truncated):
                action, _ = model.predict(observation, deterministic=True)
                observation, reward, terminated, truncated, info = environment.step(
                    np.asarray(action, dtype=np.float32)
                )
                episode_return += float(reward)
            results.append(
                _episode_result(
                    episode_index=episode_index,
                    seed=spec.seed,
                    episode_return=episode_return,
                    info=info,
                )
            )
        return results
    finally:
        if environment is not None:
            environment.close()


def _wilson_interval(successes: int, total: int, z: float = 1.96) -> list[float]:
    """Return a numerically stable approximate 95% binomial interval."""

    if total <= 0:
        return [float("nan"), float("nan")]
    proportion = successes / total
    denominator = 1.0 + z * z / total
    centre = (proportion + z * z / (2.0 * total)) / denominator
    half_width = (
        z
        * np.sqrt(
            proportion * (1.0 - proportion) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return [float(max(0.0, centre - half_width)), float(min(1.0, centre + half_width))]


def _summarize(
    results: list[dict[str, Any]],
    *,
    model: Path,
    algorithm: str,
    seed: int,
    maze_difficulty: str,
    num_envs: int,
) -> dict[str, Any]:
    ordered = sorted(results, key=lambda item: int(item["episode"]))
    outcomes = Counter(str(item["reason"]) for item in ordered)
    returns = np.asarray([float(item["return"]) for item in ordered], dtype=np.float64)
    lengths = np.asarray([int(item["steps"]) for item in ordered], dtype=np.float64)
    successes = int(outcomes.get("success", 0))
    total = len(ordered)
    return {
        "model": str(model.resolve()),
        "algorithm": algorithm,
        "seed": seed,
        "maze_difficulty": maze_difficulty,
        "episodes": total,
        "num_envs": num_envs,
        "successes": successes,
        "success_rate": float(successes / total) if total else float("nan"),
        "success_rate_wilson_95": _wilson_interval(successes, total),
        "outcomes": {key: int(value) for key, value in sorted(outcomes.items())},
        "mean_return": float(np.mean(returns)) if total else float("nan"),
        "std_return": float(np.std(returns)) if total else float("nan"),
        "mean_episode_length": float(np.mean(lengths)) if total else float("nan"),
        "episodes_detail": ordered,
    }


def _print_summary(summary: dict[str, Any]) -> None:
    outcomes = summary["outcomes"]
    for item in summary["episodes_detail"]:
        print(
            f"episode={int(item['episode']):02d} return={float(item['return']):8.3f} "
            f"steps={int(item['steps']):3d} reason={item['reason']} "
            f"seed={int(item['seed'])} "
            f"final=({float(item['x']):.2f}, {float(item['y']):.2f})"
        )
    print(
        f"success_rate={summary['success_rate']:.1%} "
        f"mean_return={summary['mean_return']:.3f} "
        f"std_return={summary['std_return']:.3f} "
        f"mean_length={summary['mean_episode_length']:.1f}"
    )
    print(
        "outcomes="
        + " ".join(
            f"{reason}:{outcomes.get(reason, 0)}"
            for reason in (
                "success",
                "collision",
                "out_of_bounds",
                "time_limit",
                "worker_restart",
            )
        )
    )
    low, high = summary["success_rate_wilson_95"]
    print(f"success_rate_wilson_95={low:.1%}..{high:.1%}")


def _evaluate_external(args: argparse.Namespace, config: RosMazeEnvConfig) -> list[dict[str, Any]]:
    environment = None
    try:
        environment = RosLidarMazeEnv(
            config=config,
            node_name="rb2301_rl_evaluation",
        )
        algorithm_class = PPO if args.algorithm == "ppo" else SAC
        model = algorithm_class.load(str(args.model), env=environment, device="cpu")
        results: list[dict[str, Any]] = []
        for episode_index in range(args.episodes):
            observation, _ = environment.reset(seed=args.seed + episode_index)
            episode_return = 0.0
            terminated = truncated = False
            info: dict[str, Any] = {"reason": "unknown"}
            while not (terminated or truncated):
                action, _ = model.predict(observation, deterministic=True)
                observation, reward, terminated, truncated, info = environment.step(
                    np.asarray(action, dtype=np.float32)
                )
                episode_return += float(reward)
            results.append(
                _episode_result(
                    episode_index=episode_index,
                    seed=args.seed,
                    episode_return=episode_return,
                    info=info,
                )
            )
        return results
    finally:
        if environment is not None:
            environment.close()


def main() -> None:
    args = parse_args()
    if args.episodes < 1:
        raise ValueError("--episodes must be at least one")
    if args.num_envs < 1:
        raise ValueError("--num-envs must be at least one")
    if args.num_envs > args.episodes:
        raise ValueError("--num-envs cannot exceed --episodes")
    if args.worker_recycle_episodes < 0 or args.worker_recovery_attempts < 0:
        raise ValueError("worker lifecycle values cannot be negative")
    if not args.model.is_file():
        raise FileNotFoundError(f"Model does not exist: {args.model}")
    if args.external_sim and args.num_envs != 1:
        raise ValueError("--external-sim requires --num-envs 1")

    config = replace(
        RosMazeEnvConfig(),
        randomize_obstacles=args.randomize_obstacles,
        layout=layout_config_for_difficulty(args.maze_difficulty),
    )
    managed_run_dir: Path | None = None
    results: list[dict[str, Any]] = []
    try:
        if args.external_sim:
            results = _evaluate_external(args, config)
        else:
            from concurrent.futures import ProcessPoolExecutor, as_completed

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            managed_run_dir = args.run_dir or (
                args.model.parent / "evaluations" / timestamp
            )
            managed_run_dir.mkdir(parents=True, exist_ok=True)
            base_ros_domain_id = select_base_ros_domain_id(
                args.base_ros_domain_id,
                args.num_envs,
            )
            specifications = [
                EvaluationWorkerSpec(
                    rank=rank,
                    episode_indices=chunk,
                    seed=args.seed,
                    model=args.model.resolve(),
                    algorithm=args.algorithm,
                    config=config,
                    run_dir=managed_run_dir,
                    base_ros_domain_id=base_ros_domain_id,
                    worker_recycle_episodes=args.worker_recycle_episodes,
                    worker_recovery_attempts=args.worker_recovery_attempts,
                )
                for rank, chunk in enumerate(_episode_chunks(args.episodes, args.num_envs))
            ]
            print(
                f"Managed evaluation logs: {managed_run_dir} "
                f"({args.num_envs} parallel worker(s))"
            )
            context = multiprocessing.get_context("spawn")
            with ProcessPoolExecutor(
                max_workers=args.num_envs,
                mp_context=context,
            ) as executor:
                futures = [executor.submit(_evaluate_worker, spec) for spec in specifications]
                for future in as_completed(futures):
                    results.extend(future.result())
    finally:
        if managed_run_dir is not None:
            from .managed_env import cleanup_managed_workers

            cleanup_managed_workers(managed_run_dir)

    summary = _summarize(
        results,
        model=args.model,
        algorithm=args.algorithm,
        seed=args.seed,
        maze_difficulty=args.maze_difficulty,
        num_envs=args.num_envs,
    )
    _print_summary(summary)
    summary_path = args.summary_json
    if summary_path is None and managed_run_dir is not None:
        summary_path = managed_run_dir / "evaluation_summary.json"
    if summary_path is not None:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(summary, indent=2, allow_nan=False),
            encoding="utf-8",
        )
        print(f"Saved evaluation summary to {summary_path}")


if __name__ == "__main__":
    main()
