"""Collect expert or DAgger-labelled trajectories in parallel Gazebo workers."""

from __future__ import annotations

import argparse
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from .expert_policy import ConeExpertPolicy
from .imitation_core import (
    DemonstrationBatch,
    allocate_episode_counts,
    concatenate_demonstrations,
    load_demonstrations,
    save_demonstrations,
)
from .maze_layout import DIFFICULTIES, layout_config_for_difficulty
from .ros_gym_env import RosLidarMazeEnv, RosMazeEnvConfig
from .runtime_utils import select_base_ros_domain_id


@dataclass(frozen=True)
class CollectionWorkerSpec:
    rank: int
    target_episodes: int
    maximum_attempts: int
    seed: int
    config: RosMazeEnvConfig
    run_dir: Path
    shard_path: Path
    base_ros_domain_id: int
    external_sim: bool
    student_model: Path | None
    student_algorithm: str
    student_control_probability: float
    safety_override_distance: float
    successful_only: bool
    recycle_episodes: int
    recovery_attempts: int
    shard_checkpoint_episodes: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument(
        "--num-envs",
        type=int,
        default=1,
        help="Independent headless collectors; episode total is split across them",
    )
    parser.add_argument("--seed", type=int, default=12301)
    parser.add_argument("--maze-difficulty", choices=DIFFICULTIES, default="medium")
    parser.add_argument("--student-model", type=Path, default=None)
    parser.add_argument("--student-algorithm", choices=("ppo", "sac"), default="ppo")
    parser.add_argument(
        "--student-control-probability",
        type=float,
        default=0.5,
        help="DAgger probability of executing the student while expert labels all states",
    )
    parser.add_argument(
        "--safety-override-distance",
        type=float,
        default=0.24,
        help="Execute the expert directly when a sampled obstacle is this close",
    )
    parser.add_argument(
        "--successful-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Retain only trajectories ending in success",
    )
    parser.add_argument(
        "--max-attempted-episodes",
        type=int,
        default=None,
        help="Global collection cap; default is ten times --episodes",
    )
    parser.add_argument(
        "--append",
        type=Path,
        action="append",
        default=[],
        help="Existing dataset to aggregate before writing --output; repeatable",
    )
    parser.add_argument("--external-sim", action="store_true")
    parser.add_argument("--base-ros-domain-id", type=int, default=None)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--worker-recycle-episodes", type=int, default=200)
    parser.add_argument("--worker-recovery-attempts", type=int, default=2)
    parser.add_argument(
        "--shard-checkpoint-episodes",
        type=int,
        default=25,
        help="Rewrite each worker shard after this many retained episodes",
    )
    return parser.parse_args()


def _batch_from_lists(
    observations: list[np.ndarray],
    expert_actions: list[np.ndarray],
    executed_actions: list[np.ndarray],
    episode_ids: list[int],
    layout_seeds: list[int],
    interventions: list[bool],
) -> DemonstrationBatch:
    return DemonstrationBatch(
        observations=np.asarray(observations, dtype=np.float32),
        expert_actions=np.asarray(expert_actions, dtype=np.float32),
        executed_actions=np.asarray(executed_actions, dtype=np.float32),
        episode_ids=np.asarray(episode_ids, dtype=np.int64),
        layout_seeds=np.asarray(layout_seeds, dtype=np.int64),
        interventions=np.asarray(interventions, dtype=bool),
    )


def _as_json_action(value: Any) -> list[float] | None:
    if value is None:
        return None
    return [float(item) for item in np.asarray(value).reshape(-1)]


def _collect_worker(spec: CollectionWorkerSpec) -> dict[str, Any]:
    """Run one isolated collector and persist a recoverable dataset shard."""

    rng = np.random.default_rng(spec.seed)
    student = None
    if spec.student_model is not None:
        from stable_baselines3 import PPO, SAC

        algorithm_class = PPO if spec.student_algorithm == "ppo" else SAC
        student = algorithm_class.load(str(spec.student_model), device="cpu")

    observations: list[np.ndarray] = []
    expert_actions: list[np.ndarray] = []
    executed_actions: list[np.ndarray] = []
    episode_ids: list[int] = []
    layout_seeds: list[int] = []
    interventions: list[bool] = []
    retained_episodes = 0
    attempted_episodes = 0
    successes = 0
    environment = None
    expert = ConeExpertPolicy()
    diagnostic_directory = spec.run_dir / "collection_logs"
    diagnostic_directory.mkdir(parents=True, exist_ok=True)
    diagnostic_path = diagnostic_directory / f"worker_{spec.rank:02d}.jsonl"
    diagnostic_path.unlink(missing_ok=True)

    def save_partial_shard() -> None:
        if not observations:
            return
        save_demonstrations(
            spec.shard_path,
            _batch_from_lists(
                observations,
                expert_actions,
                executed_actions,
                episode_ids,
                layout_seeds,
                interventions,
            ),
        )

    try:
        if spec.external_sim:
            environment = RosLidarMazeEnv(
                config=spec.config,
                node_name="rb2301_demonstration_collection",
            )
        else:
            from .managed_env import make_managed_env

            environment = make_managed_env(
                rank=spec.rank,
                base_ros_domain_id=spec.base_ros_domain_id,
                config=spec.config,
                run_dir=spec.run_dir,
                recycle_episodes=spec.recycle_episodes,
                recovery_attempts=spec.recovery_attempts,
            )()

        while (
            retained_episodes < spec.target_episodes
            and attempted_episodes < spec.maximum_attempts
        ):
            episode_observations: list[np.ndarray] = []
            episode_expert_actions: list[np.ndarray] = []
            episode_executed_actions: list[np.ndarray] = []
            episode_interventions: list[bool] = []
            observation, reset_info = environment.reset(
                seed=spec.seed + attempted_episodes
            )
            expert.reset()
            terminated = truncated = False
            final_info: dict[str, Any] = {"reason": "unknown"}
            episode_return = 0.0
            episode_minimum_scan = float("inf")
            student_steps = 0
            direct_expert_steps = 0

            while not (terminated or truncated):
                current_minimum_scan = float(
                    np.min(observation[: spec.config.ray_count])
                ) * spec.config.lidar_max_range
                episode_minimum_scan = min(
                    episode_minimum_scan,
                    current_minimum_scan,
                )
                expert_action = expert.predict(observation)
                action = expert_action
                intervention = False
                use_direct_expert = True

                if (
                    student is not None
                    and rng.random() < spec.student_control_probability
                ):
                    predicted, _ = student.predict(observation, deterministic=True)
                    student_action = np.asarray(predicted, dtype=np.float32).reshape(3)
                    if current_minimum_scan <= spec.safety_override_distance:
                        intervention = True
                    else:
                        action = np.clip(student_action, -1.0, 1.0)
                        use_direct_expert = False

                episode_observations.append(
                    np.asarray(observation, dtype=np.float32)
                )
                episode_expert_actions.append(expert_action)
                episode_interventions.append(intervention)
                if use_direct_expert:
                    direct_expert_steps += 1
                    step_result = environment.step_direct(action)
                else:
                    student_steps += 1
                    step_result = environment.step(action)

                observation, reward, terminated, truncated, final_info = step_result
                applied_action = final_info.get("applied_action", action)
                episode_executed_actions.append(
                    np.asarray(applied_action, dtype=np.float32).reshape(3)
                )
                episode_return += float(reward)
                if "minimum_scan" in final_info:
                    episode_minimum_scan = min(
                        episode_minimum_scan,
                        float(final_info["minimum_scan"]),
                    )

            attempted_episodes += 1
            reason = str(final_info.get("reason", "unknown"))
            success = reason == "success"
            infrastructure_failure = reason == "worker_restart"
            successes += int(success)
            retain = (
                (success or not spec.successful_only)
                and not infrastructure_failure
            )
            layout_seed = reset_info.get("layout_seed")
            diagnostic = {
                "worker": spec.rank,
                "attempt": attempted_episodes,
                "reset_seed": spec.seed + attempted_episodes - 1,
                "layout_seed": layout_seed,
                "return": episode_return,
                "steps": len(episode_observations),
                "reason": reason,
                "retained": retain,
                "minimum_scan": episode_minimum_scan,
                "terminal_minimum_scan": final_info.get("minimum_scan"),
                "final_x": final_info.get("x"),
                "final_y": final_info.get("y"),
                "final_yaw": final_info.get("yaw"),
                "final_requested_action": _as_json_action(
                    final_info.get("requested_action")
                ),
                "final_applied_action": _as_json_action(
                    final_info.get("applied_action")
                ),
                "action_mode": final_info.get("action_mode"),
                "student_steps": student_steps,
                "direct_expert_steps": direct_expert_steps,
                "interventions": int(sum(episode_interventions)),
            }
            with diagnostic_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(diagnostic) + "\n")
            print(
                f"worker={spec.rank:02d} attempt={attempted_episodes:04d} "
                f"return={episode_return:8.3f} "
                f"steps={len(episode_observations):4d} reason={reason} "
                f"min_scan={episode_minimum_scan:.3f} retain={retain}",
                flush=True,
            )

            if not retain:
                continue
            observations.extend(episode_observations)
            expert_actions.extend(episode_expert_actions)
            executed_actions.extend(episode_executed_actions)
            episode_ids.extend(
                [retained_episodes] * len(episode_observations)
            )
            normalized_layout_seed = -1 if layout_seed is None else int(layout_seed)
            layout_seeds.extend(
                [normalized_layout_seed] * len(episode_observations)
            )
            interventions.extend(episode_interventions)
            retained_episodes += 1
            if (
                retained_episodes % spec.shard_checkpoint_episodes == 0
                or retained_episodes == spec.target_episodes
            ):
                save_partial_shard()
    finally:
        # A Python exception or Ctrl+C still leaves all complete retained
        # episodes collected since the previous periodic shard checkpoint.
        save_partial_shard()
        if environment is not None:
            try:
                environment.close()
            except (BrokenPipeError, ConnectionResetError, EOFError):
                pass

    if retained_episodes < spec.target_episodes:
        raise RuntimeError(
            f"worker {spec.rank} collected only {retained_episodes}/"
            f"{spec.target_episodes} episodes after {attempted_episodes} attempts"
        )
    return {
        "rank": spec.rank,
        "shard": str(spec.shard_path),
        "retained_episodes": retained_episodes,
        "attempted_episodes": attempted_episodes,
        "successes": successes,
        "transitions": len(observations),
        "diagnostics": str(diagnostic_path),
    }


def main() -> None:
    args = parse_args()
    if args.episodes < 1:
        raise ValueError("--episodes must be positive")
    if args.num_envs < 1 or args.num_envs > args.episodes:
        raise ValueError("--num-envs must be between 1 and --episodes")
    if args.external_sim and args.num_envs != 1:
        raise ValueError("--external-sim requires --num-envs 1")
    if not 0.0 <= args.student_control_probability <= 1.0:
        raise ValueError("--student-control-probability must lie in [0, 1]")
    if args.safety_override_distance < 0.0:
        raise ValueError("--safety-override-distance cannot be negative")
    if args.worker_recycle_episodes < 0 or args.worker_recovery_attempts < 0:
        raise ValueError("worker lifecycle values cannot be negative")
    if args.shard_checkpoint_episodes < 1:
        raise ValueError("--shard-checkpoint-episodes must be positive")
    if args.student_model is not None and not args.student_model.is_file():
        raise FileNotFoundError(f"Student model does not exist: {args.student_model}")
    for path in args.append:
        if not path.is_file():
            raise FileNotFoundError(f"Appended dataset does not exist: {path}")

    maximum_attempts = args.max_attempted_episodes or args.episodes * 10
    if maximum_attempts < args.episodes:
        raise ValueError("--max-attempted-episodes cannot be below --episodes")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = (args.run_dir or Path("runs") / f"demonstrations_{timestamp}").resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    shard_directory = run_dir / "shards"
    shard_directory.mkdir(parents=True, exist_ok=True)
    config = replace(
        RosMazeEnvConfig(),
        layout=layout_config_for_difficulty(args.maze_difficulty),
        randomize_obstacles=True,
    )
    managed = not args.external_sim
    base_ros_domain_id = (
        select_base_ros_domain_id(args.base_ros_domain_id, args.num_envs)
        if managed
        else 0
    )
    episode_counts = allocate_episode_counts(args.episodes, args.num_envs)
    attempt_counts = allocate_episode_counts(maximum_attempts, args.num_envs)
    student_model = (
        None if args.student_model is None else args.student_model.resolve()
    )
    specs = [
        CollectionWorkerSpec(
            rank=rank,
            target_episodes=episode_counts[rank],
            maximum_attempts=attempt_counts[rank],
            seed=args.seed + rank * 1_000_003,
            config=config,
            run_dir=run_dir,
            shard_path=shard_directory / f"worker_{rank:02d}.npz",
            base_ros_domain_id=base_ros_domain_id,
            external_sim=args.external_sim,
            student_model=student_model,
            student_algorithm=args.student_algorithm,
            student_control_probability=args.student_control_probability,
            safety_override_distance=args.safety_override_distance,
            successful_only=args.successful_only,
            recycle_episodes=args.worker_recycle_episodes,
            recovery_attempts=args.worker_recovery_attempts,
            shard_checkpoint_episodes=args.shard_checkpoint_episodes,
        )
        for rank in range(args.num_envs)
    ]
    run_configuration = {
        "release": "1.1.1",
        "episodes": args.episodes,
        "num_envs": args.num_envs,
        "episode_counts": episode_counts,
        "seed": args.seed,
        "worker_seeds": [spec.seed for spec in specs],
        "maze_difficulty": args.maze_difficulty,
        "successful_only": args.successful_only,
        "student_model": None if student_model is None else str(student_model),
        "student_algorithm": args.student_algorithm,
        "student_control_probability": args.student_control_probability,
        "safety_override_distance": args.safety_override_distance,
        "base_ros_domain_id": base_ros_domain_id if managed else None,
        "expert_action_mode": "direct_unfiltered",
        "student_action_mode": "safety_aware_filter",
        "environment": asdict(config),
    }
    (run_dir / "collection_config.json").write_text(
        json.dumps(run_configuration, indent=2),
        encoding="utf-8",
    )

    summaries: list[dict[str, Any]] = []
    try:
        if args.num_envs == 1:
            summaries.append(_collect_worker(specs[0]))
        else:
            context = multiprocessing.get_context("spawn")
            with ProcessPoolExecutor(
                max_workers=args.num_envs,
                mp_context=context,
            ) as executor:
                futures = [executor.submit(_collect_worker, spec) for spec in specs]
                for future in as_completed(futures):
                    summaries.append(future.result())
    finally:
        if managed:
            from .managed_env import cleanup_managed_workers

            cleanup_managed_workers(run_dir)

    summaries.sort(key=lambda item: int(item["rank"]))
    collected_batches = [
        load_demonstrations(summary["shard"])
        for summary in summaries
    ]
    batches = [load_demonstrations(path) for path in args.append] + collected_batches
    complete = concatenate_demonstrations(batches)
    save_demonstrations(args.output, complete)
    run_configuration["workers"] = summaries
    run_configuration["output"] = str(args.output.resolve())
    run_configuration["total_transitions"] = len(complete.observations)
    (run_dir / "collection_summary.json").write_text(
        json.dumps(run_configuration, indent=2),
        encoding="utf-8",
    )
    print(
        f"Saved {len(complete.observations)} labelled transitions from "
        f"{len(np.unique(complete.episode_ids))} episodes and "
        f"{args.num_envs} worker(s) to {args.output}"
    )


if __name__ == "__main__":
    main()
