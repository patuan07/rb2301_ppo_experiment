"""Train or resume PPO/SAC for continuous holonomic LiDAR navigation."""

from __future__ import annotations

import argparse
import json
import math
import traceback
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import torch as th
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import FloatSchedule
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from .anchored_ppo import TeacherAnchoredPPO
from .maze_layout import DIFFICULTIES, layout_config_for_difficulty
from .ros_gym_env import RosLidarMazeEnv, RosMazeEnvConfig
from .runtime_utils import select_base_ros_domain_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--timesteps",
        type=int,
        default=200_000,
        help="Transitions to add; when resuming, this is additional training",
    )
    parser.add_argument("--algorithm", choices=("ppo", "sac"), default="ppo")
    parser.add_argument(
        "--num-envs",
        type=int,
        default=1,
        help="Independent managed headless Gazebo workers",
    )
    parser.add_argument(
        "--external-sim",
        action="store_true",
        help="For one environment, use an already-running simulator",
    )
    parser.add_argument(
        "--base-ros-domain-id",
        type=int,
        default=None,
        help="First managed worker ROS domain; omitted chooses a fresh block",
    )
    parser.add_argument("--seed", type=int, default=2301)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--device", default="auto", help="SB3 device, e.g. cpu or cuda")
    parser.add_argument("--checkpoint-frequency", type=int, default=25_000)
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="PPO/SAC model zip to continue; --timesteps remains additional",
    )
    parser.add_argument(
        "--resume-replay-buffer",
        type=Path,
        default=None,
        help="Optional SAC replay-buffer pickle paired with --resume",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Override learning rate; resumed models otherwise keep their value",
    )
    parser.add_argument(
        "--ent-coef",
        type=float,
        default=None,
        help=(
            "Override PPO entropy coefficient; resumed PPO models otherwise "
            "keep their saved value"
        ),
    )
    parser.add_argument(
        "--action-std",
        type=float,
        default=None,
        help=(
            "Set PPO Gaussian action standard deviation before training; "
            "resumed PPO models otherwise keep their saved value"
        ),
    )
    parser.add_argument(
        "--teacher-model",
        type=Path,
        default=None,
        help=(
            "Frozen PPO teacher used to anchor actor updates; normally the "
            "DAgger-3 model"
        ),
    )
    parser.add_argument(
        "--teacher-anchor-strength",
        type=float,
        default=0.02,
        help="Fraction of actor drift removed after each PPO update",
    )
    parser.add_argument(
        "--ppo-clip-range",
        type=float,
        default=None,
        help="Override PPO clip range when resuming or creating PPO",
    )
    parser.add_argument(
        "--ppo-n-epochs",
        type=int,
        default=None,
        help="Override PPO optimization epochs per rollout",
    )
    parser.add_argument(
        "--ppo-target-kl",
        type=float,
        default=None,
        help="Stop an update early when approximate KL exceeds this value",
    )
    parser.add_argument(
        "--maze-difficulty",
        choices=DIFFICULTIES,
        default="medium",
    )
    parser.add_argument(
        "--worker-recycle-episodes",
        type=int,
        default=200,
        help="Restart each managed simulator after this many episodes; 0 disables",
    )
    parser.add_argument(
        "--worker-recovery-attempts",
        type=int,
        default=2,
        help="Transparent simulator restart attempts after reset/step timeouts",
    )
    parser.add_argument(
        "--randomize-obstacles",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def _save_recovery_artifacts(model: Any, run_dir: Path, prefix: str) -> None:
    """Best-effort model and SAC replay save without masking the primary error."""

    try:
        model.save(str(run_dir / f"{prefix}_model"))
        print(f"Saved recovery model to {run_dir / f'{prefix}_model.zip'}")
    except Exception as save_error:
        print(f"WARNING: could not save recovery model: {save_error}")
    if isinstance(model, SAC):
        try:
            model.save_replay_buffer(str(run_dir / f"{prefix}_replay_buffer.pkl"))
            print(
                "Saved recovery replay buffer to "
                f"{run_dir / f'{prefix}_replay_buffer.pkl'}"
            )
        except Exception as save_error:
            print(f"WARNING: could not save recovery replay buffer: {save_error}")


def _set_ppo_action_std(model: PPO, action_std: float) -> None:
    """Set every diagonal Gaussian action standard deviation to one value."""

    log_std = getattr(model.policy, "log_std", None)
    if log_std is None:
        raise TypeError("The loaded PPO policy does not expose a Gaussian log_std")
    with th.no_grad():
        log_std.fill_(math.log(action_std))


def _ppo_hyperparameters(model: PPO) -> dict[str, Any]:
    """Return the effective PPO settings used at the start of this run."""

    action_std = model.policy.log_std.detach().exp().cpu().tolist()
    return {
        "learning_rate_at_start": float(model.lr_schedule(1.0)),
        "ent_coef": float(model.ent_coef),
        "action_std": [float(value) for value in action_std],
        "clip_range_at_start": float(model.clip_range(1.0)),
        "n_epochs": int(model.n_epochs),
        "target_kl": None if model.target_kl is None else float(model.target_kl),
        "teacher_anchor_strength": float(
            getattr(model, "teacher_anchor_strength", 0.0)
        ),
    }


def _load_or_create_model(
    args: argparse.Namespace,
    environment: Any,
    run_dir: Path,
) -> tuple[Any, bool]:
    anchored = args.algorithm == "ppo" and args.teacher_model is not None
    algorithm_class = (
        TeacherAnchoredPPO if anchored else PPO
    ) if args.algorithm == "ppo" else SAC
    if args.resume is not None:
        if not args.resume.is_file():
            raise FileNotFoundError(f"Resume model does not exist: {args.resume}")
        model = algorithm_class.load(
            str(args.resume),
            env=environment,
            device=args.device,
        )
        model.tensorboard_log = str(run_dir / "tensorboard")
        model.verbose = 1
        if args.learning_rate is not None:
            model.learning_rate = args.learning_rate
            model.lr_schedule = FloatSchedule(args.learning_rate)
        if isinstance(model, PPO):
            if args.ent_coef is not None:
                model.ent_coef = args.ent_coef
            if args.action_std is not None:
                _set_ppo_action_std(model, args.action_std)
            if args.ppo_clip_range is not None:
                model.clip_range = FloatSchedule(args.ppo_clip_range)
            if args.ppo_n_epochs is not None:
                model.n_epochs = args.ppo_n_epochs
            if args.ppo_target_kl is not None:
                model.target_kl = args.ppo_target_kl
            if args.teacher_model is not None:
                teacher_model = PPO.load(
                    str(args.teacher_model),
                    device=args.device,
                )
                if not isinstance(model, TeacherAnchoredPPO):
                    raise RuntimeError(
                        "--teacher-model requires a TeacherAnchoredPPO model "
                        "class; start a new anchored run from the teacher"
                    )
                model.set_teacher(teacher_model, args.teacher_anchor_strength)
        replay_buffer_path = args.resume_replay_buffer
        if isinstance(model, SAC) and replay_buffer_path is None:
            marker = "rb2301_sac_"
            if args.resume.stem.startswith(marker):
                candidate = args.resume.with_name(
                    marker
                    + "replay_buffer_"
                    + args.resume.stem[len(marker) :]
                    + ".pkl"
                )
                if candidate.is_file():
                    replay_buffer_path = candidate
        if replay_buffer_path is not None:
            if not isinstance(model, SAC):
                raise ValueError("--resume-replay-buffer is only valid for SAC")
            model.load_replay_buffer(str(replay_buffer_path))
            print(f"Loaded SAC replay buffer from {replay_buffer_path}")
        elif isinstance(model, SAC):
            print(
                "WARNING: resuming SAC without a replay buffer; use "
                "--resume-replay-buffer when one is available"
            )
        print(
            f"Resumed {args.algorithm.upper()} from {args.resume} at "
            f"{model.num_timesteps} stored timesteps"
        )
        return model, True

    learning_rate = 3e-4 if args.learning_rate is None else args.learning_rate
    common_arguments = {
        "policy": "MlpPolicy",
        "env": environment,
        "learning_rate": learning_rate,
        "gamma": 0.99,
        "policy_kwargs": {"net_arch": [128, 128]},
        "tensorboard_log": str(run_dir / "tensorboard"),
        "seed": args.seed,
        "device": args.device,
        "verbose": 1,
    }
    if args.algorithm == "ppo":
        ppo_class = TeacherAnchoredPPO if args.teacher_model is not None else PPO
        model = ppo_class(
            **common_arguments,
            n_steps=512,
            batch_size=256,
            gae_lambda=0.95,
            ent_coef=0.005 if args.ent_coef is None else args.ent_coef,
            clip_range=0.2 if args.ppo_clip_range is None else args.ppo_clip_range,
            n_epochs=10 if args.ppo_n_epochs is None else args.ppo_n_epochs,
            target_kl=args.ppo_target_kl,
        )
        if args.action_std is not None:
            _set_ppo_action_std(model, args.action_std)
        if args.teacher_model is not None:
            teacher_model = PPO.load(str(args.teacher_model), device=args.device)
            model.set_teacher(teacher_model, args.teacher_anchor_strength)
    else:
        model = SAC(
            **common_arguments,
            buffer_size=100_000,
            learning_starts=5_000,
            batch_size=256,
            train_freq=1,
            gradient_steps=1,
        )
    return model, False


def main() -> None:
    args = parse_args()
    if args.num_envs < 1:
        raise ValueError("--num-envs must be at least one")
    if args.timesteps < 1:
        raise ValueError("--timesteps must be positive")
    if args.checkpoint_frequency < 1:
        raise ValueError("--checkpoint-frequency must be positive")
    if args.external_sim and args.num_envs != 1:
        raise ValueError("--external-sim requires --num-envs 1")
    if args.worker_recycle_episodes < 0 or args.worker_recovery_attempts < 0:
        raise ValueError("worker lifecycle values cannot be negative")
    if args.resume_replay_buffer is not None and args.resume is None:
        raise ValueError("--resume-replay-buffer requires --resume")
    if args.learning_rate is not None and (
        not math.isfinite(args.learning_rate) or args.learning_rate <= 0.0
    ):
        raise ValueError("--learning-rate must be finite and greater than zero")
    if args.ent_coef is not None and (
        not math.isfinite(args.ent_coef) or args.ent_coef < 0.0
    ):
        raise ValueError("--ent-coef must be finite and non-negative")
    if args.action_std is not None and (
        not math.isfinite(args.action_std) or args.action_std <= 0.0
    ):
        raise ValueError("--action-std must be finite and greater than zero")
    if args.algorithm != "ppo" and args.ent_coef is not None:
        raise ValueError("--ent-coef is only valid with --algorithm ppo")
    if args.algorithm != "ppo" and args.action_std is not None:
        raise ValueError("--action-std is only valid with --algorithm ppo")
    if args.teacher_model is not None and args.algorithm != "ppo":
        raise ValueError("--teacher-model is only valid with --algorithm ppo")
    if args.teacher_model is not None and not args.teacher_model.is_file():
        raise FileNotFoundError(f"Teacher model does not exist: {args.teacher_model}")
    if not 0.0 <= args.teacher_anchor_strength <= 1.0:
        raise ValueError("--teacher-anchor-strength must lie in [0, 1]")
    if args.ppo_clip_range is not None and (
        not math.isfinite(args.ppo_clip_range) or not 0.0 < args.ppo_clip_range < 1.0
    ):
        raise ValueError("--ppo-clip-range must be finite and lie in (0, 1)")
    if args.ppo_n_epochs is not None and args.ppo_n_epochs < 1:
        raise ValueError("--ppo-n-epochs must be positive")
    if args.ppo_target_kl is not None and (
        not math.isfinite(args.ppo_target_kl) or args.ppo_target_kl <= 0.0
    ):
        raise ValueError("--ppo-target-kl must be finite and greater than zero")
    if args.algorithm != "ppo" and any(
        value is not None
        for value in (args.ppo_clip_range, args.ppo_n_epochs, args.ppo_target_kl)
    ):
        raise ValueError("PPO-specific optimization flags require --algorithm ppo")

    managed_simulators = not args.external_sim
    base_ros_domain_id = select_base_ros_domain_id(
        args.base_ros_domain_id,
        args.num_envs,
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_kind = f"{args.algorithm}_resume" if args.resume else args.algorithm
    run_dir = args.run_dir or Path("runs") / f"{run_kind}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    config = replace(
        RosMazeEnvConfig(),
        randomize_obstacles=args.randomize_obstacles,
        layout=layout_config_for_difficulty(args.maze_difficulty),
    )
    run_configuration = {
        "release": "1.1.1",
        "algorithm": args.algorithm.upper(),
        "additional_timesteps": args.timesteps,
        "num_envs": args.num_envs,
        "seed": args.seed,
        "device": args.device,
        "resume": None if args.resume is None else str(args.resume.resolve()),
        "resume_replay_buffer": (
            None
            if args.resume_replay_buffer is None
            else str(args.resume_replay_buffer.resolve())
        ),
        "requested_hyperparameter_overrides": {
            "learning_rate": args.learning_rate,
            "ppo_ent_coef": args.ent_coef,
            "ppo_action_std": args.action_std,
            "teacher_model": (
                None if args.teacher_model is None else str(args.teacher_model.resolve())
            ),
            "teacher_anchor_strength": args.teacher_anchor_strength,
            "ppo_clip_range": args.ppo_clip_range,
            "ppo_n_epochs": args.ppo_n_epochs,
            "ppo_target_kl": args.ppo_target_kl,
        },
        "checkpoint_frequency": args.checkpoint_frequency,
        "maze_difficulty": args.maze_difficulty,
        "managed_simulators": managed_simulators,
        "base_ros_domain_id": base_ros_domain_id if managed_simulators else None,
        "worker_recycle_episodes": args.worker_recycle_episodes,
        "worker_recovery_attempts": args.worker_recovery_attempts,
        "environment": asdict(config),
    }
    (run_dir / "run_config.json").write_text(
        json.dumps(run_configuration, indent=2),
        encoding="utf-8",
    )

    environment = None
    model = None
    try:
        if args.external_sim:
            environment = Monitor(
                RosLidarMazeEnv(config=config),
                filename=str(run_dir / "monitor.csv"),
                info_keywords=("reason", "minimum_scan", "x", "y", "yaw"),
            )
        else:
            from .managed_env import make_managed_env

            factories = [
                make_managed_env(
                    rank=rank,
                    base_ros_domain_id=base_ros_domain_id,
                    config=config,
                    run_dir=run_dir,
                    recycle_episodes=args.worker_recycle_episodes,
                    recovery_attempts=args.worker_recovery_attempts,
                )
                for rank in range(args.num_envs)
            ]
            if args.num_envs == 1:
                environment = Monitor(
                    factories[0](),
                    filename=str(run_dir / "monitor.csv"),
                    info_keywords=("reason", "minimum_scan", "x", "y", "yaw"),
                )
            else:
                environment = VecMonitor(
                    SubprocVecEnv(factories, start_method="spawn"),
                    filename=str(run_dir / "monitor.csv"),
                    info_keywords=("reason", "minimum_scan", "x", "y", "yaw"),
                )

        checkpoint = CheckpointCallback(
            save_freq=max(args.checkpoint_frequency // args.num_envs, 1),
            save_path=str(run_dir / "checkpoints"),
            name_prefix=f"rb2301_{args.algorithm}",
            save_replay_buffer=args.algorithm == "sac",
        )
        model, resumed = _load_or_create_model(args, environment, run_dir)
        run_configuration["starting_timesteps"] = int(model.num_timesteps)
        if isinstance(model, PPO):
            effective_hyperparameters = _ppo_hyperparameters(model)
            run_configuration["effective_ppo_hyperparameters"] = (
                effective_hyperparameters
            )
            print(
                "Effective PPO hyperparameters: "
                f"learning_rate={effective_hyperparameters['learning_rate_at_start']}, "
                f"ent_coef={effective_hyperparameters['ent_coef']}, "
                f"action_std={effective_hyperparameters['action_std']}, "
                f"clip_range={effective_hyperparameters['clip_range_at_start']}, "
                f"n_epochs={effective_hyperparameters['n_epochs']}, "
                f"target_kl={effective_hyperparameters['target_kl']}, "
                f"teacher_anchor_strength="
                f"{effective_hyperparameters['teacher_anchor_strength']}"
            )
        (run_dir / "run_config.json").write_text(
            json.dumps(run_configuration, indent=2),
            encoding="utf-8",
        )
        model.learn(
            total_timesteps=args.timesteps,
            callback=checkpoint,
            reset_num_timesteps=not resumed,
        )
        model.save(str(run_dir / "final_model"))
        if isinstance(model, SAC):
            model.save_replay_buffer(str(run_dir / "final_replay_buffer.pkl"))
        print(f"Saved trained model to {run_dir / 'final_model.zip'}")
    except KeyboardInterrupt:
        if model is not None:
            _save_recovery_artifacts(model, run_dir, "interrupted")
    except Exception:
        (run_dir / "emergency_error.txt").write_text(
            traceback.format_exc(),
            encoding="utf-8",
        )
        if model is not None:
            _save_recovery_artifacts(model, run_dir, "emergency")
        raise
    finally:
        if environment is not None:
            try:
                environment.close()
            except (BrokenPipeError, ConnectionResetError, EOFError):
                pass
        if managed_simulators:
            from .managed_env import cleanup_managed_workers

            cleanup_managed_workers(run_dir)


if __name__ == "__main__":
    main()
