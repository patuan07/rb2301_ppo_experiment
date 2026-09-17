"""Pretrain or update a PPO actor by behavior cloning expert labels."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO

from .imitation_core import (
    concatenate_demonstrations,
    episode_split_indices,
    load_demonstrations,
)
from .rl_core import observation_bounds


class ImitationSpaceEnv(gym.Env[np.ndarray, np.ndarray]):
    """Minimal environment used only to construct a serializable SB3 policy."""

    metadata = {"render_modes": []}

    def __init__(self, observation_width: int) -> None:
        if observation_width != 42:
            raise ValueError("The current policy interface requires 42 observations")
        # These bounds intentionally match RosLidarMazeEnv exactly. SB3 embeds
        # spaces in its model archive and refuses to resume into an environment
        # whose bounds differ, even when the vector shape is identical.
        observation_low, observation_high = observation_bounds(36)
        self.observation_space = gym.spaces.Box(
            low=observation_low,
            high=observation_high,
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(3,),
            dtype=np.float32,
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        del options
        return np.zeros(self.observation_space.shape, dtype=np.float32), {}

    def step(self, action):
        del action
        return (
            np.zeros(self.observation_space.shape, dtype=np.float32),
            0.0,
            False,
            True,
            {},
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--initial-model", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--action-std", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=2301)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def _mean_action(model: PPO, observations: torch.Tensor) -> torch.Tensor:
    distribution = model.policy.get_distribution(observations)
    return distribution.distribution.mean


def _mean_squared_error(
    model: PPO,
    observations: np.ndarray,
    actions: np.ndarray,
    batch_size: int,
) -> float:
    losses: list[float] = []
    model.policy.set_training_mode(False)
    with torch.no_grad():
        for start in range(0, len(observations), batch_size):
            observation_tensor = torch.as_tensor(
                observations[start : start + batch_size],
                device=model.device,
            )
            action_tensor = torch.as_tensor(
                actions[start : start + batch_size],
                device=model.device,
            )
            prediction = _mean_action(model, observation_tensor)
            losses.append(
                float(torch.mean(torch.square(prediction - action_tensor)).cpu())
            )
    return float(np.mean(losses))


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("--epochs and --batch-size must be positive")
    if args.learning_rate <= 0.0:
        raise ValueError("--learning-rate must be positive")
    if args.action_std <= 0.0:
        raise ValueError("--action-std must be positive")

    batch = concatenate_demonstrations(
        [load_demonstrations(path) for path in args.dataset]
    )
    if batch.observations.shape[1] != 42:
        raise ValueError(
            "This release expects 42 observation values; dataset contains "
            f"{batch.observations.shape[1]}"
        )
    train_indices, validation_indices = episode_split_indices(
        batch.episode_ids,
        args.validation_fraction,
        args.seed,
    )
    environment = ImitationSpaceEnv(batch.observations.shape[1])
    if args.initial_model is None:
        model = PPO(
            "MlpPolicy",
            environment,
            learning_rate=3e-4,
            n_steps=512,
            batch_size=256,
            gae_lambda=0.95,
            gamma=0.99,
            ent_coef=0.005,
            policy_kwargs={"net_arch": [128, 128]},
            seed=args.seed,
            device=args.device,
            verbose=0,
        )
    else:
        model = PPO.load(
            str(args.initial_model),
            env=environment,
            device=args.device,
        )

    optimizer = torch.optim.Adam(model.policy.parameters(), lr=args.learning_rate)
    rng = np.random.default_rng(args.seed)
    train_observations = batch.observations[train_indices]
    train_actions = batch.expert_actions[train_indices]
    validation_observations = batch.observations[validation_indices]
    validation_actions = batch.expert_actions[validation_indices]
    history: list[dict[str, float | int]] = []

    for epoch in range(1, args.epochs + 1):
        model.policy.set_training_mode(True)
        shuffled = rng.permutation(len(train_observations))
        epoch_losses: list[float] = []
        for start in range(0, len(shuffled), args.batch_size):
            indices = shuffled[start : start + args.batch_size]
            observation_tensor = torch.as_tensor(
                train_observations[indices],
                device=model.device,
            )
            action_tensor = torch.as_tensor(
                train_actions[indices],
                device=model.device,
            )
            optimizer.zero_grad()
            prediction = _mean_action(model, observation_tensor)
            loss = torch.mean(torch.square(prediction - action_tensor))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.policy.parameters(), 0.5)
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))

        validation_loss = _mean_squared_error(
            model,
            validation_observations,
            validation_actions,
            args.batch_size,
        )
        train_loss = float(np.mean(epoch_losses))
        history.append(
            {
                "epoch": epoch,
                "train_mse": train_loss,
                "validation_mse": validation_loss,
            }
        )
        print(
            f"epoch={epoch:03d} train_mse={train_loss:.6f} "
            f"validation_mse={validation_loss:.6f}"
        )

    if not hasattr(model.policy, "log_std"):
        raise RuntimeError("PPO policy does not expose a continuous-action log_std")
    with torch.no_grad():
        model.policy.log_std.fill_(math.log(args.action_std))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(args.output))
    metrics_path = args.output.with_suffix(".bc_metrics.json")
    metrics_path.write_text(
        json.dumps(
            {
                "datasets": [str(path.resolve()) for path in args.dataset],
                "initial_model": (
                    None
                    if args.initial_model is None
                    else str(args.initial_model.resolve())
                ),
                "samples": len(batch.observations),
                "episodes": int(len(np.unique(batch.episode_ids))),
                "train_samples": len(train_indices),
                "validation_samples": len(validation_indices),
                "action_std": args.action_std,
                "history": history,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved behavior-cloned PPO model to {args.output}")
    print(f"Saved training metrics to {metrics_path}")


if __name__ == "__main__":
    main()
