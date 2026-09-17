"""Exercise ROS topics, Gazebo reset, actions, rewards, and termination."""

from __future__ import annotations

import argparse
from dataclasses import replace

import numpy as np

from .maze_layout import DIFFICULTIES, layout_config_for_difficulty
from .ros_gym_env import RosLidarMazeEnv, RosMazeEnvConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=2301)
    parser.add_argument("--maze-difficulty", choices=DIFFICULTIES, default="medium")
    parser.add_argument(
        "--randomize-obstacles",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = replace(
        RosMazeEnvConfig(),
        max_episode_steps=args.steps,
        randomize_obstacles=args.randomize_obstacles,
        layout=layout_config_for_difficulty(args.maze_difficulty),
    )
    environment = RosLidarMazeEnv(config=config, node_name="rb2301_rl_smoke_test")
    try:
        observation, reset_info = environment.reset(seed=args.seed)
        print(
            f"reset ok: observation={observation.shape}, "
            f"range=[{observation.min():.3f}, {observation.max():.3f}], "
            f"layout_seed={reset_info['layout_seed']}"
        )
        for index in range(args.steps):
            action = np.asarray(environment.action_space.sample(), dtype=np.float32)
            observation, reward, terminated, truncated, info = environment.step(action)
            print(
                f"step={index + 1:03d} action={np.round(action, 2)} reward={reward:7.3f} "
                f"position=({info['x']:.2f}, {info['y']:.2f}) "
                f"clearance={info['minimum_scan']:.2f} reason={info['reason']}"
            )
            if terminated or truncated:
                break
        print("ROS/Gazebo smoke test passed")
    finally:
        environment.close()


if __name__ == "__main__":
    main()
