import sys
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np


fake_ros_environment = types.ModuleType("rb2301_ca1.ros_gym_env")
fake_ros_environment.RosLidarMazeEnv = object
fake_ros_environment.RosMazeEnvConfig = object
sys.modules.setdefault("rb2301_ca1.ros_gym_env", fake_ros_environment)

from rb2301_ca1.collect_demonstrations import (  # noqa: E402
    CollectionWorkerSpec,
    _collect_worker,
)
from rb2301_ca1.imitation_core import load_demonstrations  # noqa: E402


class FakeDirectEnvironment:
    direct_calls = 0
    filtered_calls = 0

    def __init__(self, config, node_name):
        del config, node_name
        self.steps = 0

    def reset(self, *, seed):
        self.steps = 0
        observation = np.ones(42, dtype=np.float32)
        observation[-3:] = 0.0
        return observation, {"layout_seed": seed + 10}

    def step(self, action):
        del action
        type(self).filtered_calls += 1
        raise AssertionError("pure expert collection must bypass filtering")

    def step_direct(self, action):
        type(self).direct_calls += 1
        self.steps += 1
        terminated = self.steps == 2
        info = {
            "reason": "success" if terminated else "running",
            "minimum_scan": 1.0,
            "x": float(self.steps),
            "y": 0.0,
            "yaw": 0.0,
            "requested_action": np.asarray(action, dtype=np.float32),
            "applied_action": np.asarray(action, dtype=np.float32),
            "action_mode": "direct_expert",
        }
        return np.ones(42, dtype=np.float32), 1.0, terminated, False, info

    def close(self):
        pass


class CollectionWorkerTests(unittest.TestCase):
    def test_pure_expert_uses_direct_steps_and_writes_shard(self):
        FakeDirectEnvironment.direct_calls = 0
        FakeDirectEnvironment.filtered_calls = 0
        with TemporaryDirectory() as directory:
            run_dir = Path(directory)
            shard_path = run_dir / "shards" / "worker_00.npz"
            shard_path.parent.mkdir()
            spec = CollectionWorkerSpec(
                rank=0,
                target_episodes=2,
                maximum_attempts=2,
                seed=100,
                config=SimpleNamespace(ray_count=36, lidar_max_range=10.0),
                run_dir=run_dir,
                shard_path=shard_path,
                base_ros_domain_id=40,
                external_sim=True,
                student_model=None,
                student_algorithm="ppo",
                student_control_probability=0.5,
                safety_override_distance=0.24,
                successful_only=True,
                recycle_episodes=200,
                recovery_attempts=2,
                shard_checkpoint_episodes=1,
            )
            with patch(
                "rb2301_ca1.collect_demonstrations.RosLidarMazeEnv",
                FakeDirectEnvironment,
            ):
                summary = _collect_worker(spec)

            batch = load_demonstrations(shard_path)
            self.assertEqual(summary["retained_episodes"], 2)
            self.assertEqual(summary["transitions"], 4)
            self.assertEqual(FakeDirectEnvironment.direct_calls, 4)
            self.assertEqual(FakeDirectEnvironment.filtered_calls, 0)
            np.testing.assert_array_equal(
                batch.expert_actions,
                batch.executed_actions,
            )
            diagnostics = (
                run_dir / "collection_logs" / "worker_00.jsonl"
            ).read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(diagnostics), 2)


if __name__ == "__main__":
    unittest.main()
