import sys
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np


if "gymnasium" not in sys.modules:
    fake_gym = types.ModuleType("gymnasium")

    class FakeGymEnv:
        pass

    fake_gym.Env = FakeGymEnv
    sys.modules["gymnasium"] = fake_gym

from rb2301_ca1.gazebo_control import GazeboCommandError
from rb2301_ca1.managed_env import ManagedEnvironment, WorkerRuntime


class FakeSpace:
    shape = (42,)
    dtype = np.dtype(np.float32)


class FakeProcess:
    pid = 999999

    def poll(self):
        return 0


class FakeLog:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class FakeEnvironment:
    metadata = {"render_modes": []}
    action_space = FakeSpace()
    observation_space = FakeSpace()

    def __init__(self, *, reset_error=False, step_error=False):
        self.reset_error = reset_error
        self.step_error = step_error
        self.closed = False

    def reset(self, **kwargs):
        del kwargs
        if self.reset_error:
            raise GazeboCommandError("simulated reset timeout")
        return np.ones(42, dtype=np.float32), {"x": 0.0}

    def step(self, action):
        del action
        if self.step_error:
            raise TimeoutError("simulated sensor timeout")
        return np.ones(42, dtype=np.float32), 1.0, False, False, {}

    def step_direct(self, action):
        return self.step(action)

    def close(self):
        self.closed = True


class ManagedEnvironmentTests(unittest.TestCase):
    def runtime(self, environment):
        return WorkerRuntime(environment, FakeProcess(), FakeLog())

    def test_reset_failure_restarts_and_retries(self):
        environments = [
            FakeEnvironment(reset_error=True),
            FakeEnvironment(),
        ]
        calls = []

        def launcher(restart_count):
            calls.append(restart_count)
            return self.runtime(environments[len(calls) - 1])

        managed = ManagedEnvironment(launcher, recovery_attempts=1)
        observation, _ = managed.reset(seed=3)
        self.assertTrue(np.all(observation == 1.0))
        self.assertEqual(calls, [0, 1])
        self.assertTrue(environments[0].closed)
        managed.close()

    def test_initial_launch_failure_is_retried(self):
        calls = []

        def launcher(restart_count):
            calls.append(restart_count)
            if len(calls) == 1:
                raise RuntimeError("simulated launch failure")
            return self.runtime(FakeEnvironment())

        with patch("rb2301_ca1.managed_env.time.sleep"):
            managed = ManagedEnvironment(launcher, recovery_attempts=1)
        self.assertEqual(calls, [0, 1])
        managed.close()

    def test_periodic_recycling_happens_before_next_reset(self):
        calls = []

        def launcher(restart_count):
            calls.append(restart_count)
            return self.runtime(FakeEnvironment())

        managed = ManagedEnvironment(launcher, recycle_episodes=1)
        managed.reset()
        managed.reset()
        self.assertEqual(calls, [0, 1])
        managed.close()

    def test_step_failure_becomes_truncation_after_restart(self):
        environments = [
            FakeEnvironment(step_error=True),
            FakeEnvironment(),
        ]
        calls = []

        def launcher(restart_count):
            calls.append(restart_count)
            return self.runtime(environments[len(calls) - 1])

        with TemporaryDirectory() as directory:
            managed = ManagedEnvironment(
                launcher,
                recovery_attempts=1,
                restart_log=Path(directory) / "restarts.jsonl",
            )
            _, reward, terminated, truncated, info = managed.step(np.zeros(3))
            self.assertEqual(reward, 0.0)
            self.assertFalse(terminated)
            self.assertTrue(truncated)
            self.assertEqual(info["reason"], "worker_restart")
            self.assertEqual(calls, [0, 1])
            managed.close()

    def test_direct_expert_step_uses_managed_recovery_path(self):
        environments = [
            FakeEnvironment(step_error=True),
            FakeEnvironment(),
        ]
        calls = []

        def launcher(restart_count):
            calls.append(restart_count)
            return self.runtime(environments[len(calls) - 1])

        managed = ManagedEnvironment(launcher, recovery_attempts=1)
        _, _, _, truncated, info = managed.step_direct(np.zeros(3))
        self.assertTrue(truncated)
        self.assertEqual(info["reason"], "worker_restart")
        self.assertEqual(calls, [0, 1])
        managed.close()


if __name__ == "__main__":
    unittest.main()
