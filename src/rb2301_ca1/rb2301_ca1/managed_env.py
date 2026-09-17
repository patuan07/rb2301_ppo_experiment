"""Launch, monitor, recycle, and recover isolated headless Gazebo workers."""

from __future__ import annotations

import json
import os
import signal
import shutil
import subprocess
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

import gymnasium as gym
import numpy as np

from .gazebo_control import GazeboCommandError, GazeboController


RECOVERABLE_ERRORS = (
    GazeboCommandError,
    TimeoutError,
    BrokenPipeError,
    ConnectionError,
)


@dataclass
class WorkerRuntime:
    environment: Any
    process: subprocess.Popen[str]
    log_file: Any


class ManagedEnvironment(gym.Env):
    """Gym proxy that restarts only the Gazebo/ROS process group it owns."""

    def __init__(
        self,
        launcher: Callable[[int], WorkerRuntime],
        *,
        recycle_episodes: int = 200,
        recovery_attempts: int = 2,
        restart_log: Path | None = None,
    ) -> None:
        if recycle_episodes < 0:
            raise ValueError("recycle_episodes cannot be negative")
        if recovery_attempts < 0:
            raise ValueError("recovery_attempts cannot be negative")
        self._launcher = launcher
        self._recycle_episodes = recycle_episodes
        self._recovery_attempts = recovery_attempts
        self._restart_log = restart_log
        self._restart_count = 0
        self._episodes_started = 0
        self._closed = False
        last_error: Exception | None = None
        for attempt in range(self._recovery_attempts + 1):
            try:
                self._runtime = self._launcher(self._restart_count)
                break
            except Exception as launch_error:
                last_error = launch_error
                self._record_restart("initial_launch_failure", launch_error)
                if attempt < self._recovery_attempts:
                    self._restart_count += 1
                    time.sleep(2.0)
        else:
            raise RuntimeError(
                "Gazebo worker could not complete its initial launch after "
                f"{self._recovery_attempts + 1} attempt(s)"
            ) from last_error
        self.action_space = self._runtime.environment.action_space
        self.observation_space = self._runtime.environment.observation_space
        self.metadata = self._runtime.environment.metadata
        self.render_mode = None

    @property
    def environment(self) -> Any:
        return self._runtime.environment

    @property
    def process(self) -> subprocess.Popen[str]:
        return self._runtime.process

    def __getattr__(self, name: str) -> Any:
        return getattr(self.environment, name)

    def _record_restart(self, reason: str, error: Exception | None) -> None:
        if self._restart_log is None:
            return
        event = {
            "time": time.time(),
            "restart_count": self._restart_count,
            "reason": reason,
            "error": None if error is None else f"{type(error).__name__}: {error}",
        }
        with self._restart_log.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event) + "\n")

    def _restart(self, reason: str, error: Exception | None = None) -> None:
        _close_runtime(self._runtime)
        last_error: Exception | None = None
        for attempt in range(self._recovery_attempts + 1):
            self._restart_count += 1
            self._record_restart(reason, error)
            try:
                self._runtime = self._launcher(self._restart_count)
                self._episodes_started = 0
                return
            except Exception as launch_error:
                last_error = launch_error
                self._record_restart("replacement_launch_failure", launch_error)
                if attempt < self._recovery_attempts:
                    time.sleep(2.0)
        raise RuntimeError(
            f"Gazebo worker could not recover after {self._recovery_attempts + 1} "
            f"launch attempt(s)"
        ) from last_error

    def reset(self, **kwargs: Any) -> Any:
        if (
            self._recycle_episodes > 0
            and self._episodes_started >= self._recycle_episodes
        ):
            self._restart("periodic_recycle")

        last_error: Exception | None = None
        for attempt in range(self._recovery_attempts + 1):
            try:
                result = self.environment.reset(**kwargs)
                self._episodes_started += 1
                return result
            except RECOVERABLE_ERRORS as error:
                last_error = error
                if attempt >= self._recovery_attempts:
                    break
                self._restart("reset_failure", error)
        raise RuntimeError(
            f"Gazebo reset failed after {self._recovery_attempts + 1} "
            "managed attempt(s)"
        ) from last_error

    def step(self, action: Any) -> Any:
        return self._step_method("step", action)

    def step_direct(self, action: Any) -> Any:
        """Preserve recovery while bypassing filtering for expert commands."""

        return self._step_method("step_direct", action)

    def _step_method(self, method_name: str, action: Any) -> Any:
        try:
            return getattr(self.environment, method_name)(action)
        except RECOVERABLE_ERRORS as error:
            # A vector worker must return a valid boundary transition instead of
            # dying and breaking every other worker's control pipe. SB3
            # will immediately call reset(), now against the restarted world.
            self._restart("step_failure", error)
            observation = np.zeros(
                self.observation_space.shape,
                dtype=self.observation_space.dtype,
            )
            return observation, 0.0, False, True, {
                "reason": "worker_restart",
                "minimum_scan": 0.0,
                "x": 0.0,
                "y": 0.0,
                "yaw": 0.0,
                "step_count": 0,
                "action_mode": method_name,
                "infrastructure_error": f"{type(error).__name__}: {error}",
            }

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        _close_runtime(self._runtime)


def _close_runtime(runtime: WorkerRuntime) -> None:
    try:
        runtime.environment.close()
    except Exception:
        pass
    finally:
        _stop_process_group(runtime.process)
        runtime.log_file.close()


def _stop_process_group(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGINT)
        process.wait(timeout=8.0)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=2.0)
    except ProcessLookupError:
        pass


def _is_expected_launch_process(process_id: int) -> bool:
    try:
        command = Path(f"/proc/{process_id}/cmdline").read_bytes().replace(
            b"\0", b" "
        )
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return False
    return b"ros2" in command and b"ca1_gazebo.launch.py" in command


def cleanup_managed_workers(run_dir: Path) -> None:
    """Best-effort fallback cleanup if a SubprocVecEnv control pipe dies."""

    process_ids: list[int] = []
    for config_path in (Path(run_dir) / "gazebo_logs").glob(
        "worker_*/worker_config.json"
    ):
        try:
            process_id = int(json.loads(config_path.read_text())["launch_pid"])
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
        if process_id > 1 and _is_expected_launch_process(process_id):
            process_ids.append(process_id)

    for process_id in process_ids:
        try:
            os.killpg(process_id, signal.SIGINT)
        except ProcessLookupError:
            pass
    if process_ids:
        time.sleep(1.0)
    for process_id in process_ids:
        if not _is_expected_launch_process(process_id):
            continue
        try:
            os.killpg(process_id, signal.SIGTERM)
        except ProcessLookupError:
            pass


def make_managed_env(
    *,
    rank: int,
    base_ros_domain_id: int,
    config: Any,
    run_dir: Path,
    recycle_episodes: int = 200,
    recovery_attempts: int = 2,
) -> Callable[[], ManagedEnvironment]:
    """Return a picklable factory used by ``SubprocVecEnv(start_method='spawn')``."""

    def initialize() -> ManagedEnvironment:
        worker_name = f"worker_{rank:02d}"
        worker_log_directory = (
            Path(run_dir) / "gazebo_logs" / worker_name
        ).resolve()
        worker_log_directory.mkdir(parents=True, exist_ok=True)
        worker_config_path = worker_log_directory / "worker_config.json"
        worker_config_path.unlink(missing_ok=True)
        restart_log = worker_log_directory / "restart_events.jsonl"
        restart_log.unlink(missing_ok=True)
        ros_domain_id = base_ros_domain_id + rank
        os.environ["ROS_DOMAIN_ID"] = str(ros_domain_id)
        os.environ["ROS_LOG_DIR"] = str(worker_log_directory / "ros")

        def launch_runtime(restart_count: int) -> WorkerRuntime:
            gazebo_partition = (
                f"rb2301_rl_{os.getpid()}_{rank}_{restart_count:03d}"
            )
            os.environ["GZ_PARTITION"] = gazebo_partition

            from ament_index_python.packages import get_package_share_directory

            from .obstacle_generator import generate_sdf_file

            package_directory = Path(get_package_share_directory("rb2301_gz"))
            worker_world = worker_log_directory / (
                f"obstacle_world_ca1_{restart_count:03d}.sdf"
            )
            shutil.copy2(
                package_directory / "worlds" / "obstacle_world_ca1.sdf",
                worker_world,
            )
            generate_sdf_file(
                worker_world,
                package_directory / "meshes" / "coke" / "6",
                seed=2301 + rank + restart_count * 10_000,
                layout_config=config.layout,
            )
            log_file = (worker_log_directory / f"launch_{restart_count:03d}.log").open(
                "w", encoding="utf-8", buffering=1
            )
            process = subprocess.Popen(
                [
                    "ros2",
                    "launch",
                    "rb2301_gz",
                    "ca1_gazebo.launch.py",
                    "headless:=true",
                    "randomize_world:=false",
                    f"world_path:={worker_world}",
                ],
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            worker_config_path.write_text(
                json.dumps(
                    {
                        "rank": rank,
                        "restart_count": restart_count,
                        "ros_domain_id": ros_domain_id,
                        "gz_partition": gazebo_partition,
                        "world_path": str(worker_world),
                        "launch_pid": process.pid,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            try:
                from .ros_gym_env import RosLidarMazeEnv

                gazebo = GazeboController(
                    world_name=config.world_name,
                    timeout_seconds=30.0,
                    retry_attempts=3,
                    retry_delay_seconds=1.0,
                )
                gazebo.ensure_available()
                gazebo.wait_for_service(
                    f"/world/{config.world_name}/set_pose_vector",
                    timeout_seconds=45.0,
                )
                environment = RosLidarMazeEnv(
                    config=replace(config, message_timeout_seconds=45.0),
                    node_name=f"rb2301_rl_environment_{rank}",
                    gazebo=gazebo,
                )
                return WorkerRuntime(environment, process, log_file)
            except Exception as error:
                _stop_process_group(process)
                log_file.close()
                raise RuntimeError(
                    f"Gazebo worker {rank} restart {restart_count} failed to start; "
                    f"inspect {worker_log_directory / f'launch_{restart_count:03d}.log'}"
                ) from error

        return ManagedEnvironment(
            launch_runtime,
            recycle_episodes=recycle_episodes,
            recovery_attempts=recovery_attempts,
            restart_log=restart_log,
        )

    return initialize
