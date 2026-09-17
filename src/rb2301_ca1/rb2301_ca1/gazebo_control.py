"""Small, testable wrapper around Gazebo Harmonic transport services."""

from __future__ import annotations

import math
import shutil
import subprocess
import time
from collections.abc import Callable, Sequence

from .maze_layout import LayoutConfig, MazeLayout, generate_layout


class GazeboCommandError(RuntimeError):
    """Raised when a Gazebo transport service fails."""


class GazeboController:
    def __init__(
        self,
        world_name: str = "empty",
        timeout_seconds: float = 5.0,
        retry_attempts: int = 1,
        retry_delay_seconds: float = 0.5,
        runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    ) -> None:
        if timeout_seconds <= 0.0:
            raise ValueError("timeout_seconds must be positive")
        if retry_attempts <= 0:
            raise ValueError("retry_attempts must be positive")
        self.world_name = world_name
        self.timeout_seconds = timeout_seconds
        self.retry_attempts = retry_attempts
        self.retry_delay_seconds = retry_delay_seconds
        self._runner = runner

    def ensure_available(self) -> None:
        if shutil.which("gz") is None:
            raise GazeboCommandError(
                "The 'gz' executable was not found. Source ROS/Gazebo before training."
            )

    def wait_for_service(
        self,
        service: str,
        timeout_seconds: float = 45.0,
        poll_interval_seconds: float = 0.5,
    ) -> None:
        """Wait until a service appears inside this worker's GZ partition."""

        if timeout_seconds <= 0.0:
            raise ValueError("timeout_seconds must be positive")
        if poll_interval_seconds <= 0.0:
            raise ValueError("poll_interval_seconds must be positive")

        deadline = time.monotonic() + timeout_seconds
        last_detail = "service list was empty"
        while time.monotonic() < deadline:
            remaining = max(deadline - time.monotonic(), 0.01)
            try:
                result = self._runner(
                    ["gz", "service", "-l"],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=min(3.0, remaining),
                )
                output = f"{result.stdout}\n{result.stderr}".strip()
                if result.returncode == 0 and service in output.splitlines():
                    return
                last_detail = f"code {result.returncode}: {output or 'no services'}"
            except subprocess.TimeoutExpired as error:
                last_detail = f"service listing timed out after {error.timeout} seconds"

            time.sleep(min(poll_interval_seconds, remaining))

        raise GazeboCommandError(
            f"Gazebo service {service!r} did not appear within "
            f"{timeout_seconds:.1f} seconds ({last_detail})"
        )

    def _call(self, service: str, request_type: str, request: str) -> str:
        command = [
            "gz",
            "service",
            "-s",
            service,
            "--reqtype",
            request_type,
            "--reptype",
            "gz.msgs.Boolean",
            "--timeout",
            str(int(self.timeout_seconds * 1000)),
            "--req",
            request,
        ]
        last_detail = "no response"
        for attempt in range(1, self.retry_attempts + 1):
            try:
                result = self._runner(
                    command,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_seconds + 1.0,
                )
                output = f"{result.stdout}\n{result.stderr}".strip()
                if result.returncode == 0 and "data: true" in output.lower():
                    return output
                last_detail = f"code {result.returncode}: {output}"
            except subprocess.TimeoutExpired as error:
                last_detail = f"client timeout after {error.timeout} seconds"

            if attempt < self.retry_attempts:
                time.sleep(self.retry_delay_seconds)

        raise GazeboCommandError(
            f"Gazebo service {service!r} failed after {self.retry_attempts} "
            f"attempt(s): {last_detail}"
        )

    @property
    def _world_prefix(self) -> str:
        return f"/world/{self.world_name}"

    def reset_world(self) -> None:
        self._call(
            f"{self._world_prefix}/control",
            "gz.msgs.WorldControl",
            "reset: {all: true}",
        )

    def set_entity_pose(
        self,
        name: str,
        x: float,
        y: float,
        z: float = 0.03,
        yaw: float = 0.0,
    ) -> None:
        half_yaw = yaw / 2.0
        request = (
            f'name: "{name}" '
            f"position {{x: {x:.8f} y: {y:.8f} z: {z:.8f}}} "
            "orientation {"
            f"x: 0 y: 0 z: {math.sin(half_yaw):.10f} w: {math.cos(half_yaw):.10f}"
            "}"
        )
        self._call(
            f"{self._world_prefix}/set_pose",
            "gz.msgs.Pose",
            request,
        )

    def set_entity_poses(
        self,
        poses: Sequence[tuple[str, float, float, float, float]],
    ) -> None:
        messages = []
        for name, x, y, z, yaw in poses:
            half_yaw = yaw / 2.0
            messages.append(
                "pose {"
                f' name: "{name}"'
                f" position {{x: {x:.8f} y: {y:.8f} z: {z:.8f}}}"
                " orientation {"
                f"x: 0 y: 0 z: {math.sin(half_yaw):.10f} w: {math.cos(half_yaw):.10f}"
                "}"
                " }"
            )
        self._call(
            f"{self._world_prefix}/set_pose_vector",
            "gz.msgs.Pose_V",
            " ".join(messages),
        )

    def randomize_obstacles(
        self,
        seed: int | None = None,
        layout_config: LayoutConfig = LayoutConfig(),
    ) -> MazeLayout:
        layout = generate_layout(seed, layout_config)
        poses = [
            (f"coke{index}", x, y, z, 0.0)
            for index, (x, y, z) in enumerate(layout.all_positions, start=1)
        ]
        self.set_entity_poses(poses)
        return layout
