"""Gymnasium environment that connects Stable-Baselines3 to ROS 2 and Gazebo."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import gymnasium as gym
import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from sensor_msgs.msg import LaserScan

from .gazebo_control import GazeboController
from .maze_layout import LayoutConfig
from .rl_core import (
    RewardConfig,
    action_to_velocity,
    build_observation,
    evaluate_transition,
    observation_bounds,
    sample_scan_metres,
    smooth_normalized_action,
)


@dataclass(frozen=True)
class RosMazeEnvConfig:
    scan_topic: str = "/scan"
    odom_topic: str = "/odom"
    cmd_vel_topic: str = "/cmd_vel"
    world_name: str = "empty"
    robot_name: str = "nanocar"
    ray_count: int = 36
    lidar_min_range: float = 0.05
    lidar_max_range: float = 10.0
    max_x_velocity: float = 0.4
    max_y_velocity: float = 0.4
    max_turn_velocity: float = 0.8
    action_smoothing_alpha: float = 0.15
    scans_per_action: int = 1
    message_timeout_seconds: float = 5.0
    start_x: float = 0.0
    start_y: float = 0.0
    start_z: float = 0.03
    start_yaw: float = 0.0
    goal_x: float = 7.2
    goal_y: float = 0.0
    goal_distance_scale: float = 8.0
    corridor_half_width: float = 2.15
    collision_distance: float = 0.12
    max_episode_steps: int = 800
    randomize_obstacles: bool = True
    layout: LayoutConfig = field(default_factory=LayoutConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)


class RosLidarMazeEnv(gym.Env[np.ndarray, np.ndarray]):
    """Continuous 3-DOF, 36-ray navigation task for the RB2301 NanoCar.

    Gazebo must already be running. The environment owns one ROS node and spins
    it synchronously, which keeps every action aligned with fresh sensor data.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        config: RosMazeEnvConfig | None = None,
        node_name: str = "rb2301_rl_environment",
        gazebo: GazeboController | None = None,
    ) -> None:
        super().__init__()
        self.config = config or RosMazeEnvConfig()
        if self.config.scans_per_action <= 0:
            raise ValueError("scans_per_action must be positive")

        observation_low, observation_high = observation_bounds(
            self.config.ray_count
        )
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

        self._owns_rclpy = not rclpy.ok()
        if self._owns_rclpy:
            rclpy.init()
        self.node = Node(node_name)
        self.gazebo = gazebo or GazeboController(self.config.world_name)
        self.gazebo.ensure_available()

        self._publisher = self.node.create_publisher(
            Twist,
            self.config.cmd_vel_topic,
            10,
        )
        scan_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self._scan_subscription = self.node.create_subscription(
            LaserScan,
            self.config.scan_topic,
            self._scan_callback,
            scan_qos,
        )
        self._odom_subscription = self.node.create_subscription(
            Odometry,
            self.config.odom_topic,
            self._odom_callback,
            QoSProfile(depth=1),
        )

        self._raw_scan: np.ndarray | None = None
        self._x: float | None = None
        self._y: float | None = None
        self._yaw: float | None = None
        self._scan_sequence = 0
        self._odom_sequence = 0
        self._previous_x = self.config.start_x
        self._applied_action = np.zeros(3, dtype=np.float32)
        self._step_count = 0
        self._needs_reset = True
        self._closed = False

        self.node.get_logger().info(
            f"Waiting for {self.config.scan_topic} and {self.config.odom_topic}..."
        )
        self._wait_for_initial_messages()

    def _scan_callback(self, message: LaserScan) -> None:
        self._raw_scan = np.asarray(message.ranges, dtype=np.float32)
        self._scan_sequence += 1

    def _odom_callback(self, message: Odometry) -> None:
        self._x = float(message.pose.pose.position.x)
        self._y = float(message.pose.pose.position.y)
        orientation = message.pose.pose.orientation
        sin_yaw = 2.0 * (
            float(orientation.w) * float(orientation.z)
            + float(orientation.x) * float(orientation.y)
        )
        cos_yaw = 1.0 - 2.0 * (
            float(orientation.y) ** 2 + float(orientation.z) ** 2
        )
        self._yaw = float(np.arctan2(sin_yaw, cos_yaw))
        self._odom_sequence += 1

    def _wait_until(self, predicate: Any, description: str) -> None:
        deadline = time.monotonic() + self.config.message_timeout_seconds
        while time.monotonic() < deadline:
            rclpy.spin_once(self.node, timeout_sec=0.05)
            if predicate():
                return
        raise TimeoutError(
            f"Timed out waiting for {description}. Check that Gazebo is running, "
            "the bridge is active, and no topic remapping is required."
        )

    def _wait_for_initial_messages(self) -> None:
        self._wait_until(
            lambda: (
                self._raw_scan is not None
                and self._x is not None
                and self._y is not None
                and self._yaw is not None
            ),
            "initial LiDAR and odometry messages",
        )

    def _publish_velocity(self, x: float, y: float, turn: float) -> None:
        message = Twist()
        message.linear.x = float(x)
        message.linear.y = float(y)
        message.angular.z = float(turn)
        self._publisher.publish(message)

    def stop(self) -> None:
        for _ in range(3):
            self._publish_velocity(0.0, 0.0, 0.0)
            rclpy.spin_once(self.node, timeout_sec=0.02)

    def _current_observation(self) -> np.ndarray:
        if self._raw_scan is None or self._x is None or self._y is None or self._yaw is None:
            raise RuntimeError("LiDAR or odometry state is unavailable")
        return build_observation(
            self._raw_scan,
            x=self._x,
            y=self._y,
            yaw=self._yaw,
            goal_x=self.config.goal_x,
            goal_y=self.config.goal_y,
            previous_action=self._applied_action,
            ray_count=self.config.ray_count,
            min_range=self.config.lidar_min_range,
            max_range=self.config.lidar_max_range,
            distance_scale=self.config.goal_distance_scale,
        ).copy()

    def _wait_for_action_observations(
        self,
        starting_scan_sequence: int,
        starting_odom_sequence: int,
        velocity: tuple[float, float, float],
    ) -> None:
        deadline = time.monotonic() + self.config.message_timeout_seconds
        target_scan_sequence = starting_scan_sequence + self.config.scans_per_action
        next_publish = 0.0

        while time.monotonic() < deadline:
            now = time.monotonic()
            if now >= next_publish:
                self._publish_velocity(*velocity)
                next_publish = now + 0.05
            rclpy.spin_once(self.node, timeout_sec=0.02)
            if (
                self._scan_sequence >= target_scan_sequence
                and self._odom_sequence > starting_odom_sequence
            ):
                return
        raise TimeoutError(
            "Timed out waiting for fresh sensor data after an action. "
            "Check /scan, /odom, /clock, and the Gazebo play state."
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        del options
        self.stop()

        if self.config.randomize_obstacles:
            layout_seed = int(self.np_random.integers(0, 2**31 - 1))
            layout = self.gazebo.randomize_obstacles(
                layout_seed,
                self.config.layout,
            )
            active_obstacles = len(layout.active_positions)
        else:
            layout_seed = None
            active_obstacles = None

        # Reset the robot and wait until odometry confirms that it has
        # reached the starting pose.
        odom_before = self._odom_sequence

        self.gazebo.set_entity_pose(
            self.config.robot_name,
            self.config.start_x,
            self.config.start_y,
            self.config.start_z,
            self.config.start_yaw,
        )

        self._wait_until(
            lambda: (
                self._odom_sequence > odom_before
                and self._x is not None
                and self._y is not None
                and self._yaw is not None
                and abs(self._x - self.config.start_x) < 0.25
                and abs(self._y - self.config.start_y) < 0.25
            ),
            "post-reset odometry",
        )

        # Now require another LiDAR scan. This prevents reset() from
        # returning a scan captured at the previous episode's final pose.
        scan_after_pose = self._scan_sequence

        self._wait_until(
            lambda: self._scan_sequence > scan_after_pose,
            "post-reset LiDAR message",
        )

        self._previous_x = float(self._x)
        self._applied_action = np.zeros(3, dtype=np.float32)
        self._step_count = 0
        self._needs_reset = False

        observation = self._current_observation()

        return observation, {
            "x": float(self._x),
            "y": float(self._y),
            "layout_seed": layout_seed,
            "active_obstacles": active_obstacles,
        }

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Apply a learned-policy command through the safety-aware filter."""

        return self._step_action(action, apply_filter=True)

    def step_direct(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Apply an expert command exactly, matching ``ca1.sh`` execution."""

        return self._step_action(action, apply_filter=False)

    def _step_action(
        self,
        action: np.ndarray,
        *,
        apply_filter: bool,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._needs_reset:
            raise RuntimeError("Call reset() before step(), and after an episode ends")
        requested_action = np.asarray(action, dtype=np.float32).reshape(-1)
        if not self.action_space.contains(requested_action):
            raise ValueError(f"Invalid action: {action!r}")

        previous_action = self._applied_action.copy()
        if apply_filter:
            self._applied_action = smooth_normalized_action(
                requested_action,
                previous_action,
                self.config.action_smoothing_alpha,
            )
            action_mode = "safety_filter"
        else:
            self._applied_action = requested_action.copy()
            action_mode = "direct_expert"
        velocity = action_to_velocity(
            self._applied_action,
            self.config.max_x_velocity,
            self.config.max_y_velocity,
            self.config.max_turn_velocity,
        )
        scan_before = self._scan_sequence
        odom_before = self._odom_sequence
        self._wait_for_action_observations(scan_before, odom_before, velocity)

        if self._x is None or self._y is None or self._yaw is None or self._raw_scan is None:
            raise RuntimeError("Sensor state unexpectedly became unavailable")
        self._step_count += 1
        scan_metres = sample_scan_metres(
            self._raw_scan,
            ray_count=self.config.ray_count,
            min_range=self.config.lidar_min_range,
            max_range=self.config.lidar_max_range,
        )
        minimum_scan = float(np.min(scan_metres))
        transition = evaluate_transition(
            previous_x=self._previous_x,
            current_x=self._x,
            current_y=self._y,
            minimum_scan=minimum_scan,
            step_count=self._step_count,
            goal_x=self.config.goal_x,
            corridor_half_width=self.config.corridor_half_width,
            collision_distance=self.config.collision_distance,
            max_episode_steps=self.config.max_episode_steps,
            action_delta_squared=float(
                np.sum(np.square(self._applied_action - previous_action))
            ),
            turn_velocity=velocity[2],
            reward_config=self.config.reward,
        )
        self._previous_x = self._x
        self._needs_reset = transition.terminated or transition.truncated
        if self._needs_reset:
            self.stop()

        info = {
            "x": self._x,
            "y": self._y,
            "yaw": self._yaw,
            "requested_action": requested_action.copy(),
            "applied_action": self._applied_action.copy(),
            "action_mode": action_mode,
            "velocity_command": np.asarray(velocity, dtype=np.float32),
            "minimum_scan": minimum_scan,
            "step_count": self._step_count,
            "reason": transition.reason,
            "reward_terms": transition.terms,
        }
        return (
            self._current_observation(),
            transition.reward,
            transition.terminated,
            transition.truncated,
            info,
        )

    def close(self) -> None:
        if self._closed:
            return
        self.stop()
        self.node.destroy_node()
        if self._owns_rclpy and rclpy.ok():
            rclpy.shutdown()
        self._closed = True
