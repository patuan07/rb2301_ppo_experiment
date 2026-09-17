"""Run a trained PPO policy on the real robot, with NumPy only.

The Gazebo-trained policy is a small MLP, so the robot never needs PyTorch or
Stable-Baselines3: :mod:`rb2301_ca1.policy_runtime` evaluates the exported
weights directly and :mod:`rb2301_ca1.policy_runtime.self_test` proves on the
robot itself that the NumPy inference matches the training machine.

The node assumes two other processes are already running and does not start
them::

    ros2 launch base_control_ros2 base_control.launch.py   # /cmd_vel -> wheels
    ros2 launch rplidar_ros rplidar.launch.py              # /scan

Bring it up with ``publish_commands:=false`` first.  Nothing here has seen real
sensor noise, wheel slip, or a collision.
"""

from __future__ import annotations

import argparse
import math
import signal
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from sensor_msgs.msg import Imu, LaserScan
from std_srvs.srv import Trigger

from .policy_runtime import (
    POLICY_RAY_COUNT,
    ActorMLP,
    self_test,
    sim_scan_from_ranges,
    wrap_to_pi,
)
from .rl_core import action_to_velocity, build_observation, smooth_normalized_action

#: Training values that the observation and reward were defined against.
TRAINING_GOAL_DISTANCE_SCALE = 8.0
TRAINING_LIDAR_MIN_RANGE = 0.05
TRAINING_LIDAR_MAX_RANGE = 10.0
TRAINING_CORRIDOR_HALF_WIDTH = 2.15
TRAINING_MAX_EPISODE_STEPS = 800
TRAINING_SMOOTHING_ALPHA = 0.15
TRAINING_SCAN_RATE = 20.0

POSE_SOURCES = ("auto", "odom", "dead_reckon")

#: Every subscription asks for BEST_EFFORT, which is strictly more compatible
#: than RELIABLE.  ROS 2 pairs a publisher and subscriber only when the
#: subscriber's requested reliability is no stronger than the publisher's, so a
#: BEST_EFFORT subscriber accepts both RELIABLE and BEST_EFFORT publishers while
#: a RELIABLE subscriber silently receives nothing from a BEST_EFFORT one --
#: which is exactly how ``rplidar_ros`` publishes.  Since none of these topics
#: are ours, the weaker request is the one that cannot fail this way.
_SENSOR_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
)

#: Every tunable, as ``(name, default)``.  Kept at module scope rather than on
#: the node: ``rclpy.node.Node`` stores its own parameter dictionary in
#: ``self._parameters``, so an identically named attribute silently breaks
#: ``declare_parameter``.
_PARAMETER_DEFAULTS: tuple[tuple[str, Any], ...] = (
    ("weights_path", ""),
    ("reference_path", ""),
    ("scan_topic", "/scan"),
    ("odom_topic", "/odom"),
    ("imu_topic", ""),
    ("cmd_vel_topic", "/cmd_vel"),
    ("pose_source", "auto"),
    ("odom_relative", True),
    ("odom_timeout", 0.5),
    ("goal_x", 7.2),
    ("goal_y", 0.0),
    ("publish_commands", True),
    ("allow_shared_cmd_vel", False),
    ("scan_timeout", 0.5),
    # The training collision distance is 0.12 m, but that is below the minimum
    # range of a real RPLidar (~0.15 m), so a threshold there could never fire
    # from a real reading.  Stop early instead of never.
    ("emergency_stop_distance", 0.25),
    ("control_rate", 20.0),
    ("action_smoothing_alpha", TRAINING_SMOOTHING_ALPHA),
    ("adapt_smoothing_to_rate", True),
    ("nominal_scan_rate", TRAINING_SCAN_RATE),
    ("max_decisions", TRAINING_MAX_EPISODE_STEPS),
    ("corridor_half_width", TRAINING_CORRIDOR_HALF_WIDTH),
    ("scan_yaw_offset", 0.0),
    ("max_x_velocity", 0.4),
    ("max_y_velocity", 0.4),
    ("max_turn_velocity", 0.8),
    ("status_period", 5.0),
)


def _scan_timestamp(message: LaserScan) -> float | None:
    """Return a message stamp in seconds, or ``None`` when the driver omits it."""

    stamp = message.header.stamp
    value = float(stamp.sec) + float(stamp.nanosec) * 1e-9
    return value if value > 0.0 else None


def _yaw_from_orientation(orientation: Any) -> float:
    """Extract planar yaw from a quaternion the way ``ros_gym_env`` does.

    Sharing the formula with the training environment keeps the odometry path
    numerically identical to the one the policy was evaluated against, rather
    than merely equivalent up to a different branch cut.
    """

    sin_yaw = 2.0 * (
        float(orientation.w) * float(orientation.z)
        + float(orientation.x) * float(orientation.y)
    )
    cos_yaw = 1.0 - 2.0 * (
        float(orientation.y) ** 2 + float(orientation.z) ** 2
    )
    return float(math.atan2(sin_yaw, cos_yaw))


@dataclass
class Pose:
    """Planar pose in the policy's own frame, which starts at ``(0, 0, 0)``."""

    x: float
    y: float
    yaw: float
    source: str


class DeployPolicy(Node):
    """Drive the robot with an exported policy, with interlocks."""

    def __init__(self, overrides: dict[str, Any] | None = None) -> None:
        super().__init__("rb2301_deploy_policy")
        self._read_parameters(overrides or {})

        self._actor = ActorMLP.load(self._weights_path)
        if self._actor.observation_dim != POLICY_RAY_COUNT + 6:
            raise ValueError(
                f"Exported actor expects {self._actor.observation_dim} observations; "
                f"this node builds {POLICY_RAY_COUNT + 6}"
            )

        # --- topics ---------------------------------------------------------
        self._publisher = self.create_publisher(Twist, self._cmd_vel_topic, 10)
        self._scan_subscription = self.create_subscription(
            LaserScan, self._scan_topic, self._scan_callback, _SENSOR_QOS
        )
        self._odom_subscription = self.create_subscription(
            Odometry, self._odom_topic, self._odom_callback, _SENSOR_QOS
        )
        self._imu_subscription = (
            self.create_subscription(Imu, self._imu_topic, self._imu_callback, _SENSOR_QOS)
            if self._imu_topic
            else None
        )
        self._clear_estop_service = self.create_service(
            Trigger, "~/clear_estop", self._clear_estop_callback
        )

        # --- sensor state ---------------------------------------------------
        self._scan_message: LaserScan | None = None
        self._scan_received_wall = 0.0
        self._scan_sequence = 0
        self._processed_sequence = 0
        self._last_stamp: float | None = None
        self._last_processed_wall: float | None = None
        self._period_ema: float | None = None
        self._rate_warned = False

        self._odom_pose: tuple[float, float, float] | None = None
        self._odom_origin: tuple[float, float, float] | None = None
        self._odom_received_wall = 0.0
        self._imu_yaw: float | None = None
        self._imu_origin: float | None = None
        self._imu_received_wall = 0.0

        # --- control state --------------------------------------------------
        self._previous_action = np.zeros(3, dtype=np.float32)
        self._held_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._last_applied_velocity: tuple[float, float, float] | None = None
        self._dead_reckon = Pose(0.0, 0.0, 0.0, "dead_reckon")
        self._using_dead_reckoning = self._pose_source != "odom"
        self._ever_used_odom = False
        self._resume_from_halt = True
        self._decisions = 0
        self._halt_count = 0
        self._last_halt_reason: str | None = None
        self._announced_first_scan = False
        self._stop_reason: str | None = None
        self._estop_latched = False
        self._last_pose: Pose | None = None
        self._shutdown = False
        self._started_wall = time.monotonic()
        self._publisher_conflict: list[str] = []
        self._publisher_checked = False

        self._alpha = float(self._action_smoothing_alpha)
        self.get_logger().info(
            f"Loaded {self._actor.observation_dim}->{self._actor.action_dim} actor from "
            f"{self._weights_path} (activation={self._actor.activation})"
        )
        self.get_logger().info(
            f"scan={self._scan_topic} odom={self._odom_topic} "
            f"imu={self._imu_topic or 'disabled'} cmd_vel={self._cmd_vel_topic} "
            f"pose_source={self._pose_source}"
        )
        if not self._publish_commands:
            self.get_logger().warn(
                "publish_commands:=false -- the robot will not move.  The policy "
                "still runs and its actions are logged."
            )
            if self._pose_source != "odom":
                self.get_logger().warn(
                    "Dry run with a dead-reckoning pose source: commanded velocity "
                    "is integrated only when it is actually published, so the pose "
                    "and every logged observation are frozen fiction.  Use "
                    "pose_source:=odom for a meaningful dry run."
                )
        if self._pose_source == "dead_reckon":
            self.get_logger().warn(
                "Dead reckoning integrates commanded velocity, which assumes the "
                "base tracks its command exactly -- the same assumption Gazebo "
                "makes.  Expect the goal bearing to drift; see the deployment notes."
            )

        self._timer = self.create_timer(1.0 / self._control_rate, self._control_loop)

    # ------------------------------------------------------------------ setup

    def _read_parameters(self, overrides: dict[str, Any]) -> None:
        """Declare every tunable, then apply the few command-line overrides.

        ROS parameters are the real interface -- ``ros2 run rb2301_ca1
        deploy_policy --ros-args -p pose_source:=dead_reckon``.  The overrides
        exist only for values that must be known before ROS can be configured,
        which is why the list is short.
        """

        for name, default in _PARAMETER_DEFAULTS:
            self.declare_parameter(name, default)

        if overrides:
            unknown = sorted(set(overrides) - {name for name, _ in _PARAMETER_DEFAULTS})
            if unknown:
                raise ValueError(f"Unknown parameter overrides: {unknown}")
            self.set_parameters(
                [Parameter(name, value=value) for name, value in overrides.items()]
            )

        def read(name: str) -> Any:
            return self.get_parameter(name).value

        self._weights_path, self._reference_path = _resolve_policy_paths(
            str(read("weights_path")), str(read("reference_path"))
        )
        self._scan_topic = str(read("scan_topic"))
        self._odom_topic = str(read("odom_topic"))
        self._imu_topic = str(read("imu_topic"))
        self._cmd_vel_topic = str(read("cmd_vel_topic"))
        self._pose_source = str(read("pose_source"))
        self._odom_relative = bool(read("odom_relative"))
        self._odom_timeout = float(read("odom_timeout"))
        self._goal_x = float(read("goal_x"))
        self._goal_y = float(read("goal_y"))
        self._publish_commands = bool(read("publish_commands"))
        self._allow_shared_cmd_vel = bool(read("allow_shared_cmd_vel"))
        self._scan_timeout = float(read("scan_timeout"))
        self._emergency_stop_distance = float(read("emergency_stop_distance"))
        self._control_rate = float(read("control_rate"))
        self._action_smoothing_alpha = float(read("action_smoothing_alpha"))
        self._adapt_smoothing_to_rate = bool(read("adapt_smoothing_to_rate"))
        self._nominal_scan_rate = float(read("nominal_scan_rate"))
        self._max_decisions = int(read("max_decisions"))
        self._corridor_half_width = float(read("corridor_half_width"))
        self._scan_yaw_offset = float(read("scan_yaw_offset"))
        self._max_x_velocity = float(read("max_x_velocity"))
        self._max_y_velocity = float(read("max_y_velocity"))
        self._max_turn_velocity = float(read("max_turn_velocity"))
        self._status_period = float(read("status_period"))

        self._validate_parameters()

    def _validate_parameters(self) -> None:
        """Reject configurations whose failure would be silent in the field."""

        problems: list[str] = []
        if self._pose_source not in POSE_SOURCES:
            problems.append(
                f"pose_source is {self._pose_source!r}; expected one of {list(POSE_SOURCES)}"
            )
        if not 0.0 < self._action_smoothing_alpha <= 1.0:
            problems.append("action_smoothing_alpha must lie in (0, 1]")
        if self._control_rate <= 0.0:
            problems.append("control_rate must be positive")
        if self._scan_timeout <= 0.0 or self._odom_timeout <= 0.0:
            problems.append("scan_timeout and odom_timeout must be positive")
        if self._emergency_stop_distance < 0.0:
            problems.append("emergency_stop_distance cannot be negative")
        if self._max_decisions < 1:
            problems.append("max_decisions must be positive")
        if self._nominal_scan_rate <= 0.0:
            problems.append("nominal_scan_rate must be positive")
        if min(self._max_x_velocity, self._max_y_velocity, self._max_turn_velocity) <= 0.0:
            problems.append("velocity limits must be positive")
        if self._pose_source == "odom" and not self._publish_commands:
            self.get_logger().warn(
                "pose_source:=odom with publish_commands:=false is the informative "
                "dry run: obstacles and goals are real, only the wheels are idle."
            )
        if problems:
            raise ValueError(
                "Refusing to start with an unusable configuration:\n  - "
                + "\n  - ".join(problems)
            )

    # ------------------------------------------------------------- callbacks

    def _scan_callback(self, message: LaserScan) -> None:
        self._scan_message = message
        self._scan_received_wall = time.monotonic()
        self._scan_sequence += 1

    def _odom_callback(self, message: Odometry) -> None:
        position = message.pose.pose.position
        pose = (
            float(position.x),
            float(position.y),
            _yaw_from_orientation(message.pose.pose.orientation),
        )
        if self._odom_origin is None:
            self._odom_origin = pose
        self._odom_pose = pose
        self._odom_received_wall = time.monotonic()

    def _imu_callback(self, message: Imu) -> None:
        yaw = _yaw_from_orientation(message.orientation)
        if self._imu_origin is None:
            self._imu_origin = yaw
        self._imu_yaw = yaw
        self._imu_received_wall = time.monotonic()

    def _clear_estop_callback(
        self, request: Trigger.Request, response: Trigger.Response
    ) -> Trigger.Response:
        """Clear a latched stop, but only when it is currently safe to do so."""

        if self._estop_latched and not self._scan_is_fresh(time.monotonic()):
            response.success = False
            response.message = "refusing to clear: no fresh scan"
            return response
        if self._estop_latched:
            conversion = self._convert_scan(self._scan_message)
            if conversion is not None and (
                conversion.minimum_valid_range <= self._emergency_stop_distance
            ):
                response.success = False
                response.message = (
                    f"refusing to clear: obstacle at "
                    f"{conversion.minimum_valid_range:.3f} m"
                )
                return response
        self._estop_latched = False
        self._stop_reason = None
        self._resume_from_halt = True
        self.get_logger().warn("Emergency stop cleared by operator request")
        response.success = True
        response.message = "cleared"
        return response

    # ---------------------------------------------------------------- helpers

    def _scan_is_fresh(self, now_wall: float) -> bool:
        return (
            self._scan_message is not None
            and (now_wall - self._scan_received_wall) <= self._scan_timeout
        )

    def _convert_scan(self, message: LaserScan | None) -> Any:
        """Map a LaserScan onto the simulated sensor's ray bearings."""

        if message is None:
            return None
        return sim_scan_from_ranges(
            message.ranges,
            angle_min=float(message.angle_min),
            angle_increment=float(message.angle_increment),
            range_min=float(message.range_min),
            range_max=float(message.range_max),
            mount_yaw=self._scan_yaw_offset,
            max_range=TRAINING_LIDAR_MAX_RANGE,
        )

    def _publish_velocity(self, x: float, y: float, turn: float) -> None:
        if not self._publish_commands or self._shutdown:
            return
        message = Twist()
        message.linear.x = float(x)
        message.linear.y = float(y)
        message.angular.z = float(turn)
        self._publisher.publish(message)

    def _halt(self, reason: str, *, startup: bool = False) -> None:
        """Stop and forget the motion history, so a resume starts from rest.

        ``startup`` marks the benign "nothing published yet" case, which is
        announced once at info level rather than repeated as a warning.  The two
        severities cannot share a throttled call: rclpy raises "Logger severity
        cannot be changed between calls", so the startup notice is emitted
        unthrottled behind a flag instead.
        """

        if self._last_halt_reason != reason:
            self._halt_count += 1
            self._last_halt_reason = reason
        if startup:
            if not self._announced_first_scan:
                self._announced_first_scan = True
                self.get_logger().info(f"Holding still: {reason}")
        else:
            self.get_logger().warn(f"Holding still: {reason}", throttle_duration_sec=2.0)
        self._held_velocity = (0.0, 0.0, 0.0)
        self._last_applied_velocity = None
        self._resume_from_halt = True
        self._publish_velocity(0.0, 0.0, 0.0)

    def _latch(self, reason: str) -> None:
        """Stop and stay stopped until an operator clears it."""

        self._estop_latched = True
        self._stop_reason = reason
        self._held_velocity = (0.0, 0.0, 0.0)
        self._last_applied_velocity = None
        self._resume_from_halt = True
        self._publish_velocity(0.0, 0.0, 0.0)
        self.get_logger().error(
            f"LATCHED STOP: {reason}.  Clear with "
            "`ros2 service call /rb2301_deploy_policy/clear_estop std_srvs/srv/Trigger` "
            "once the path is clear."
        )

    # ----------------------------------------------------------------- timing

    def _measure_period(self, stamp: float | None, now_wall: float) -> float:
        """Return seconds since the previous decision, preferring header stamps.

        Dead reckoning scales its whole trajectory by this number, so an assumed
        1/20 s would bake the scan-period error straight into the pose.  Drivers
        that leave ``header.stamp`` at zero are handled by falling back to the
        node clock rather than by integrating nothing.
        """

        period: float | None = None
        if stamp is not None and self._last_stamp is not None:
            candidate = stamp - self._last_stamp
            if 0.0 < candidate <= 1.0:
                period = candidate
        if period is None and self._last_processed_wall is not None:
            candidate = now_wall - self._last_processed_wall
            if candidate > 0.0:
                period = candidate

        self._last_stamp = stamp
        self._last_processed_wall = now_wall
        if period is None:
            return 0.0

        period = min(period, 0.5)
        if self._period_ema is None:
            self._period_ema = period
        else:
            self._period_ema = 0.9 * self._period_ema + 0.1 * period
        self._update_smoothing()
        return period

    def _update_smoothing(self) -> None:
        """Match the filter's per-second response to the training scan rate.

        ``smooth_normalized_action`` applies ``alpha`` once per *call*, so at a
        scan rate below the training 20 Hz every ramp takes proportionally longer
        in wall-clock time and the robot reacts later than the policy expects.
        Solving ``(1 - a_eff)^f == (1 - a)^20`` scales the response back to the
        rate the policy was trained against.  The observation still carries the
        previous applied action, so this narrows the mismatch rather than
        removing it.
        """

        if not self._adapt_smoothing_to_rate or not self._period_ema:
            self._alpha = float(self._action_smoothing_alpha)
            return
        measured_rate = 1.0 / self._period_ema
        self._alpha = float(
            1.0 - (1.0 - self._action_smoothing_alpha) ** (self._nominal_scan_rate / measured_rate)
        )

    # ------------------------------------------------------------------- pose

    def _odom_estimate(self) -> tuple[float, float, float] | None:
        if self._odom_pose is None:
            return None
        x, y, yaw = self._odom_pose
        if self._odom_relative and self._odom_origin is not None:
            origin_x, origin_y, origin_yaw = self._odom_origin
            return (x - origin_x, y - origin_y, float(wrap_to_pi(yaw - origin_yaw)))
        return (x, y, yaw)

    def _advance_dead_reckoning(self, period: float) -> None:
        """Integrate the last published body velocity over one control period.

        A dry run publishes nothing, so integrating the command would invent a
        trajectory the robot never travelled.  The pose is frozen instead, and
        ``__init__`` says so.
        """

        if not self._publish_commands or period <= 0.0:
            return
        if self._last_applied_velocity is None:
            return
        vx, vy, turn = self._last_applied_velocity
        yaw = self._dead_reckon.yaw
        cosine = math.cos(yaw)
        sine = math.sin(yaw)
        self._dead_reckon.x += (vx * cosine - vy * sine) * period
        self._dead_reckon.y += (vx * sine + vy * cosine) * period
        self._dead_reckon.yaw = float(wrap_to_pi(yaw + turn * period))
        # A fresh IMU heading rescues dead reckoning exactly where it is weakest.
        if (
            self._imu_yaw is not None
            and (time.monotonic() - self._imu_received_wall) <= self._odom_timeout
            and self._imu_origin is not None
        ):
            self._dead_reckon.yaw = float(
                wrap_to_pi(self._imu_yaw - self._imu_origin)
            )

    def _pose(self, now_wall: float, period: float) -> Pose | None:
        odom = self._odom_estimate()
        odom_fresh = (
            odom is not None
            and (now_wall - self._odom_received_wall) <= self._odom_timeout
        )

        if self._pose_source == "odom":
            return None if not odom_fresh else Pose(*odom, "odom")

        if self._pose_source == "dead_reckon":
            self._advance_dead_reckoning(period)
            return self._dead_reckon

        if odom_fresh:
            # Only a genuine recovery is worth announcing; "auto" starts out
            # marked as integrating, so the first odom message would otherwise
            # claim to have recovered from something that never happened.
            if self._using_dead_reckoning and self._ever_used_odom:
                self.get_logger().info(
                    "Odometry is fresh again; re-anchoring the pose to it"
                )
            self._using_dead_reckoning = False
            self._ever_used_odom = True
            # Re-anchor continuously, so a later fallback resumes from the last
            # trustworthy pose instead of from wherever integration drifted to.
            self._dead_reckon = Pose(*odom, "odom")
            return self._dead_reckon
        if not self._using_dead_reckoning:
            self.get_logger().warn(
                f"{self._odom_topic} has been silent for more than "
                f"{self._odom_timeout:.1f}s; falling back to dead reckoning"
            )
            self._using_dead_reckoning = True
        self._advance_dead_reckoning(period)
        return Pose(
            self._dead_reckon.x, self._dead_reckon.y, self._dead_reckon.yaw, "dead_reckon"
        )

    # ------------------------------------------------------------ publisher IQ

    def _check_publisher_ownership(self) -> None:
        """Refuse to arm if something else already commands the base.

        Two nodes writing ``/cmd_vel`` fight over the robot, and the loser is
        whichever one is slower.  ``get_publishers_info_by_topic`` is used rather
        than ``count_publishers`` because the count includes this node's own
        publisher, which would make ``> 1`` ambiguous.
        """

        self._publisher_checked = True
        try:
            infos = self.get_publishers_info_by_topic(self._cmd_vel_topic)
        except Exception as error:  # pragma: no cover - graph API availability
            self.get_logger().warn(
                f"Could not inspect publishers on {self._cmd_vel_topic}: {error}"
            )
            return
        others = [
            f"{info.node_namespace.rstrip('/')}/{info.node_name}"
            for info in infos
            if info.node_name != self.get_name()
        ]
        if not others:
            return
        self._publisher_conflict = others
        if not self._publish_commands:
            # A passive dry run moves nothing, so a competing publisher is not
            # our problem -- but it is worth knowing about before arming.
            self.get_logger().info(
                f"{self._cmd_vel_topic} is also published by {others}; harmless "
                "while publish_commands:=false"
            )
        elif not self._allow_shared_cmd_vel:
            self._latch(
                f"{self._cmd_vel_topic} is already published by {others}; "
                "set allow_shared_cmd_vel:=true only if that is intentional"
            )
        else:
            self.get_logger().warn(
                f"Sharing {self._cmd_vel_topic} with {others} as requested"
            )

    # ------------------------------------------------------------- main loop

    def _control_loop(self) -> None:
        now_wall = time.monotonic()

        if not self._publisher_checked and (now_wall - self._started_wall) >= 2.0:
            self._check_publisher_ownership()

        if self._stop_reason is not None:
            self._publish_velocity(0.0, 0.0, 0.0)
            return

        if self._scan_message is None:
            # Not a fault yet, just a node that has not been given anything to
            # act on.  It becomes a fault at the next branch below.
            self._halt(
                f"waiting for the first message on {self._scan_topic}", startup=True
            )
            return
        if not self._scan_is_fresh(now_wall):
            self._halt(
                f"no {self._scan_topic} message for more than "
                f"{self._scan_timeout:.1f}s"
            )
            return

        if self._scan_sequence == self._processed_sequence:
            # Training takes one decision per fresh scan, so a free-running timer
            # must not advance the policy between measurements.  It still
            # republishes, because base controllers commonly stop the robot when
            # commands lapse.
            self._publish_velocity(*self._held_velocity)
            return

        self._processed_sequence = self._scan_sequence
        self._decide(now_wall)

        if (now_wall - self._started_wall) % self._status_period < 1.0 / self._control_rate:
            self._log_status()

    def _decide(self, now_wall: float) -> None:
        message = self._scan_message
        assert message is not None

        period = self._measure_period(_scan_timestamp(message), now_wall)

        conversion = self._convert_scan(message)
        if conversion is None:
            return

        # The emergency stop reads every beam of the raw scan, not the 36-ray
        # subsample, which can step straight over the nearest obstacle.
        if conversion.minimum_valid_range <= self._emergency_stop_distance:
            self._latch(
                f"obstacle at {conversion.minimum_valid_range:.3f} m "
                f"(threshold {self._emergency_stop_distance:.2f} m)"
            )
            return

        if not conversion.is_healthy:
            self._halt(
                f"{conversion.invalid_fraction:.0%} of LiDAR returns measured "
                f"nothing at all ({conversion.out_of_range_fraction:.0%} merely out "
                "of range); refusing to drive on a sensor that is not returning data"
            )
            return

        if self._resume_from_halt:
            # The robot was stationary, and training's reset() also starts from a
            # zero previous action, so the filter restarts from rest rather than
            # from whatever was commanded before the pause.
            self._previous_action = np.zeros(3, dtype=np.float32)
            self._resume_from_halt = False

        pose = self._pose(now_wall, period)
        if pose is None:
            self._halt(f"{self._odom_topic} is silent and pose_source:=odom")
            return
        self._last_pose = pose

        if abs(pose.y) >= self._corridor_half_width:
            self._latch(
                f"lateral position {pose.y:+.2f} m left the "
                f"+/-{self._corridor_half_width:.2f} m corridor"
            )
            return
        if pose.x >= self._goal_x:
            self._latch(
                f"goal reached: x={pose.x:.2f} m >= {self._goal_x:.2f} m "
                f"(pose source: {pose.source})"
            )
            return

        observation = build_observation(
            conversion.ranges,
            x=pose.x,
            y=pose.y,
            yaw=pose.yaw,
            goal_x=self._goal_x,
            goal_y=self._goal_y,
            previous_action=self._previous_action,
            ray_count=POLICY_RAY_COUNT,
            min_range=TRAINING_LIDAR_MIN_RANGE,
            max_range=TRAINING_LIDAR_MAX_RANGE,
            distance_scale=TRAINING_GOAL_DISTANCE_SCALE,
        )
        target = self._actor.forward(observation)
        applied = smooth_normalized_action(target, self._previous_action, self._alpha)
        self._previous_action = np.asarray(applied, dtype=np.float32)
        velocity = action_to_velocity(
            applied,
            self._max_x_velocity,
            self._max_y_velocity,
            self._max_turn_velocity,
        )
        self._held_velocity = velocity
        self._last_applied_velocity = velocity
        self._decisions += 1
        self._announced_first_scan = False
        self._publish_velocity(*velocity)

        if self._decisions >= self._max_decisions:
            self._latch(
                f"episode budget exhausted after {self._decisions} decisions "
                "(training truncates at "
                f"{TRAINING_MAX_EPISODE_STEPS}); raise max_decisions to continue"
            )

    def _log_status(self) -> None:
        rate = "?" if not self._period_ema else f"{1.0 / self._period_ema:.1f}Hz"
        pose = self._last_pose
        pose_text = (
            "unavailable"
            if pose is None
            else f"({pose.x:+.2f},{pose.y:+.2f},{math.degrees(pose.yaw):+.0f}deg)/{pose.source}"
        )
        vx, vy, turn = self._held_velocity
        self.get_logger().info(
            f"decisions={self._decisions} halts={self._halt_count} "
            f"scan={rate} alpha={self._alpha:.3f} pose={pose_text} "
            f"cmd=({vx:+.2f},{vy:+.2f},{turn:+.2f})"
        )
        if self._period_ema and not self._rate_warned:
            measured = 1.0 / self._period_ema
            if measured < 15.0:
                self._rate_warned = True
                self.get_logger().warn(
                    f"LiDAR is running at {measured:.1f} Hz, below the {TRAINING_SCAN_RATE:.0f} Hz "
                    "the policy was trained at.  Decisions are correspondingly less "
                    "frequent; the smoothing filter has been rescaled to compensate, "
                    "but reaction latency is fundamentally limited by this rate."
                )

    # -------------------------------------------------------------- lifecycle

    def request_shutdown(self) -> None:
        """Stop the wheels before the node tears down."""

        self._shutdown = True
        if self._timer is not None:
            self._timer.cancel()
        for _ in range(3):
            message = Twist()
            self._publisher.publish(message)
            time.sleep(0.02)


#: Set by the SIGINT/SIGTERM handler and polled by the spin loop.  Raising an
#: exception from the handler instead would unwind out of rclpy's C extension
#: mid-call, which surfaces as a pybind11 conversion error rather than a clean
#: stop, and would tear the context down before the wheels could be stopped.
_STOP_REQUESTED = False


def _install_interrupt_handler() -> None:
    """Stop on Ctrl-C or SIGTERM without letting rclpy kill the context first.

    ``rclpy.init`` installs a handler that shuts the context down immediately.
    The wheels then cannot be commanded to stop, because every publish fails
    with "publisher's context is invalid" -- precisely the one message that has
    to get out.  This handler only records the request; :func:`main` notices and
    performs an orderly stop while the context is still usable.
    """

    def handler(signum: int, frame: Any) -> None:
        global _STOP_REQUESTED
        _STOP_REQUESTED = True

    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)


def _resolve_policy_paths(weights: str, reference: str) -> tuple[Path, Path]:
    """Locate the exported actor, defaulting to the installed package share."""

    if weights:
        weights_path = Path(weights).expanduser()
    else:
        try:
            from ament_index_python.packages import get_package_share_directory
        except ImportError as error:  # pragma: no cover - ROS is present on the robot
            raise RuntimeError(
                "ament_index_python is unavailable, so the default weights path "
                "cannot be resolved; pass --weights-path explicitly"
            ) from error
        weights_path = (
            Path(get_package_share_directory("rb2301_ca1")) / "policy" / "actor.npz"
        )
    reference_path = (
        Path(reference).expanduser()
        if reference
        else weights_path.with_name("actor_reference.npz")
    )
    return weights_path, reference_path


def _run_self_test(weights: Path, reference: Path) -> int:
    """Prove NumPy inference matches the training machine, without torch."""

    print(f"actor:     {weights}")
    print(f"reference: {reference}")
    actor = ActorMLP.load(weights)
    result = self_test(actor, reference)
    print(f"  layers={[w.shape for w in actor.weights]} activation={actor.activation}")
    print(
        f"  {result.rows} rows: float64 max|err|={result.float64_error:.3e} "
        f"(tol {result.float64_tolerance:.1e})"
    )
    print(
        f"           float32 max|err|={result.float32_error:.3e} "
        f"(tol {result.float32_tolerance:.1e})"
    )
    if result.passed:
        print("PASS: this machine's inference reproduces the checkpoint.")
        return 0
    print(
        "FAIL: exported weights or NumPy inference disagree with the training "
        f"machine (worst row {result.worst_float32_row}).  Do not run the policy."
    )
    return 1


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Deploy the exported PPO policy on the real robot.  Every setting "
            "other than the three below is a ROS parameter, e.g. "
            "--ros-args -p pose_source:=dead_reckon -p goal_x:=3.0"
        )
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Verify NumPy inference against the exported reference, then exit",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Shorthand for -p publish_commands:=false: run the policy, move nothing",
    )
    parser.add_argument(
        "--weights-path",
        default="",
        help="Exported actor.npz; defaults to the installed package share",
    )
    parser.add_argument(
        "--reference-path",
        default="",
        help="Exported actor_reference.npz; defaults to a sibling of the actor",
    )
    args, ros_arguments = parser.parse_known_args(argv)

    overrides: dict[str, Any] = {}
    if args.weights_path:
        overrides["weights_path"] = args.weights_path
    if args.reference_path:
        overrides["reference_path"] = args.reference_path
    if args.dry_run:
        overrides["publish_commands"] = False

    if args.self_test:
        weights, reference = _resolve_policy_paths(
            args.weights_path, args.reference_path
        )
        raise SystemExit(_run_self_test(weights, reference))

    rclpy.init(args=ros_arguments)
    _install_interrupt_handler()
    node: DeployPolicy | None = None
    try:
        node = DeployPolicy(overrides)
        # spin_once rather than spin(), so the stop flag is noticed between
        # callbacks instead of by unwinding through one.
        while rclpy.ok() and not _STOP_REQUESTED:
            rclpy.spin_once(node, timeout_sec=0.1)
        print("Stopping the robot.")
    except KeyboardInterrupt:  # pragma: no cover - fallback if the handler is replaced
        print("Interrupted; stopping the robot.")
    finally:
        if node is not None:
            try:
                node.request_shutdown()
            except Exception as error:  # pragma: no cover - keep teardown clean
                print(f"WARNING: could not publish the final stop: {error}")
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
