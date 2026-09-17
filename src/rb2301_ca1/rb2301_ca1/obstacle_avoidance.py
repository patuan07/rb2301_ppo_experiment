"""Run the 20 Hz, 90-degree-cone expert directly as a ROS 2 controller."""

from __future__ import annotations

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from rclpy.logging import LoggingSeverity, set_logger_level
from rclpy.node import Node
from sensor_msgs.msg import LaserScan

from .expert_policy import ConeExpertPolicy
from .rl_core import preprocess_scan


MAX_TRANSLATE_VELOCITY = 0.4
MAX_TURN_VELOCITY = 0.8
set_logger_level("obstacle_avoidance", level=LoggingSeverity.DEBUG)


class ObstacleAvoidanceNode(Node):
    def __init__(self) -> None:
        super().__init__("obstacle_avoidance")
        self.get_logger().info("Starting 20 Hz obstacle-avoidance expert")
        self.pub_cmd_vel = self.create_publisher(Twist, "cmd_vel", 10)
        self.sub_scan = self.create_subscription(
            LaserScan,
            "scan",
            self.sub_scan_callback,
            2,
        )
        self.last_scan: np.ndarray | None = None
        self.expert = ConeExpertPolicy()
        self.timer = self.create_timer(0.05, self.timer_callback)

    def move_2D(self, x: float = 0.0, y: float = 0.0, turn: float = 0.0) -> None:
        message = Twist()
        message.linear.x = float(
            np.clip(x, -MAX_TRANSLATE_VELOCITY, MAX_TRANSLATE_VELOCITY)
        )
        message.linear.y = float(
            np.clip(y, -MAX_TRANSLATE_VELOCITY, MAX_TRANSLATE_VELOCITY)
        )
        message.angular.z = float(
            np.clip(turn, -MAX_TURN_VELOCITY, MAX_TURN_VELOCITY)
        )
        self.pub_cmd_vel.publish(message)

    def sub_scan_callback(self, message: LaserScan) -> None:
        self.last_scan = preprocess_scan(
            message.ranges,
            ray_count=36,
            min_range=0.05,
            max_range=10.0,
        )

    def timer_callback(self) -> None:
        if self.last_scan is None:
            return
        observation = np.zeros(42, dtype=np.float32)
        observation[:36] = self.last_scan
        action = self.expert.predict(observation)
        self.move_2D(
            action[0] * MAX_TRANSLATE_VELOCITY,
            action[1] * MAX_TRANSLATE_VELOCITY,
            action[2] * MAX_TURN_VELOCITY,
        )
        self.get_logger().debug(
            f"heading={self.expert.heading_degrees} action={np.round(action, 2)}"
        )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ObstacleAvoidanceNode()
    try:
        rclpy.spin(node)
    finally:
        node.move_2D()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
