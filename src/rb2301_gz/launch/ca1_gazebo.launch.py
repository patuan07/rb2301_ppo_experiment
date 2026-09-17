"""Launch the RB2301 obstacle world, NanoCar, sensors, and ROS bridges."""

from __future__ import annotations

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    Command,
    LaunchConfiguration,
    PathJoinSubstitution,
    PythonExpression,
    TextSubstitution,
)
from launch_ros.actions import Node
from launch_ros.descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare

from rb2301_ca1.obstacle_generator import (
    generate_sdf_file,
    resolve_coke_model_uris,
)
from rb2301_ca1.maze_layout import layout_config_for_difficulty


def _prepare_world(context):
    package_directory = get_package_share_directory("rb2301_gz")
    world_path = LaunchConfiguration("world_path").perform(context)
    model_directory = os.path.join(package_directory, "meshes", "coke", "6")
    randomize = LaunchConfiguration("randomize_world").perform(context).lower() in {
        "1",
        "true",
        "yes",
    }
    if not randomize:
        resolve_coke_model_uris(world_path, model_directory)
        return []

    seed_text = LaunchConfiguration("world_seed").perform(context)
    seed = None if seed_text in {"", "-1", "none", "None"} else int(seed_text)
    maze_difficulty = LaunchConfiguration("maze_difficulty").perform(context)
    generate_sdf_file(
        world_path,
        model_directory,
        seed,
        layout_config_for_difficulty(maze_difficulty),
    )
    return []


def generate_launch_description():
    package_gz = FindPackageShare("rb2301_gz")
    package_ros_gz_sim = FindPackageShare("ros_gz_sim")

    model = LaunchConfiguration("model")
    world = LaunchConfiguration("world")
    world_path = LaunchConfiguration("world_path")
    headless = LaunchConfiguration("headless")
    x = LaunchConfiguration("x")
    y = LaunchConfiguration("y")
    yaw = LaunchConfiguration("yaw")

    model_path = PathJoinSubstitution([package_gz, "urdf", model])
    server_only_argument = PythonExpression(
        ["' -s' if '", headless, "'.lower() in ('1', 'true', 'yes') else ''"]
    )

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="screen",
        parameters=[
            {
                "robot_description": ParameterValue(
                    Command(["xacro ", model_path]),
                    value_type=str,
                ),
                "use_sim_time": True,
            }
        ],
    )

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [PathJoinSubstitution([package_ros_gz_sim, "launch", "gz_sim.launch.py"])]
        ),
        launch_arguments={
            "gz_args": [
                world_path,
                TextSubstitution(text=" -r -v 1"),
                server_only_argument,
            ]
        }.items(),
    )

    spawn_robot = Node(
        package="ros_gz_sim",
        executable="create",
        arguments=[
            "-name",
            "nanocar",
            "-topic",
            "robot_description",
            "-x",
            x,
            "-y",
            y,
            "-z",
            "0.03",
            "-Y",
            yaw,
        ],
        output="screen",
        parameters=[{"use_sim_time": True}],
    )

    bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=[
            "/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock",
            "/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist",
            "/odom@nav_msgs/msg/Odometry[gz.msgs.Odometry",
            "/joint_states@sensor_msgs/msg/JointState[gz.msgs.Model",
            "/tf@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V",
            "/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan",
            "/imu@sensor_msgs/msg/Imu[gz.msgs.IMU",
            "/world/empty/dynamic_pose/info@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V",
        ],
        output="screen",
        parameters=[{"use_sim_time": True}],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("model", default_value="nanocar_description.urdf"),
            DeclareLaunchArgument("world", default_value="obstacle_world_ca1.sdf"),
            DeclareLaunchArgument(
                "world_path",
                default_value=PathJoinSubstitution([package_gz, "worlds", world]),
                description="Absolute SDF path; managed RL workers use private copies",
            ),
            DeclareLaunchArgument("x", default_value="0.0"),
            DeclareLaunchArgument("y", default_value="0.0"),
            DeclareLaunchArgument("yaw", default_value="0.0"),
            DeclareLaunchArgument(
                "headless",
                default_value="false",
                description="Run the Gazebo server without its GUI",
            ),
            DeclareLaunchArgument(
                "randomize_world",
                default_value="true",
                description="Generate a seeded obstacle layout before launch",
            ),
            DeclareLaunchArgument(
                "world_seed",
                default_value="-1",
                description="Obstacle seed; -1 selects a fresh random seed",
            ),
            DeclareLaunchArgument(
                "maze_difficulty",
                default_value="medium",
                description="Connected maze generator level: easy, medium, or hard",
            ),
            OpaqueFunction(function=_prepare_world),
            robot_state_publisher,
            gazebo,
            spawn_robot,
            bridge,
        ]
    )
