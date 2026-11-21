"""Start joint-state extraction and the policy skeleton. No motion is sent."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    config = str(Path(get_package_share_directory("ur7e_policy")) / "config" / "real.yaml")
    return LaunchDescription([
        DeclareLaunchArgument("model_path", default_value="", description="Path to the future PPO deployment checkpoint"),
        Node(
            package="ur7e_state", executable="joint_state_node",
            name="joint_state_node", parameters=[config], output="screen",
        ),
        Node(
            package="ur7e_policy", executable="policy_node",
            name="policy_node", parameters=[config, {
                "model_path": ParameterValue(LaunchConfiguration("model_path"), value_type=str),
            }], output="screen",
        ),
    ])
