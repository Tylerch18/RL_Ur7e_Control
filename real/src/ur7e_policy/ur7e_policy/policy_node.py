"""Receive arm state; reserve the interface for future PPO commands.

This skeleton does not load the model or publish motion commands.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray


class PolicyNode(Node):
    def __init__(self):
        super().__init__("policy_node")
        self.declare_parameter("state_topic", "/ur7e/arm_state")
        self.declare_parameter("command_topic", "/forward_velocity_controller/commands")
        self.declare_parameter("model_path", "")
        self.model_path = self.get_parameter("model_path").value
        self.latest_state = None
        self.subscription = self.create_subscription(
            JointState, self.get_parameter("state_topic").value,
            self.on_arm_state, qos_profile_sensor_data,
        )
        self.command_publisher = self.create_publisher(
            Float64MultiArray, self.get_parameter("command_topic").value, 1,
        )
        self.get_logger().info("Policy skeleton running. Model inference and motion commands are not implemented.")

    def on_arm_state(self, message):
        first_state = self.latest_state is None
        self.latest_state = message
        if first_state:
            self.get_logger().info("Arm state received. Positions and velocities are available to the policy.")

        # Next implementation:
        # 1. Build the same 31 observations using the reference and measured state.
        # 2. Run PPO at 50 Hz and hold its six joint-velocity corrections.
        # 3. Run the existing tracker at 500 Hz with limits and stale-data stopping.
        # 4. Publish the final joint velocities through self.command_publisher.


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = PolicyNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
