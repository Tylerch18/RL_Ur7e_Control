"""Select the arm joints from the UR driver's JointState messages."""

import math

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState

ARM_JOINTS = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
]


class JointStateNode(Node):
    def __init__(self):
        super().__init__("joint_state_node")
        self.declare_parameter("input_topic", "/joint_states")
        self.declare_parameter("output_topic", "/ur7e/arm_state")
        self.declare_parameter("joint_names", ARM_JOINTS)
        self.joint_names = list(self.get_parameter("joint_names").value)
        if len(self.joint_names) != 6 or len(set(self.joint_names)) != 6:
            raise ValueError("joint_names must contain six different arm joints")
        self.publisher = self.create_publisher(
            JointState, self.get_parameter("output_topic").value, qos_profile_sensor_data,
        )
        self.subscription = self.create_subscription(
            JointState, self.get_parameter("input_topic").value,
            self.on_joint_state, qos_profile_sensor_data,
        )
        self.reported_invalid = False
        self.get_logger().info("Waiting for joint positions and velocities from the UR driver.")

    def on_joint_state(self, message):
        indices = {name: i for i, name in enumerate(message.name)}
        try:
            positions = [message.position[indices[name]] for name in self.joint_names]
            velocities = [message.velocity[indices[name]] for name in self.joint_names]
        except (KeyError, IndexError):
            if not self.reported_invalid:
                self.get_logger().warning("JointState is missing configured joints or velocities.")
                self.reported_invalid = True
            return
        if not all(math.isfinite(value) for value in positions + velocities):
            return
        self.reported_invalid = False
        arm_state = JointState()
        arm_state.header = message.header
        arm_state.name = self.joint_names
        arm_state.position = positions
        arm_state.velocity = velocities
        self.publisher.publish(arm_state)


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = JointStateNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
