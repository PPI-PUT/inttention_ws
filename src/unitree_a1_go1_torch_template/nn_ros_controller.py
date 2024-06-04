import numpy as np
from scipy.spatial.transform import Rotation as R
import torch

from policy import get_policy

import rclpy
from rclpy.node import Node
from unitree_a1_legged_msgs.msg import LowState, LowCmd
from sensor_msgs.msg import Imu, Joy
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from geometry_msgs.msg import TwistStamped


class RobotHandler(Node):
    def __init__(self):
        super().__init__("robot_handler")

        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            depth=10
        )
        self.state_sub = self.create_subscription(
            LowState,
            "/unitree_a1_legged/joint_states",
            self.data_callback,
            qos_profile
        )
        self.imu_sub = self.create_subscription(
            Imu,
            "/unitree_a1_legged/sensors/imu/data",
            self.imu_callback,
            qos_profile
        )
        self.velocity_command_sub = self.create_subscription(
            TwistStamped,
            "/unitree_a1_legged/sensors/joy/cmd_vel",
            self.velocity_command_callback,
            qos_profile
        )
        # "/unitree_a1_legged/controller/nn/cmd"
        self.publisher = self.create_publisher(
            LowCmd, "/unitree_a1_legged/controller/nn/cmd", qos_profile)
        timer_period = 0.02  # 50 Hz
        self.timer = self.create_timer(timer_period, self.timer_callback)

        self.joint_positions = np.zeros(12)
        self.joint_velocities = np.zeros(12)
        self.orientation = np.array([0.0, 0.0, 0.0, 1.0])
        self.angular_velocity = np.zeros(3)
        self.x_goal_velocity = 0.0
        self.y_goal_velocity = 0.0
        self.yaw_goal_velocity = 0.0

        self.kp = 50.0
        self.kd = 0.5
        self.scaling_factor = 0.25
        # self.nominal_joint_positions = np.array([
        #     -0.1, -0.8, 1.5,
        #     0.1, 0.8, -1.5,
        #     0.1, -1.0, 1.5,
        #     -0.1, 1.0, -1.5
        # ])
        self.nominal_joint_positions = np.array([
            -0.1, 0.8, -1.5,
            0.1, 0.8, -1.5,
            -0.1, 1.0, -1.5,
            0.1, 1.0, -1.5
        ])

        self.previous_action = np.zeros(12)

        # For go1 change to "go1_dynamic_joint_description.npy"
        dynamic_joint_description_path = "a1_dynamic_joint_description.npy"
        # For go1 change to "go1_dynamic_foot_description.npy"
        dynamic_foot_description_path = "a1_dynamic_foot_description.npy"
        # For go1 change to "go1_general_policy_state.npy"
        general_policy_state_second_part_path = "a1_general_policy_state.npy"
        self.dynamic_joint_description = torch.tensor(
            np.load(dynamic_joint_description_path), dtype=torch.float32)
        self.dynamic_foot_description = torch.tensor(
            np.load(dynamic_foot_description_path), dtype=torch.float32)
        self.general_policy_state_second_part = np.load(
            general_policy_state_second_part_path)

        self.policy = get_policy()

        self.nn_active = False

        print(f"Robot ready. Using device: CPU")

    def data_callback(self, msg):
        self.joint_positions = np.array([
            msg.motor_state.front_right.hip.q,
            msg.motor_state.front_right.thigh.q,
            msg.motor_state.front_right.calf.q,
            msg.motor_state.front_left.hip.q,
            msg.motor_state.front_left.thigh.q,
            msg.motor_state.front_left.calf.q,
            msg.motor_state.rear_right.hip.q,
            msg.motor_state.rear_right.thigh.q,
            msg.motor_state.rear_right.calf.q,
            msg.motor_state.rear_left.hip.q,
            msg.motor_state.rear_left.thigh.q,
            msg.motor_state.rear_left.calf.q,
        ])
        self.joint_velocities = np.array([
            msg.motor_state.front_right.hip.q,
            msg.motor_state.front_right.thigh.q,
            msg.motor_state.front_right.calf.q,
            msg.motor_state.front_left.hip.q,
            msg.motor_state.front_left.thigh.q,
            msg.motor_state.front_left.calf.q,
            msg.motor_state.rear_right.hip.q,
            msg.motor_state.rear_right.thigh.q,
            msg.motor_state.rear_right.calf.q,
            msg.motor_state.rear_left.hip.q,
            msg.motor_state.rear_left.thigh.q,
            msg.motor_state.rear_left.calf.q,
        ])

    def imu_callback(self, msg):
        self.orientation = np.array(
            [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w])
        self.angular_velocity = np.array(
            [msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z])
        # print(self.orientation)

    def velocity_command_callback(self, msg):
        self.x_goal_velocity = msg.twist.linear.x * 2.0
        self.y_goal_velocity = msg.twist.linear.y * 2.0
        self.yaw_goal_velocity = msg.twist.angular.z * 2.0
        print(self.x_goal_velocity, self.y_goal_velocity, self.yaw_goal_velocity)

    def joystick_callback(self, msg):
        # TODO: Implement turning on and off nn with joystick
        # if ...:
        # self.nn_active = True
        # else:
        # self.nn_active = False

        pass

    def timer_callback(self):

        transposed_trunk_rotation_matrix = R.from_quat(
            self.orientation).as_matrix().T
        qpos = self.joint_positions - self.nominal_joint_positions
        qvel = self.joint_velocities
        ang_vel = self.angular_velocity
        projected_gravity_vector = np.matmul(
            transposed_trunk_rotation_matrix, np.array([0.0, 0.0, -1.0]))

        dynamic_joint_state = np.concatenate([
            qpos.reshape(1, 12, 1) / 4.6,
            qvel.reshape(1, 12, 1) / 35.0,
            self.previous_action.reshape(1, 12, 1) / 10.0
        ], axis=2)

        dynamic_foot_state = np.zeros((1, 4, 2))
        dynamic_foot_state[:, :, 0] = (dynamic_foot_state[:, :, 0] / 0.5) - 1.0
        dynamic_foot_state[:, :, 1] = np.clip(
            (dynamic_foot_state[:, :, 1] / (5.0 / 2)) - 1.0, -1.0, 1.0)

        general_policy_state = np.concatenate([
            [np.clip(ang_vel / 50.0, -1.0, 1.0)],
            [[self.x_goal_velocity, self.y_goal_velocity, self.yaw_goal_velocity]],
            [projected_gravity_vector],
            self.general_policy_state_second_part
        ], axis=1)
        # print(self.x_goal_velocity, self.y_goal_velocity, self.yaw_goal_velocity)
        with torch.no_grad():
            dynamic_joint_state = torch.tensor(
                dynamic_joint_state, dtype=torch.float32)
            dynamic_foot_state = torch.tensor(
                dynamic_foot_state, dtype=torch.float32)
            general_policy_state = torch.tensor(
                general_policy_state, dtype=torch.float32)
            action = self.policy(self.dynamic_joint_description, dynamic_joint_state,
                                 self.dynamic_foot_description, dynamic_foot_state, general_policy_state)
        action = action.numpy()[0]

        target_joint_positions = self.nominal_joint_positions + self.scaling_factor * action
        msg = LowCmd()
        msg.common.mode = 0x0A
        msg.common.kp = self.kp
        msg.common.kd = self.kd
        msg.motor_cmd.front_right.hip.q = target_joint_positions[0]
        msg.motor_cmd.front_right.thigh.q = target_joint_positions[1]
        msg.motor_cmd.front_right.calf.q = target_joint_positions[2]
        msg.motor_cmd.front_left.hip.q = target_joint_positions[3]
        msg.motor_cmd.front_left.thigh.q = target_joint_positions[4]
        msg.motor_cmd.front_left.calf.q = target_joint_positions[5]
        msg.motor_cmd.rear_right.hip.q = target_joint_positions[6]
        msg.motor_cmd.rear_right.thigh.q = target_joint_positions[7]
        msg.motor_cmd.rear_right.calf.q = target_joint_positions[8]
        msg.motor_cmd.rear_left.hip.q = target_joint_positions[9]
        msg.motor_cmd.rear_left.thigh.q = target_joint_positions[10]
        msg.motor_cmd.rear_left.calf.q = target_joint_positions[11]
        # print(target_joint_positions)
        self.publisher.publish(msg)
        # TODO: Create message and set target joint position and PD gains. E.g.:
        # joint_command_msg.kp = self.kp
        # joint_command_msg.kd = self.kd
        # joint_command_msg.t_pos = target_joint_positions.tolist()

        # TODO: Publish message

        self.previous_action = action


def main(args=None):
    rclpy.init(args=args)
    robot_handler = RobotHandler()
    rclpy.spin(robot_handler)
    robot_handler.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
