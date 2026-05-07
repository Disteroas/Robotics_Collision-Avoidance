import math
import random
import rclpy
import rclpy.parameter
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from std_srvs.srv import Empty
from gazebo_msgs.srv import SetEntityState
import numpy as np

from usv_logic import process_lidar, compute_reward, LIDAR_MAX_RANGE, LIDAR_BEAMS, LINEAR_VEL

# Random spawn positions per maze — validate with ./test_spawns.sh before training
SPAWN_LISTS = {
    1: [
        (-3.0, -5.0,  1.57),  # M1-A1: south open, heading N
        ( 0.0, -4.5,  1.57),  # M1-A2: centre-south, heading N
        ( 2.5, -5.0,  1.57),  # M1-A3: right-south, heading N
        (-1.5, -5.0,  0.0 ),  # M1-A4: south open, heading E
        (-2.9, -2.0,  1.57),  # M1-B1: left channel entry, heading N
        (-2.9,  0.5,  0.0 ),  # M1-B2: left channel mid, heading E
        ( 2.5, -2.0,  1.57),  # M1-C1: right outer, heading N
        ( 0.5, -2.5,  1.57),  # M1-D1: centre-bottom, heading N
    ],
    2: [
        (-6.0,  0.0,  0.0 ),  # M2-A1: left entrance, heading E
        (-6.0, -1.5,  0.0 ),  # M2-A2: lower-left, heading E
        (-6.0,  2.0,  0.0 ),  # M2-A3: upper-left, heading E
        (-6.0,  0.0,  1.57),  # M2-A4: left entrance, heading N
        (-3.5,  0.5,  0.0 ),  # M2-B1: centre-left, heading E
        (-3.5, -2.5,  1.57),  # M2-B2: centre-left low, heading N
        (-1.5, -2.5,  0.0 ),  # M2-C1: between Wall_32/Wall_20, heading E
        ( 1.5,  0.0,  3.14),  # M2-D1: right-centre, heading W
    ],
}


class UsvEnv(Node):

    def __init__(self):
        super().__init__(
            'usv_rl_environment',
            parameter_overrides=[
                rclpy.parameter.Parameter(
                    'use_sim_time', rclpy.Parameter.Type.BOOL, True
                )
            ]
        )

        self.vel_pub        = self.create_publisher(Twist, 'cmd_vel', 10)
        self.scan_sub       = self.create_subscription(LaserScan, 'scan', self._scan_cb, 10)
        self.reset_client   = self.create_client(Empty, '/reset_world')
        self.teleport_client = self.create_client(SetEntityState, '/gazebo/set_entity_state')

        self.current_scan   = np.ones(LIDAR_BEAMS, dtype=np.float32) * LIDAR_MAX_RANGE
        self.accepting_scans = True
        self._lidar_checked = False

        self.get_logger().info("Attendo clock simulato di Gazebo...")
        while self.get_clock().now().nanoseconds == 0:
            rclpy.spin_once(self, timeout_sec=0.1)
        self.get_logger().info("Clock simulato attivo.")

    # ──────────────────────────────────────────────────────────────
    # CLOCK SIMULATO
    # ──────────────────────────────────────────────────────────────
    def _wait_sim_seconds(self, sim_sec: float) -> None:
        start_time = self.get_clock().now()
        wait_duration = rclpy.time.Duration(seconds=sim_sec)
        target_time = start_time + wait_duration
        while self.get_clock().now() < target_time:
            rclpy.spin_once(self, timeout_sec=0.001)

    # ──────────────────────────────────────────────────────────────
    # TELEPORT
    # ──────────────────────────────────────────────────────────────
    def _teleport(self, x: float, y: float, yaw: float) -> None:
        while not self.teleport_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn("Attendo /gazebo/set_entity_state...")
        req = SetEntityState.Request()
        req.state.name = 'usv_robot'
        req.state.pose.position.x = float(x)
        req.state.pose.position.y = float(y)
        req.state.pose.position.z = 0.0
        req.state.pose.orientation.x = 0.0
        req.state.pose.orientation.y = 0.0
        req.state.pose.orientation.z = math.sin(yaw / 2.0)
        req.state.pose.orientation.w = math.cos(yaw / 2.0)
        req.state.twist.linear.x  = 0.0
        req.state.twist.angular.z = 0.0
        future = self.teleport_client.call_async(req)
        while not future.done():
            rclpy.spin_once(self, timeout_sec=0.01)
        self._wait_sim_seconds(0.3)

    # ──────────────────────────────────────────────────────────────
    # RESET
    # ──────────────────────────────────────────────────────────────
    def reset_environment(self, maze_id: int = 1) -> np.ndarray:
        self.vel_pub.publish(Twist())
        self.accepting_scans = False
        self.current_scan = np.ones(LIDAR_BEAMS, dtype=np.float32) * LIDAR_MAX_RANGE
        self._lidar_checked = False

        while not self.reset_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn("Attendo /reset_world...")

        future = self.reset_client.call_async(Empty.Request())
        while not future.done():
            rclpy.spin_once(self, timeout_sec=0.1)

        x, y, yaw = random.choice(SPAWN_LISTS[maze_id])
        self._teleport(x, y, yaw)

        for _ in range(20):
            rclpy.spin_once(self, timeout_sec=0.0)

        self._wait_sim_seconds(0.8)
        self.accepting_scans = True

        for _ in range(5):
            rclpy.spin_once(self, timeout_sec=0.1)

        return self.get_state()

    # ──────────────────────────────────────────────────────────────
    # SCAN CALLBACK
    # ──────────────────────────────────────────────────────────────
    def _scan_cb(self, msg: LaserScan) -> None:
        if not self.accepting_scans:
            return

        if not self._lidar_checked:
            import math as _math
            min_deg = _math.degrees(msg.angle_min)
            max_deg = _math.degrees(msg.angle_max)
            self.get_logger().info(
                f"LIDAR INFO: Ricevuti {len(msg.ranges)} raggi | "
                f"FOV=[{min_deg:.1f}°, {max_deg:.1f}°]"
            )
            self._lidar_checked = True

        self.current_scan = process_lidar(msg.ranges)

    # ──────────────────────────────────────────────────────────────
    # STEP
    # ──────────────────────────────────────────────────────────────
    def step_action(self, action_index: int):
        cmd = Twist()
        cmd.linear.x  = LINEAR_VEL
        cmd.angular.z = -0.8 + 0.16 * action_index
        self.vel_pub.publish(cmd)

        self._wait_sim_seconds(0.1)
        rclpy.spin_once(self, timeout_sec=0.05)

        reward, done = compute_reward(self.current_scan, action_index)
        return self.get_state(), reward, done

    def get_state(self) -> np.ndarray:
        return (self.current_scan / LIDAR_MAX_RANGE).copy()
