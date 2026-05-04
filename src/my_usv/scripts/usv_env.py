import math
import rclpy
import rclpy.parameter
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from std_srvs.srv import Empty
import numpy as np

from usv_logic import process_lidar, compute_reward, LIDAR_MAX_RANGE, LIDAR_BEAMS, LINEAR_VEL

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

        self.vel_pub      = self.create_publisher(Twist, 'cmd_vel', 10)
        self.scan_sub     = self.create_subscription(LaserScan, 'scan', self._scan_cb, 10)
        self.reset_client = self.create_client(Empty, '/reset_world')

        self.current_scan = np.ones(LIDAR_BEAMS, dtype=np.float32) * LIDAR_MAX_RANGE
        self.accepting_scans = True
        self._lidar_checked = False

        # Attesa del clock simulato di Gazebo senza wallclock
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

        # Usa rigorosamente il clock di ROS2 dipendente da Gazebo
        while self.get_clock().now() < target_time:
            rclpy.spin_once(self, timeout_sec=0.001)

    # ──────────────────────────────────────────────────────────────
    # RESET
    # ──────────────────────────────────────────────────────────────
    def reset_environment(self) -> np.ndarray:
        self.vel_pub.publish(Twist())
        self.accepting_scans = False
        self.current_scan = np.ones(LIDAR_BEAMS, dtype=np.float32) * LIDAR_MAX_RANGE
        self._lidar_checked = False

        while not self.reset_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn("Attendo /reset_world...")

        future = self.reset_client.call_async(Empty.Request())

        while not future.done():
            rclpy.spin_once(self, timeout_sec=0.1)

        # Drenaggio deterministico della coda QoS
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
            min_deg = math.degrees(msg.angle_min)
            max_deg = math.degrees(msg.angle_max)
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
