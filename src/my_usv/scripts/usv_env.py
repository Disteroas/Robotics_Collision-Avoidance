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
        # Zone A: sud (4 spawn)
        (-3.0, -5.0,  1.57 ),  # A1: south open, heading N
        ( 0.0, -4.5,  1.57 ),  # A2: centre-south, heading N
        ( 2.5, -5.0,  1.57 ),  # A3: right-south, heading N
        (-1.5, -5.0,  0.0  ),  # A4: south open, heading E

        # Zone B: sinistra (3 spawn)
        (-2.9, -2.0,  1.57 ),  # B1: left channel entry, heading N
        (-2.9,  0.5,  0.0  ),  # B2: left channel mid, heading E
        (-2.5, -3.5,  0.785),  # B3: left-mid, heading NE

        # Zone C: centro (3 spawn)
        ( 0.5, -2.5,  1.57 ),  # C1: centre-bottom, heading N
        ( 0.0, -1.0,  3.142),  # C2: centre, heading W
        ( 0.0,  0.0,  1.57 ),  # C3: centre, heading N

        # Zone D: destra (2 spawn)
        ( 2.5, -2.0,  1.57 ),  # D1: right outer, heading N
        ( 2.5,  0.0,  3.142),  # D2: right, heading W

        # Zone E: superiore (2 spawn)
        (-1.0,  1.5,  0.0  ),  # E1: upper-left, heading E
        ( 1.0,  1.5,  3.142),  # E2: upper-right, heading W

        # Zone F: copertura extra (2 spawn)
        (-3.0, -4.0,  0.0  ),  # F1: south-left, heading E
        ( 1.5, -4.0,  1.57 ),  # F2: south-right, heading N
    ],
    2: [
        # Zone A: ingresso sinistro (2 spawn) — validated min≥0.43m
        (-6.0,  0.0,  0.0  ),  # A1: heading E  — min=1.352m
        (-6.5, -0.5,  0.0  ),  # A2: heading E  — min=1.803m

        # Zone B: centro-sinistra (3 spawn) — validated min≥0.43m
        (-4.5,  0.5,  0.0  ),  # B1: heading E  — min=0.995m
        (-4.0, -1.0,  1.571),  # B2: heading N  — min=0.523m
        (-4.5,  1.5,  2.356),  # B3: heading NW — min=0.497m

        # Zone C: centro (3 spawn) — validated min≥0.43m
        (-2.5,  1.0,  0.0  ),  # C1: heading E  — min=0.434m
        (-7.0,  5.0,  0.0  ),  # C2: heading E  — min=0.860m
        (-2.0, -1.0,  0.785),  # C3: heading NE — min=0.795m

        # Zone D: centro-destra (3 spawn) — validated min≥0.43m
        ( 1.5,  0.0,  3.142),  # D1: heading W  — min=0.693m
        ( 0.5, -2.0,  1.571),  # D2: heading N  — min=0.430m
        ( 3.5,  0.5,  4.712),  # D3: heading S  — min=0.780m

        # Zone E: superiore (2 spawn) — validated min≥0.43m
        (-3.0,  3.0,  0.0  ),  # E1: heading E  — min=0.890m
        ( 0.0,  3.5,  3.142),  # E2: heading W  — min=0.650m

        # Zone F: inferiore (3 spawn) — validated min≥0.43m
        (-4.5, -3.5,  0.0  ),  # F1: heading E  — min=1.162m
        (-1.5, -4.0,  1.571),  # F2: heading N  — min=1.008m
        ( 6.0,  6.0,  3.142),  # F3: heading W  — min=0.456m
    ],
}

SPAWN_SAFETY_DIST = 0.40   # min LIDAR (m) acceptable after teleport
SPAWN_MAX_RETRIES = 3


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

        while not self.reset_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn("Attendo /reset_world...")

        future = self.reset_client.call_async(Empty.Request())
        while not future.done():
            rclpy.spin_once(self, timeout_sec=0.1)

        for attempt in range(SPAWN_MAX_RETRIES):
            x, y, yaw = random.choice(SPAWN_LISTS[maze_id])
            self._teleport(x, y, yaw)

            for _ in range(20):
                rclpy.spin_once(self, timeout_sec=0.0)

            self._wait_sim_seconds(0.8)
            self.accepting_scans = True

            for _ in range(5):
                rclpy.spin_once(self, timeout_sec=0.1)

            min_dist = float(self.current_scan.min()) * LIDAR_MAX_RANGE
            if min_dist >= SPAWN_SAFETY_DIST:
                break

            self.get_logger().warn(
                f"Spawn ({x:.1f},{y:.1f}) unsafe: min={min_dist:.2f}m < "
                f"{SPAWN_SAFETY_DIST}m, retry {attempt + 1}/{SPAWN_MAX_RETRIES}"
            )
            self.accepting_scans = False
            self.current_scan = np.ones(LIDAR_BEAMS, dtype=np.float32) * LIDAR_MAX_RANGE

        self.accepting_scans = True
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
