import math
import random
from collections import deque
import rclpy
import rclpy.parameter
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from std_srvs.srv import Empty
from gazebo_msgs.srv import SetEntityState
import numpy as np

from usv_logic import process_lidar, compute_reward, LIDAR_MAX_RANGE, LIDAR_BEAMS, LINEAR_VEL

FRAME_STACK = 3    # Mnih et al. 2015: k frame stack risolve POMDP aliasing
STEP_DT     = 0.1  # durata simulata per step (s) — accoppiata con _wait_sim_seconds

# Random spawn positions per maze — validate with ./test_spawns.sh before training
SPAWN_LISTS = {
    1: [
        (-2.9, -2.0, 1.571),  # P1: left channel, heading N — validated
        ( 1.0, -1.0, 1.571),  # P2: inner chamber, heading N — validated
    ],
    2: [
        # Zone A: ingresso sinistro (1 spawn) — A2 rimosso (rischio uscita labirinto)
        (-6.0,  0.0,  0.0  ),  # A1: heading E  — min=1.352m

        # Zone B: rimosso — B3 tossico (0% max-steps su 8000 ep, 1137 crash, strutturale)

        # Zone C: centro (1 spawn) — C1/C3 rimossi (percorso interno, U-turn non verificato)
        (-7.0,  5.0,  0.0  ),  # C2: heading E  — min=0.860m

        # Zone D: centro-destra (1 spawn) — D2/D3 rimossi (0% completion su 3 run, <130 step medi)
        ( 1.5,  0.0,  3.142),  # D1: heading W  — min=0.693m  [bimodale: 12% max-steps]

        # Zone F: inferiore (3 spawn) — validated min≥0.43m
        (-4.5, -3.5,  0.0  ),  # F1: heading E  — min=1.162m
        (-1.5, -4.0,  1.571),  # F2: heading N  — min=1.008m
        ( 6.0,  6.0,  3.142),  # F3: heading W  — min=0.456m
    ],
    3: [
        (-2.5, -0.25,  0.0  ),  # fixed spawn — test only, never used in training
    ],
}

# Deterministic test spawn sets — reproducible subset of SPAWN_LISTS.
# M2: identico a SPAWN_LISTS[2] — misura training success, non within-M2 generalization.
#     La generalization cross-environment è già coperta da M3 (zero-shot).
#     Cobbe et al. 2019: test set misura generalization solo se diverso da training;
#     per M2 vogliamo misurare "quali spawn ha risolto?", non generalization intra-maze.
# M3: single fixed point (maze never seen in training — zero-shot generalization).
TEST_SPAWN_LISTS = {
    1: SPAWN_LISTS[1],  # both M1 points
    2: SPAWN_LISTS[2],  # identical to training — measures policy quality, not generalization
    3: SPAWN_LISTS[3],  # single fixed point
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

        self.current_scan    = np.ones(LIDAR_BEAMS, dtype=np.float32) * LIDAR_MAX_RANGE
        self.accepting_scans = True
        self._lidar_checked  = False
        self.last_spawn      = (0.0, 0.0, 0.0)
        self._frame_buffer   = deque(maxlen=FRAME_STACK)
        self._current_yaw    = 0.0

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
    def reset_environment(self, maze_id: int = 1, test_mode: bool = False) -> np.ndarray:
        self.vel_pub.publish(Twist())
        self.accepting_scans = False
        self.current_scan = np.ones(LIDAR_BEAMS, dtype=np.float32) * LIDAR_MAX_RANGE

        while not self.reset_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn("Attendo /reset_world...")

        future = self.reset_client.call_async(Empty.Request())
        while not future.done():
            rclpy.spin_once(self, timeout_sec=0.1)

        spawn_list = TEST_SPAWN_LISTS[maze_id] if test_mode else SPAWN_LISTS[maze_id]
        for attempt in range(SPAWN_MAX_RETRIES):
            x, y, yaw = random.choice(spawn_list)
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

        self.last_spawn      = (x, y, yaw)
        self._current_yaw    = yaw
        self._frame_buffer.clear()
        self.accepting_scans = True
        self._push_frame()
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
        # Integrazione yaw da velocità angolare comandata (dt=STEP_DT).
        # Approssimazione: assume tracking perfetto del comando (no slip/lag idrodinamico).
        self._current_yaw = (self._current_yaw + cmd.angular.z * STEP_DT) % (2 * math.pi)
        self.vel_pub.publish(cmd)

        self._wait_sim_seconds(STEP_DT)
        rclpy.spin_once(self, timeout_sec=0.05)

        reward, done = compute_reward(self.current_scan, action_index)
        self._push_frame()
        return self.get_state(), reward, done

    def _push_frame(self) -> None:
        """Avanza il frame buffer con lo scan corrente. Chiamare una sola volta per step/reset."""
        scan = (self.current_scan / LIDAR_MAX_RANGE).copy()
        self._frame_buffer.append(scan)
        # Padding: primi FRAME_STACK-1 step dell'episodio hanno frame iniziale duplicato (Mnih 2015).
        while len(self._frame_buffer) < FRAME_STACK:
            self._frame_buffer.appendleft(scan)

    def get_state(self) -> np.ndarray:
        """Lettura pura dello stato corrente — non modifica il frame buffer."""
        stacked = np.concatenate(list(self._frame_buffer))                        # 150 dim
        heading = np.array([np.cos(self._current_yaw),
                            np.sin(self._current_yaw)], dtype=np.float32)         # 2 dim
        return np.concatenate([stacked, heading])                                  # 152 dim
