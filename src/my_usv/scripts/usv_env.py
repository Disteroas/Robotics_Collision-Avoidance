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

# ---------------------------------------------------------------------------
# SPAWN LISTS
#
# Spawn F1 (-4.5, -3.5) RIMOSSO da SPAWN_LISTS[2].
# Motivazione cinematica: con v_lin = 0.5 m/s e omega_max = 0.8 rad/s,
# il raggio minimo di sterzata è R_min = v / omega_max = 0.5 / 0.8 = 0.625 m.
# Quella posizione richiede una manovra con raggio < R_min → nessuna policy
# discreta può evitare la collisione. Lasciarlo in training avvelena il replay
# buffer con crash inevitabili e non informativi per l'agente.
# ---------------------------------------------------------------------------
SPAWN_LISTS = {
    1: [(-2.9, -2.0, 1.571), (1.0, -1.0, 1.571)],
    2: [
        (-6.0,  0.0, 0.0),
        (-4.5,  1.5, 2.356),
        (-7.0,  5.0, 0.0),
        ( 1.5,  0.0, 3.142),
        ( 0.5, -2.0, 1.571),
        ( 3.5,  0.5, 4.712),
        ( 0.0,  3.5, 3.142),
        # (-4.5, -3.5, 0.0)  ← RIMOSSO: infeasibile cinematicamente (R_min = 0.625 m)
        (-1.5, -4.0, 1.571),
        ( 6.0,  6.0, 3.142),
    ],
    3: [(-2.0, -1.0, 0.0)],
}

TEST_SPAWN_LISTS = {
    1: SPAWN_LISTS[1],
    2: [
        (-6.0,  0.0, 0.0),
        (-4.5,  1.5, 2.356),
        (-7.0,  5.0, 0.0),
        ( 0.5, -2.0, 1.571),
        ( 0.0,  3.5, 3.142),
        # (-4.5, -3.5, 0.0)  ← RIMOSSO: stessa ragione
    ],
    3: SPAWN_LISTS[3],
}

SPAWN_SAFETY_DIST = 0.40
SPAWN_MAX_RETRIES = 3


class UsvEnv(Node):
    def __init__(self):
        super().__init__(
            'usv_rl_environment',
            parameter_overrides=[rclpy.parameter.Parameter('use_sim_time', rclpy.Parameter.Type.BOOL, True)]
        )

        self.vel_pub         = self.create_publisher(Twist, 'cmd_vel', 10)
        self.scan_sub        = self.create_subscription(LaserScan, 'scan', self._scan_cb, 10)
        self.reset_client    = self.create_client(Empty, '/reset_world')
        self.teleport_client = self.create_client(SetEntityState, '/gazebo/set_entity_state')

        self.current_scan      = np.ones(LIDAR_BEAMS, dtype=np.float32) * LIDAR_MAX_RANGE
        self.accepting_scans   = True
        self.new_scan_received = False
        self.last_spawn        = (0.0, 0.0, 0.0)

        self.get_logger().info("Attendo clock simulato di Gazebo...")
        while self.get_clock().now().nanoseconds == 0:
            rclpy.spin_once(self, timeout_sec=0.1)
        self.get_logger().info("Clock simulato attivo.")

    def _wait_sim_seconds(self, sim_sec: float) -> None:
        start_time  = self.get_clock().now()
        target_time = start_time + rclpy.time.Duration(seconds=sim_sec)
        while self.get_clock().now() < target_time:
            rclpy.spin_once(self, timeout_sec=0.001)

    def _teleport(self, x: float, y: float, yaw: float) -> None:
        req = SetEntityState.Request()
        req.state.name              = 'usv_robot'
        req.state.pose.position.x   = float(x)
        req.state.pose.position.y   = float(y)
        req.state.pose.orientation.z = math.sin(yaw / 2.0)
        req.state.pose.orientation.w = math.cos(yaw / 2.0)
        future = self.teleport_client.call_async(req)
        while not future.done():
            rclpy.spin_once(self, timeout_sec=0.01)
        self._wait_sim_seconds(0.3)

    def reset_environment(self, maze_id: int = 1, test_mode: bool = False) -> np.ndarray:
        self.vel_pub.publish(Twist())
        self.accepting_scans = False
        self.current_scan    = np.ones(LIDAR_BEAMS, dtype=np.float32) * LIDAR_MAX_RANGE

        future = self.reset_client.call_async(Empty.Request())
        while not future.done():
            rclpy.spin_once(self, timeout_sec=0.1)

        spawn_list = TEST_SPAWN_LISTS[maze_id] if test_mode else SPAWN_LISTS[maze_id]
        for attempt in range(SPAWN_MAX_RETRIES):
            x, y, yaw = random.choice(spawn_list)
            self._teleport(x, y, yaw)

            self.new_scan_received = False
            self.accepting_scans   = True

            t0 = self.get_clock().now()
            while not self.new_scan_received:
                rclpy.spin_once(self, timeout_sec=0.05)
                if (self.get_clock().now() - t0).nanoseconds * 1e-9 > 1.0:
                    break

            # FIX: current_scan è già in metri (output di process_lidar in usv_logic.py).
            # La versione precedente moltiplicava per LIDAR_MAX_RANGE (5.0),
            # gonfiando ogni distanza di 5x → il check era sempre True
            # anche con il robot a 0.1 m dal muro. Ora il confronto è diretto.
            if float(self.current_scan.min()) >= SPAWN_SAFETY_DIST:
                break

            self.get_logger().warn(
                f"[SPAWN] Tentativo {attempt + 1}/{SPAWN_MAX_RETRIES}: "
                f"min_lidar={self.current_scan.min():.3f}m < {SPAWN_SAFETY_DIST}m. Riprovo..."
            )
            self.accepting_scans = False

        self.last_spawn      = (x, y, yaw)
        self.accepting_scans = True
        return self.get_state()

    def _scan_cb(self, msg: LaserScan) -> None:
        if not self.accepting_scans:
            return
        self.current_scan      = process_lidar(msg.ranges)
        self.new_scan_received = True

    def step_action(self, action_index: int):
        self.new_scan_received = False
        cmd = Twist()
        cmd.linear.x  = LINEAR_VEL
        cmd.angular.z = -0.8 + 0.16 * action_index
        self.vel_pub.publish(cmd)

        t0 = self.get_clock().now()
        while not self.new_scan_received:
            rclpy.spin_once(self, timeout_sec=0.01)
            if (self.get_clock().now() - t0).nanoseconds * 1e-9 > 0.5:
                break

        reward, done = compute_reward(self.current_scan, action_index)
        return self.get_state(), reward, done

    def get_state(self) -> np.ndarray:
        # Normalizzazione in [0, 1] + protezione NaN
        state = np.array(self.current_scan, dtype=np.float32) / LIDAR_MAX_RANGE
        return np.nan_to_num(state, nan=1.0, posinf=1.0, neginf=0.0).clip(0.0, 1.0)
