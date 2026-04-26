"""
usv_env.py  –  Ambiente ROS2/Gazebo per il training DDQN di collision avoidance.

Modifiche principali:
  1. [BUG FIX]     Ghost collision: flag accepting_scans blocca scan stantii.
  2. [BUG FIX]     Reward shaping: penalità graduale zona pericolo [0.25, 1.0]m.
  3. [PERFORMANCE] use_sim_time=True: tutto il timing scala con real_time_factor.
  4. [QUALITÀ]     Input LIDAR normalizzato in [0,1].
  5. [ROBUSTEZZA]  NaN/Inf handling nei dati LIDAR.
"""

import rclpy
import rclpy.parameter
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from std_srvs.srv import Empty

import numpy as np
import time as wallclock   # orologio reale, usato SOLO per timeout di sicurezza

# ============================================================ #
#  COSTANTI DI CONFIGURAZIONE                                   #
# ============================================================ #
LIDAR_MAX_RANGE = 5.0    # m   – distanza massima LIDAR (paper)
LIDAR_BEAMS     = 50     # n   – numero di raggi (paper)
COLLISION_DIST  = 0.25   # m   – soglia collisione (paper)
DANGER_DIST     = 1.0    # m   – soglia zona pericolo per reward shaping
LINEAR_VEL      = 0.5    # m/s – velocità lineare costante del robot


class UsvEnv(Node):

    def __init__(self):
        # ------------------------------------------------------------------ #
        # CRITICO: use_sim_time nel constructor (NON dopo).                   #
        # Fa sì che self.get_clock() legga il topic /clock di Gazebo.        #
        # _wait_sim_seconds() scala quindi automaticamente con                #
        # il real_time_factor impostato in start_sim.sh.                      #
        # ------------------------------------------------------------------ #
        super().__init__(
            'usv_rl_environment',
            parameter_overrides=[
                rclpy.parameter.Parameter(
                    'use_sim_time',
                    rclpy.Parameter.Type.BOOL,
                    True
                )
            ]
        )

        self.vel_pub      = self.create_publisher(Twist, 'cmd_vel', 10)
        self.scan_sub     = self.create_subscription(
            LaserScan, 'scan', self._scan_callback, 10
        )
        self.reset_client = self.create_client(Empty, '/reset_world')

        self.current_scan    = np.ones(LIDAR_BEAMS, dtype=np.float32) * LIDAR_MAX_RANGE

        # Quando False, scan_callback scarta ogni messaggio in arrivo.
        # Attivato=False durante il reset per bloccare scan stantii
        # (erano la causa del reward -995 "fantasma").
        self.accepting_scans = True

        # Aspetta il clock prima di procedere
        self._wait_for_sim_clock(timeout_real_sec=30.0)

    # ====================================================================== #
    #  GESTIONE CLOCK SIMULATO                                                 #
    # ====================================================================== #

    def _wait_for_sim_clock(self, timeout_real_sec: float = 30.0) -> None:
        """
        Blocca finché Gazebo non pubblica il primo messaggio su /clock.
        Finché non arriva, get_clock().now().nanoseconds == 0.
        """
        self.get_logger().info("Attendo il clock simulato di Gazebo...")
        deadline = wallclock.time() + timeout_real_sec

        while self.get_clock().now().nanoseconds == 0:
            if wallclock.time() > deadline:
                raise RuntimeError(
                    f"Timeout {timeout_real_sec}s: Gazebo non ha pubblicato "
                    "/clock. Verificare che start_sim.sh sia in esecuzione."
                )
            rclpy.spin_once(self, timeout_sec=0.1)

        self.get_logger().info("Clock simulato attivo. Inizio training.")

    def _wait_sim_seconds(self, sim_seconds: float) -> None:
        """
        Attende esattamente `sim_seconds` di TEMPO SIMULATO.

        Esempio con real_time_factor=5 (impostato in start_sim.sh):
            _wait_sim_seconds(0.1)  →  attesa reale  0.02 s  (20 ms)
            _wait_sim_seconds(0.8)  →  attesa reale  0.16 s  (160 ms)

        Il loop chiama spin_once durante l'attesa, quindi i messaggi ROS2
        (nuovi scan LIDAR inclusi) vengono processati mentre si aspetta.

        Safety: se il wall-clock supera sim_seconds * 10 secondi reali,
        esce comunque per non bloccare il training in caso di freeze di Gazebo.
        """
        clock     = self.get_clock()
        start_ns  = clock.now().nanoseconds
        target_ns = start_ns + int(sim_seconds * 1e9)

        real_deadline = wallclock.time() + sim_seconds * 10.0

        while clock.now().nanoseconds < target_ns:
            if wallclock.time() > real_deadline:
                self.get_logger().warn(
                    f"_wait_sim_seconds({sim_seconds}): safety timeout. "
                    "Gazebo potrebbe essere freezato."
                )
                break
            rclpy.spin_once(self, timeout_sec=0.001)

    # ====================================================================== #
    #  RESET                                                                   #
    # ====================================================================== #

    def reset_environment(self) -> np.ndarray:
        """
        Resetta Gazebo ed elimina il ghost collision bug.

        Ordine delle operazioni:
          1. Ferma motori
          2. Blocca scan_callback (accepting_scans = False)
          3. Chiama /reset_world (teletrasporto)
          4. Drena la queue ROS2 con wall-clock (scarta messaggi stantii)
          5. Aspetta stabilizzazione fisica (tempo SIMULATO)
          6. Riabilita scan_callback
          7. Raccoglie scan freschi dalla nuova posizione
        """
        # 1. Stop motori
        self.vel_pub.publish(Twist())

        # 2. Blocca SUBITO il callback – nessun dato stantio verrà accettato
        self.accepting_scans = False
        self.current_scan    = np.ones(LIDAR_BEAMS, dtype=np.float32) * LIDAR_MAX_RANGE

        # 3. Reset Gazebo
        while not self.reset_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn("Attendo /reset_world...")
        future = self.reset_client.call_async(Empty.Request())
        rclpy.spin_until_future_complete(self, future)

        # 4. Drain della queue in tempo REALE.
        #    Usiamo wall-clock qui perché vogliamo svuotare il buffer di rete
        #    il più velocemente possibile, indipendentemente dalla sim speed.
        #    0.3 secondi reali bastano per smaltire qualsiasi backlog.
        drain_end = wallclock.time() + 0.3
        while wallclock.time() < drain_end:
            rclpy.spin_once(self, timeout_sec=0.0)

        # 5. Stabilizzazione fisica: 0.8 secondi SIMULATI.
        #    Con real_time_factor=5 → 0.16 s reali, ma 0.8 s di fisica
        #    simulata: sufficiente per fermare il robot nella nuova posizione.
        #    accepting_scans è ancora False → i messaggi vengono ricevuti
        #    dal sistema ROS2 ma ignorati dal callback.
        self._wait_sim_seconds(0.8)

        # 6. Riabilita: d'ora in poi i messaggi vengono dalla posizione corretta
        self.accepting_scans = True

        # 7. Leggi almeno 5 scan freschi prima di restituire lo stato
        for _ in range(5):
            rclpy.spin_once(self, timeout_sec=0.1)

        return self.get_state()

    # ====================================================================== #
    #  SCAN CALLBACK                                                           #
    # ====================================================================== #

    def _scan_callback(self, msg: LaserScan) -> None:
        if not self.accepting_scans:
            return

        scan = np.array(msg.ranges, dtype=np.float32)
        scan = np.nan_to_num(
            scan,
            nan=LIDAR_MAX_RANGE,
            posinf=LIDAR_MAX_RANGE,
            neginf=LIDAR_MAX_RANGE
        )
        self.current_scan = np.clip(scan, 0.0, LIDAR_MAX_RANGE)

    # ====================================================================== #
    #  STEP                                                                    #
    # ====================================================================== #

    def step_action(self, action_index: int):
        """
        Esegue un'azione e restituisce (next_state, reward, done).

        Formula azione (paper): omega_m = -0.8 + 0.16 * m
            m=0  → omega = -0.80 rad/s  (massimo a sinistra)
            m=5  → omega =  0.00 rad/s  (dritto)
            m=10 → omega = +0.80 rad/s  (massimo a destra)

        Il robot avanza per 0.1 secondi SIMULATI (10 Hz simulati).
        Con real_time_factor=5 → 0.02 secondi reali per step.
        """
        vel_cmd = Twist()
        vel_cmd.linear.x  = LINEAR_VEL
        vel_cmd.angular.z = -0.8 + 0.16 * action_index
        self.vel_pub.publish(vel_cmd)

        # _wait_sim_seconds chiama spin_once internamente → riceve lo scan
        # aggiornato durante l'attesa. Nessun ulteriore spin_once necessario.
        self._wait_sim_seconds(0.1)

        min_dist = float(np.min(self.current_scan))
        reward, done = self._compute_reward(min_dist)

        return self.get_state(), reward, done

    # ====================================================================== #
    #  REWARD                                                                  #
    # ====================================================================== #

    def _compute_reward(self, min_dist: float):
        """
        3 zone di reward:

          [0, 0.25)      → -1000  collisione, done=True
          [0.25, 1.0)    → rampa  da 0 a +5  (segnale denso)
          [1.0, ∞)       → +5.0   navigazione sicura

        Con MAX_STEPS=500 il massimo episodio = 2500.
        Il crash (-1000) ha peso reale sul totale.
        La rampa fornisce feedback continuo prima del crash.
        """
        if min_dist < COLLISION_DIST:
            return -1000.0, True

        if min_dist < DANGER_DIST:
            ratio = (min_dist - COLLISION_DIST) / (DANGER_DIST - COLLISION_DIST)
            return 5.0 * ratio, False

        return 5.0, False

    # ====================================================================== #
    #  STATO                                                                   #
    # ====================================================================== #

    def get_state(self) -> np.ndarray:
        """50 raggi LIDAR normalizzati in [0, 1]."""
        return (self.current_scan / LIDAR_MAX_RANGE).copy()
