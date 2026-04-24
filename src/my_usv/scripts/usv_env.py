import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from std_srvs.srv import Empty
import numpy as np
import time


# Costanti globali (più facili da modificare)
LIDAR_MAX_RANGE   = 5.0    # Distanza massima LIDAR (paper)
LIDAR_BEAMS       = 50     # Numero di raggi (paper)
COLLISION_DIST    = 0.25   # Soglia collisione (paper)
DANGER_DIST       = 1.0    # Soglia zona pericolo per reward shaping
LINEAR_VEL        = 0.5    # FIX: Aumentato da 0.3 → 0.5 m/s (vedi note)


class UsvEnv(Node):
    def __init__(self):
        super().__init__('usv_rl_environment')
        self.vel_pub    = self.create_publisher(Twist, 'cmd_vel', 10)
        self.scan_sub   = self.create_subscription(LaserScan, 'scan', self.scan_callback, 10)
        self.reset_client = self.create_client(Empty, '/reset_world')

        self.current_scan    = np.ones(LIDAR_BEAMS) * LIDAR_MAX_RANGE

        # ------------------------------------------------------------------ #
        # FIX BUG #1 – Ghost collision                                        #
        # Questo flag blocca scan_callback dall'accettare QUALSIASI messaggio  #
        # durante la procedura di reset. Viene riattivato SOLO dopo che        #
        # la fisica di Gazebo si è stabilizzata e la queue ROS2 è stata        #
        # svuotata. Senza questo, spin_once processa messaggi LIDAR vecchi     #
        # (quelli inviati PRIMA del reset quando il robot era vicino al muro). #
        # ------------------------------------------------------------------ #
        self.accepting_scans = True

    # ---------------------------------------------------------------------- #
    # RESET                                                                    #
    # ---------------------------------------------------------------------- #
    def reset_environment(self):
        # 1. Ferma i motori
        self.vel_pub.publish(Twist())

        # 2. BLOCCA subito l'accettazione di nuovi scan
        #    (qualsiasi messaggio arrivato da qui in poi viene ignorato)
        self.accepting_scans = False
        self.current_scan    = np.ones(LIDAR_BEAMS) * LIDAR_MAX_RANGE

        # 3. Richiede il reset di Gazebo
        while not self.reset_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('Attendo il servizio /reset_world di Gazebo...')
        future = self.reset_client.call_async(Empty.Request())
        rclpy.spin_until_future_complete(self, future)

        # 4. SVUOTA la queue ROS2 di tutti i messaggi STANTII
        #    timeout_sec=0.0 → spin non-bloccante: processa un messaggio se
        #    presente, altrimenti ritorna subito. Poiché accepting_scans=False,
        #    scan_callback li scarta senza aggiornare current_scan.
        for _ in range(60):
            rclpy.spin_once(self, timeout_sec=0.0)

        # 5. Aspetta che la fisica di Gazebo si stabilizzi
        time.sleep(0.8)

        # 6. ORA riabilita l'accettazione: i prossimi scan arriveranno
        #    dal robot nella posizione CORRETTA dopo il reset
        self.accepting_scans = True

        # 7. Raccoglie qualche scan fresco valido prima di restituire lo stato
        for _ in range(5):
            rclpy.spin_once(self, timeout_sec=0.1)

        return self.get_state()

    # ---------------------------------------------------------------------- #
    # SCAN CALLBACK                                                            #
    # ---------------------------------------------------------------------- #
    def scan_callback(self, msg):
        # FIX BUG #1: se siamo in fase di reset, ignora il messaggio
        if not self.accepting_scans:
            return

        scan_data = np.array(msg.ranges, dtype=np.float32)

        # Gestisce inf, -inf e NaN (alcuni driver LIDAR li emettono)
        scan_data = np.nan_to_num(
            scan_data,
            nan=LIDAR_MAX_RANGE,
            posinf=LIDAR_MAX_RANGE,
            neginf=LIDAR_MAX_RANGE
        )
        scan_data = np.clip(scan_data, 0.0, LIDAR_MAX_RANGE)
        self.current_scan = scan_data

    # ---------------------------------------------------------------------- #
    # STEP                                                                     #
    # ---------------------------------------------------------------------- #
    def step_action(self, action_index):
        """Applica una delle 11 azioni discrete del paper."""
        vel_cmd = Twist()
        vel_cmd.linear.x  = LINEAR_VEL
        vel_cmd.angular.z = -0.8 + 0.16 * action_index  # formula paper
        self.vel_pub.publish(vel_cmd)

        time.sleep(0.1)                          # attende 100 ms (10 Hz)
        rclpy.spin_once(self, timeout_sec=0.01)  # riceve il prossimo scan

        min_dist = float(np.min(self.current_scan))

        # ------------------------------------------------------------------ #
        # FIX BUG #2 – Reward shaping con zona pericolo                       #
        #                                                                      #
        # Problema originale:                                                   #
        #   max_reward_episodio = MAX_STEPS * 5 = 3000 * 5 = 15.000           #
        #   Crash a fine episodio → 14.995 - 1.000 = +13.995 (ancora positivo)#
        #   L'agente NON impara che crashare è male perché guadagna comunque.  #
        #                                                                      #
        # Soluzione: penalità graduale per l'avvicinamento agli ostacoli.      #
        # In [COLLISION_DIST, DANGER_DIST] il reward scende linearmente da    #
        # +5 a 0, dando un segnale denso di apprendimento anche senza crash.  #
        # ------------------------------------------------------------------ #
        if min_dist < COLLISION_DIST:
            reward = -1000.0
            done   = True
        elif min_dist < DANGER_DIST:
            # Rampa lineare: 0 alla soglia collisione → +5 a distanza sicura
            ratio  = (min_dist - COLLISION_DIST) / (DANGER_DIST - COLLISION_DIST)
            reward = 5.0 * ratio
            done   = False
        else:
            reward = 5.0
            done   = False

        return self.get_state(), reward, done

    # ---------------------------------------------------------------------- #
    # STATO                                                                    #
    # ---------------------------------------------------------------------- #
    def get_state(self):
        # Normalizza in [0, 1]: le reti neurali convergono meglio con input
        # normalizzati. La divisione per LIDAR_MAX_RANGE non cambia la logica
        # (il modello vede la stessa struttura dei dati, solo in scala diversa).
        return (self.current_scan / LIDAR_MAX_RANGE).copy()
