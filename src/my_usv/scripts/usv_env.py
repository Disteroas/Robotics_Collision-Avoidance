import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from std_srvs.srv import Empty
import numpy as np
import time

class UsvEnv(Node):
    def __init__(self):
        super().__init__('usv_rl_environment')
        self.vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.scan_sub = self.create_subscription(LaserScan, 'scan', self.scan_callback, 10)
        self.reset_client = self.create_client(Empty, '/reset_world')

        # Il paper taglia il LIDAR a 5.0 metri
        self.current_scan = np.ones(50) * 5.0 
        # Velocità lineare costante (es. 0.3 m/s)
        self.linear_velocity = 0.3 

    def reset_environment(self):
        # 1. Spegne i motori completamente (ferma il robot prima del teletrasporto)
        stop_cmd = Twist()
        self.vel_pub.publish(stop_cmd) 
        
        # 2. Chiama il reset di Gazebo
        while not self.reset_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn("Attendo Gazebo reset...")
            
        req = Empty.Request()
        future = self.reset_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        
        # 3. Attende che la fisica si stabilizzi
        time.sleep(0.5)
        
        # --- CANCELLA IL MURO FANTASMA ---
        # Forza la memoria del laser a 5.0 (tutto libero) prima di iniziare
        self.current_scan = np.ones(50) * 5.0 
        
        rclpy.spin_once(self, timeout_sec=0.1)
        return self.get_state()

    def scan_callback(self, msg):
        scan_data = np.array(msg.ranges)
        # Sostituisce inf con 5.0 e taglia i valori superiori a 5.0m come da paper
        scan_data = np.where(np.isinf(scan_data), 5.0, scan_data)
        scan_data = np.clip(scan_data, 0.0, 5.0)
        self.current_scan = scan_data

    def step_action(self, action_index):
        """ Applica una delle 11 azioni discrete del paper """
        vel_cmd = Twist()
        vel_cmd.linear.x = self.linear_velocity
        # Formula esatta dal paper: w_m = -0.8 + 0.16 * m
        vel_cmd.angular.z = -0.8 + 0.16 * action_index 
        self.vel_pub.publish(vel_cmd)

        # --- IL FIX DEL TEMPO REALE ---
        # Usa il VERO tempo di Python per forzare l'azione a durare 0.1 secondi (10 Hz)
        time.sleep(0.1) 
        rclpy.spin_once(self, timeout_sec=0.01)

        min_lidar_distance = np.min(self.current_scan)
        
        # Reward esatto dal paper: -1000 collisione, +5 sopravvivenza
        if min_lidar_distance < 0.25:
            reward = -1000.0
            done = True
        else:
            reward = 5.0
            done = False

        state = self.get_state()
        return state, reward, done

    def get_state(self):
        # Lo stato sono SOLO i 50 raggi del LIDAR
        return self.current_scan.copy()