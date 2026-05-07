#!/usr/bin/env python3
"""
validate_spawn.py — controlla se spawn point è valido (no collisione con muri)
Gira dentro container Docker dopo avvio Gazebo.

Exit code:
  0 = OK         (min LIDAR > 0.40m)
  1 = COLLISION  (min LIDAR < 0.25m — spawn dentro muro)
  2 = TIMEOUT    (Gazebo non risponde o scan non arriva)
  3 = WARNING    (min LIDAR 0.25–0.40m — troppo vicino al muro)
"""
import argparse
import sys
import time

import numpy as np
import rclpy
import rclpy.parameter
from rclpy.node import Node
from sensor_msgs.msg import LaserScan

COLLISION_DIST = 0.25
WARNING_DIST   = 0.40
MAX_RANGE      = 5.0


class SpawnValidator(Node):
    def __init__(self):
        super().__init__(
            'spawn_validator',
            parameter_overrides=[
                rclpy.parameter.Parameter(
                    'use_sim_time', rclpy.Parameter.Type.BOOL, True
                )
            ]
        )
        self.min_dist = None
        self.create_subscription(LaserScan, '/scan', self._scan_cb, 10)

    def _scan_cb(self, msg):
        if self.min_dist is not None:
            return
        ranges = np.array(msg.ranges, dtype=np.float32)
        ranges = np.nan_to_num(ranges, nan=MAX_RANGE, posinf=MAX_RANGE)
        ranges = np.clip(ranges, 0.0, MAX_RANGE)
        self.min_dist = float(np.min(ranges))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--timeout', type=float, default=20.0)
    args = parser.parse_args()

    rclpy.init()
    node = SpawnValidator()
    t0 = time.time()

    # Attendi clock simulato Gazebo
    while node.get_clock().now().nanoseconds == 0:
        rclpy.spin_once(node, timeout_sec=0.1)
        if time.time() - t0 > args.timeout:
            print('TIMEOUT: sim clock non avviato')
            node.destroy_node(); rclpy.shutdown()
            sys.exit(2)

    # Attendi primo scan LIDAR
    while node.min_dist is None:
        rclpy.spin_once(node, timeout_sec=0.1)
        if time.time() - t0 > args.timeout:
            print('TIMEOUT: nessun /scan ricevuto')
            node.destroy_node(); rclpy.shutdown()
            sys.exit(2)

    d = node.min_dist
    node.destroy_node()
    rclpy.shutdown()

    if d < COLLISION_DIST:
        print(f'COLLISION  min={d:.3f}m  (soglia={COLLISION_DIST}m)')
        sys.exit(1)
    elif d < WARNING_DIST:
        print(f'WARNING    min={d:.3f}m  (troppo vicino al muro)')
        sys.exit(3)
    else:
        print(f'OK         min={d:.3f}m')
        sys.exit(0)


if __name__ == '__main__':
    main()
