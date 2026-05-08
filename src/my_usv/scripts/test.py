"""
test.py  –  Valutazione della policy DDQN su un singolo labirinto.

Argomenti CLI (gestiti da start_test.sh):
  --maze-id     INT   ID labirinto corrente (1/2/3), solo per logging
  --model       STR   Path del file best_ddqn_model.pth
  --episodes    INT   Numero di episodi di valutazione (default: 30)
  --output-csv  STR   Path CSV dove appendere i risultati

Epsilon = 0.0: nessuna esplorazione, policy puramente greedy.
MAX_STEPS = 500: coerente con il training (non testare oltre l'orizzonte visto).
"""

import argparse
import csv
import os
import sys

import numpy as np
import rclpy
import torch

from ddqn_model import DDQN
from usv_env import UsvEnv

MAX_STEPS = 500   # identico al training


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--maze-id',    type=int, default=1)
    p.add_argument('--model',      type=str,
                   default='src/my_usv/scripts/best_ddqn_model.pth')
    p.add_argument('--episodes',   type=int, default=30)
    p.add_argument('--output-csv', type=str,
                   default='src/my_usv/scripts/test_results.csv')
    return p.parse_args()


def main():
    args = parse_args()

    # ── Carica il modello ─────────────────────────────────────────
    if not os.path.exists(args.model):
        print(f"[ERRORE] Modello non trovato: {args.model}")
        sys.exit(1)

    q_net = DDQN()
    q_net.load_state_dict(torch.load(args.model, map_location='cpu'))
    q_net.eval()
    print(f"  ✅ Modello caricato: {args.model}")

    # ── CSV ───────────────────────────────────────────────────────
    csv_is_new = (not os.path.exists(args.output_csv) or
                  os.path.getsize(args.output_csv) == 0)
    csv_f = open(args.output_csv, 'a', newline='', encoding='utf-8')
    csv_w = csv.writer(csv_f)
    if csv_is_new:
        csv_w.writerow([
            'maze_id', 'episode', 'steps', 'reward', 'crashed',
            'min_lidar', 'avg_lidar'
        ])

    # ── ROS2 + env ────────────────────────────────────────────────
    rclpy.init()
    env = UsvEnv()

    print(f"\n  🏁 Valutazione su Maze {args.maze_id} "
          f"({args.episodes} episodi, ε=0.0)\n")
    print(f"  {'Ep':>4}  {'Steps':>5}  {'Reward':>8}  {'Esito'}")
    print(f"  {'─'*4}  {'─'*5}  {'─'*8}  {'─'*10}")

    rewards  = []
    steps_l  = []
    crashes  = 0
    min_lidars = []
    avg_lidars = []

    for ep in range(1, args.episodes + 1):
        state        = env.reset_environment(maze_id=args.maze_id)
        ep_reward    = 0.0
        ep_steps     = 0
        crashed      = False
        ep_min_lidar = []

        for step in range(MAX_STEPS):
            with torch.no_grad():
                action = int(q_net(
                    torch.FloatTensor(state).unsqueeze(0)
                ).argmax(dim=1).item())

            state, reward, done = env.step_action(action)

            # stato denormalizzato: lo stato in uscita da UsvEnv è /5.0
            raw_scan = state * 5.0
            ep_min_lidar.append(float(raw_scan.min()))

            ep_reward += reward
            ep_steps  += 1

            if done:
                crashed = True
                crashes += 1
                break

        rewards.append(ep_reward)
        steps_l.append(ep_steps)
        min_lidars.append(float(np.min(ep_min_lidar)) if ep_min_lidar else 5.0)
        avg_lidars.append(float(np.mean(ep_min_lidar)) if ep_min_lidar else 5.0)

        esito = '💥 CRASH' if crashed else '✅ OK   '
        print(f"  {ep:>4}  {ep_steps:>5}  {ep_reward:>8.1f}  {esito}")

        csv_w.writerow([
            args.maze_id, ep, ep_steps, round(ep_reward, 2),
            int(crashed),
            round(min_lidars[-1], 3), round(avg_lidars[-1], 3)
        ])
        csv_f.flush()

    # ── Report per-maze ───────────────────────────────────────────
    crash_rate    = crashes / args.episodes * 100
    avg_reward    = float(np.mean(rewards))
    std_reward    = float(np.std(rewards))
    avg_steps     = float(np.mean(steps_l))
    avg_min_lidar = float(np.mean(min_lidars))

    print(f"\n  {'─'*48}")
    print(f"  📊 RISULTATI  –  Maze {args.maze_id}")
    print(f"  {'─'*48}")
    print(f"  Episodi testati    : {args.episodes}")
    print(f"  Crash rate         : {crashes}/{args.episodes} ({crash_rate:.1f}%)")
    print(f"  Reward medio       : {avg_reward:.1f} ± {std_reward:.1f}")
    print(f"  Step medi          : {avg_steps:.1f} / {MAX_STEPS}")
    print(f"  Min lidar medio    : {avg_min_lidar:.3f} m")
    print(f"  {'─'*48}\n")

    csv_f.close()
    env.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
