import argparse
import csv
import os
import torch
import numpy as np
import rclpy
from usv_env import UsvEnv
from ddqn_model import DDQN

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--maze-id',    type=int, default=1)
    p.add_argument('--model',      type=str, default='src/my_usv/scripts/best_ddqn_model.pth')
    p.add_argument('--episodes',   type=int, default=90)
    p.add_argument('--output-csv', type=str, default='src/my_usv/scripts/test_results.csv')
    return p.parse_args()

def main():
    args = parse_args()
    rclpy.init()
    env = UsvEnv()
    model = DDQN()

    if os.path.exists(args.model):
        model.load_state_dict(torch.load(args.model, weights_only=True))
        model.eval()
    else:
        print(f"❌ Modello non trovato: {args.model}")
        env.destroy_node()
        rclpy.shutdown()
        return

    file_exists = os.path.isfile(args.output_csv)
    with open(args.output_csv, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                'maze_id', 'episode', 'steps', 'reward',
                'crashed', 'min_lidar_m', 'avg_lidar_m', 'spawn'
            ])

        for ep in range(args.episodes):
            state = env.reset_environment(maze_id=args.maze_id, test_mode=True)
            state = np.nan_to_num(state, nan=1.0, posinf=1.0, neginf=0.0)

            ep_rew = 0.0
            done   = False
            steps  = 0

            sx, sy, _ = env.last_spawn
            spawn_str = f"({sx:.1f},{sy:.1f})"

            while not done and steps < 500:
                with torch.no_grad():
                    q_vals = model(torch.FloatTensor(state).unsqueeze(0))
                    action = q_vals.argmax().item()

                next_state, reward, done = env.step_action(action)
                next_state = np.nan_to_num(next_state, nan=1.0, posinf=1.0, neginf=0.0)

                ep_rew += reward
                state   = next_state
                steps  += 1

            # FIX: log LIDAR in metri fisici (env.current_scan è output di process_lidar,
            # già in [0, 5.0] m) invece di np.min(state) che è normalizzato [0,1].
            # La versione precedente scriveva valori come -1.5 perché state alla fine
            # dell'episodio poteva contenere valori anomali non ancora clippati.
            min_lidar_m = round(float(env.current_scan.min()), 3)
            avg_lidar_m = round(float(env.current_scan.mean()), 3)

            writer.writerow([
                args.maze_id, ep + 1, steps,
                round(ep_rew, 2), int(done),
                min_lidar_m, avg_lidar_m,
                spawn_str
            ])

            if args.maze_id == 2:
                print(f"  Ep {ep+1:2d} | Rew: {ep_rew:8.1f} | Crash: {int(done)} | Spawn: {spawn_str}")
            else:
                print(f"  Ep {ep+1:2d} | Rew: {ep_rew:8.1f} | Crash: {int(done)}")

    env.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
