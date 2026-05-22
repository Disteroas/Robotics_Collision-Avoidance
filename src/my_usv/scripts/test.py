"""test.py – Valutazione policy DDQN su un singolo maze, con logging rigoroso.

CLI:
  --maze-id     INT   1/2/3
  --model       STR   path best_model.pth
  --reps        INT   ripetizioni per spawn (episodi = n_spawn × reps; default 30)
  --episodes    INT   override opzionale del totale (se >0 ignora --reps)
  --seed        INT   seed globale (default 0)
  --config      STR   etichetta config (default 'default')
  --out-dir     STR   cartella output (default runs/default/seed_0)
  --log-q-full        se presente, logga tutti gli 11 Q-values per step

ε=0.0 (greedy). Successo = MAX_STEPS raggiunti senza collisione.
"""
import argparse
import csv
import os
import sys
from collections import deque

import numpy as np
import rclpy
import torch

from ddqn_model import DDQN
from usv_env import UsvEnv, TEST_SPAWN_LISTS
from usv_logic import sector_distances, crash_sector
from seeding import set_global_seed

MAX_STEPS = 500


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--maze-id',     type=int, default=1)
    p.add_argument('--model',       type=str,
                   default='src/my_usv/scripts/best_ddqn_model.pth')
    p.add_argument('--reps',        type=int, default=30)
    p.add_argument('--episodes',    type=int, default=0)
    p.add_argument('--seed',        type=int, default=0)
    p.add_argument('--config',      type=str, default='default')
    p.add_argument('--out-dir',     type=str, default='runs/default/seed_0')
    p.add_argument('--log-q-full',  action='store_true')
    return p.parse_args()


def main():
    args = parse_args()
    set_global_seed(args.seed)

    if not os.path.exists(args.model):
        print(f"[ERRORE] Modello non trovato: {args.model}")
        sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)
    m = args.maze_id
    steps_path   = os.path.join(args.out_dir, f'eval_steps_m{m}.csv')
    crashes_path = os.path.join(args.out_dir, f'eval_crashes_m{m}.csv')
    summary_path = os.path.join(args.out_dir, 'eval_summary.csv')

    n_spawns = len(TEST_SPAWN_LISTS[m])
    episodes = args.episodes if args.episodes > 0 else n_spawns * args.reps

    q_net = DDQN()
    q_net.load_state_dict(torch.load(args.model, map_location='cpu'))
    q_net.eval()
    print(f"  ✅ Modello: {args.model} | Maze {m} | {episodes} ep "
          f"({n_spawns} spawn × {args.reps} reps) | seed {args.seed}")

    # ── CSV per-step ──────────────────────────────────────────────
    steps_header = ['episode', 'step', 'spawn', 'action',
                    'q_chosen', 'q_max', 'q_spread',
                    'front_dist', 'left_dist', 'right_dist', 'min_lidar',
                    'reward', 'done']
    if args.log_q_full:
        steps_header += [f'q{i}' for i in range(11)]
    steps_f = open(steps_path, 'w', newline='', encoding='utf-8')
    steps_w = csv.writer(steps_f)
    steps_w.writerow(steps_header)

    # ── CSV crash ─────────────────────────────────────────────────
    crashes_f = open(crashes_path, 'w', newline='', encoding='utf-8')
    crashes_w = csv.writer(crashes_f)
    crashes_w.writerow(['episode', 'spawn', 'crash_step',
                        'crash_sector', 'crash_dist', 'last_actions'])

    rclpy.init()
    env = UsvEnv()

    rewards, steps_l, crashes = [], [], 0
    spawn_stats = {}

    for ep in range(1, episodes + 1):
        state = env.reset_environment(maze_id=m, test_mode=True)
        sx, sy, _ = env.last_spawn
        spawn_label = f"({sx:.1f},{sy:.1f})"
        spawn_stats.setdefault(spawn_label, {'total': 0, 'completed': 0})
        ep_reward = 0.0
        ep_steps = 0
        crashed = False
        recent_actions = deque(maxlen=5)

        for step in range(MAX_STEPS):
            with torch.no_grad():
                q = q_net(torch.FloatTensor(state).unsqueeze(0)).squeeze(0)
            action = int(q.argmax().item())
            q_chosen = float(q[action])
            q_max = float(q.max())
            q_spread = q_max - float(q.min())
            recent_actions.append(action)

            state, reward, done = env.step_action(action, training=False)
            sd = env.last_info
            ep_reward += reward
            ep_steps += 1

            row = [ep, step, spawn_label, action,
                   round(q_chosen, 4), round(q_max, 4), round(q_spread, 4),
                   round(sd['front'], 4), round(sd['left'], 4),
                   round(sd['right'], 4), round(sd['min_lidar'], 4),
                   round(reward, 4), int(done)]
            if args.log_q_full:
                row += [round(float(x), 4) for x in q.tolist()]
            steps_w.writerow(row)

            if done:
                crashed = True
                crashes += 1
                sec = crash_sector(sd['front'], sd['left'], sd['right'])
                crashes_w.writerow([
                    ep, spawn_label, step, sec,
                    round(sd['min_lidar'], 4),
                    ','.join(str(a) for a in recent_actions)])
                break

        steps_f.flush()
        crashes_f.flush()
        rewards.append(ep_reward)
        steps_l.append(ep_steps)
        spawn_stats[spawn_label]['total'] += 1
        if not crashed:
            spawn_stats[spawn_label]['completed'] += 1

        esito = '💥 CRASH' if crashed else '✅ OK   '
        print(f"  Ep {ep:>4}/{episodes}  {ep_steps:>4} step  "
              f"R:{ep_reward:>8.1f}  {spawn_label:<14} {esito}")

    n_success = episodes - crashes
    success_rate = n_success / episodes
    avg_reward = float(np.mean(rewards))
    avg_steps = float(np.mean(steps_l))

    # ── eval_summary.csv (append: una riga per maze) ──────────────
    summary_is_new = (not os.path.exists(summary_path) or
                      os.path.getsize(summary_path) == 0)
    with open(summary_path, 'a', newline='', encoding='utf-8') as sf:
        sw = csv.writer(sf)
        if summary_is_new:
            sw.writerow(['config', 'seed', 'maze', 'episodes',
                         'n_success', 'success_rate', 'avg_reward', 'avg_steps'])
        sw.writerow([args.config, args.seed, m, episodes, n_success,
                     round(success_rate, 4), round(avg_reward, 2),
                     round(avg_steps, 1)])

    print(f"\n  📊 Maze {m}: success {success_rate*100:.1f}% "
          f"({n_success}/{episodes}) | reward {avg_reward:.1f} | "
          f"steps {avg_steps:.1f}")
    print("  Per-spawn:")
    for sp, s in sorted(spawn_stats.items(),
                        key=lambda x: -x[1]['completed'] / max(x[1]['total'], 1)):
        rate = s['completed'] / s['total'] * 100
        print(f"    {sp:<14} {s['completed']}/{s['total']}  ({rate:.1f}%)")

    steps_f.close()
    crashes_f.close()
    env.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
