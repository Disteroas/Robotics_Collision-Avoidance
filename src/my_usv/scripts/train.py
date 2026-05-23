"""
train.py — Training DDQN con seed control e backup CSV automatico.
"""

import argparse
import csv
import os
import shutil
import signal
import sys
from collections import deque
from datetime import datetime

import numpy as np
import rclpy

from usv_env import UsvEnv
from train_core import (
    DDQNAgent, save_ckpt, load_ckpt,
    EPSILON_MIN, BETA_DECAY, REPLAY_START_SIZE,
    set_seed
)

MAX_STEPS = 500


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--start-ep',   type=int, default=0)
    p.add_argument('--end-ep',     type=int, default=3000)
    p.add_argument('--maze-id',    type=int, default=1)
    p.add_argument('--checkpoint', type=str, default='src/my_usv/scripts/checkpoint.pkl')
    p.add_argument('--total-ep',   type=int, default=4000)
    # Seed control: ogni run deve usare un seed esplicito da CLI.
    # Non usare mai il default 42 per run "ufficiali": specificarlo sempre
    # esplicitamente in start_train_multimaze.sh per avere seed tracciati.
    p.add_argument('--seed',       type=int, default=42,
                   help='Seed per random/numpy/torch. Usare valori diversi per run multi-seed.')
    return p.parse_args()


def _backup_csv(log_path: str) -> None:
    """
    Copia training_log.csv in ANALISI_BACKUP/ con timestamp prima di ogni reset.
    Lezione appresa: i CSV grezzi di R1 sono andati persi senza backup.
    """
    if not os.path.exists(log_path):
        return
    backup_dir = os.path.join(os.path.dirname(log_path), 'ANALISI_BACKUP')
    os.makedirs(backup_dir, exist_ok=True)
    ts          = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_path = os.path.join(backup_dir, f'training_log_{ts}.csv')
    shutil.copy2(log_path, backup_path)
    print(f"  [BACKUP] CSV salvato in {backup_path}")


def main():
    args = parse_args()

    # Seed control: fissa la randomness controllabile.
    # Gazebo resta non deterministico (timing ROS), ma almeno la rete,
    # il buffer e l'ε-greedy sono riproducibili a parità di seed + PC.
    set_seed(args.seed)
    print(f"  [SEED] seed={args.seed}")

    out_dir   = os.path.dirname(os.path.abspath(args.checkpoint))
    log_path  = os.path.join(out_dir, 'training_log.csv')
    best_path = os.path.join(out_dir, 'best_ddqn_model.pth')

    rclpy.init()
    env   = UsvEnv()
    agent = DDQNAgent()
    rh    = deque(maxlen=100)

    last_ep, crashes, best_avg = load_ckpt(agent, args.checkpoint, rh)
    _prefill_done = [False]

    if last_ep >= args.end_ep:
        print(f"  Blocco {args.start_ep}-{args.end_ep} già completato.")
        env.destroy_node()
        rclpy.shutdown()
        return

    ep_start = max(last_ep, args.start_ep)
    total_ep = args.total_ep

    is_new = not os.path.exists(log_path) or os.path.getsize(log_path) == 0
    csv_f  = open(log_path, 'a', newline='', encoding='utf-8')
    csv_w  = csv.writer(csv_f)
    if is_new:
        csv_w.writerow([
            'ep_global', 'maze', 'steps', 'reward',
            'avg100', 'epsilon', 'avg_loss', 'crashed',
            'total_steps', 'total_crashes', 'spawn', 'seed'
        ])

    _ep = [ep_start]
    _cr = [crashes]

    def _exit(sig, frame):
        print(f'\n  ⚠️  Segnale {sig}. Salvo ep={_ep[0]}...')
        save_ckpt(agent, _ep[0], rh, _cr[0], args.checkpoint, best_avg)
        _backup_csv(log_path)
        csv_f.close()
        try:
            env.destroy_node()
            rclpy.shutdown()
        except Exception:
            pass
        sys.exit(0)

    signal.signal(signal.SIGTERM, _exit)
    signal.signal(signal.SIGINT,  _exit)

    print(f"\n  🏁 Blocco M{args.maze_id} | ep {ep_start + 1}→{args.end_ep} | seed={args.seed}")

    for offset in range(args.end_ep - ep_start):
        ep_global = ep_start + offset
        _ep[0]    = ep_global

        state = env.reset_environment(maze_id=args.maze_id)
        state = np.nan_to_num(state, nan=1.0, posinf=1.0, neginf=0.0)

        sx, sy, _ = env.last_spawn
        spawn_label = f"({sx:.1f},{sy:.1f})"
        ep_rew  = 0.0
        losses  = []
        done    = False
        steps   = 0

        for steps in range(MAX_STEPS):
            a              = agent.act(state)
            ns, rew, done  = env.step_action(a)
            ns             = np.nan_to_num(ns, nan=1.0, posinf=1.0, neginf=0.0)

            agent.memory.append((state, a, rew, ns, done))

            loss = agent.learn()
            if loss is not None:
                losses.append(loss)
            agent.step_done()

            if not _prefill_done[0] and len(agent.memory) >= REPLAY_START_SIZE:
                print(f"\n  ✅ PREFILL completato ({REPLAY_START_SIZE} transizioni).\n")
                _prefill_done[0] = True

            state  = ns
            ep_rew += rew
            if done:
                crashes += 1
                _cr[0]   = crashes
                break

        agent.decay_epsilon()
        rh.append(ep_rew)

        avg100   = float(np.mean(rh))
        avg_loss = float(np.mean(losses)) if losses else 0.0
        ep_disp  = ep_global + 1

        if len(rh) >= 10 and avg100 > best_avg:
            best_avg = avg100
            import torch
            torch.save(agent.q_net.state_dict(), best_path)

        status = '💥 CRASH' if done else '✅ OK   '
        pct    = int(ep_disp / total_ep * 20)
        bar    = '█' * pct + '░' * (20 - pct)

        print(
            f"Ep {ep_disp:4d}/{total_ep} [M{args.maze_id}] {status} | "
            f"Spawn:{spawn_label} | R:{ep_rew:8.1f} | avg100:{avg100:8.1f} | "
            f"ε:{agent.epsilon:.3f} | loss:{avg_loss:.4f} | crash:{crashes} [{bar}]"
        )

        csv_w.writerow([
            ep_disp, args.maze_id, steps + 1,
            round(ep_rew, 2), round(avg100, 2),
            round(agent.epsilon, 4), round(avg_loss, 6),
            int(done), agent.total_steps, crashes,
            spawn_label, args.seed
        ])
        csv_f.flush()

        if ep_disp % 20 == 0 or (offset + 1) == (args.end_ep - ep_start):
            save_ckpt(agent, ep_disp, rh, crashes, args.checkpoint, best_avg)

    _backup_csv(log_path)
    csv_f.close()
    env.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
