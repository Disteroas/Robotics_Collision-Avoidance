"""
train.py  –  Training DDQN multi-maze interleaved.

Argomenti CLI (gestiti da start_train_multimaze.sh):
  --start-ep    INT   Primo episodio globale del blocco (0-based, incluso)
  --end-ep      INT   Ultimo episodio globale del blocco (0-based, escluso)
  --maze-id     INT   ID labirinto corrente (1 o 2)
  --checkpoint  STR   Path file checkpoint .pkl (carica se esiste, salva sempre)
  --total-ep    INT   Episodi totali del training (default 5000, per progress bar)

Calibrazione epsilon:
  Con BETA_DECAY=0.999 e 4000 episodi:
    ε dopo 1000 ep = 0.999^1000 = 0.368
    ε dopo 3000 ep = 0.050               → minimo raggiunto

Nota su GAMMA:
  GAMMA rimane 0.99. Orizzonte = 1/(1-0.99) = 100 step.
  NON usare 0.999: orizzonte = 1000 step > MAX_STEPS, Q-values esplodono.

Checkpoint:
  Salvato ogni 20 episodi e alla fine esatta del blocco.
  Perdita massima in caso di crash: 20 episodi.
"""

import argparse
import csv
import os
import signal
import sys
from collections import deque

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
    p.add_argument('--checkpoint', type=str,
                   default='src/my_usv/scripts/checkpoint.pkl')
    p.add_argument('--total-ep',   type=int, default=4000)
    return p.parse_args()


def main():
    args = parse_args()
    
    # 2. Richiama il seed prima di inizializzare qualsiasi altra cosa
    set_seed(42)

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
        env.destroy_node(); rclpy.shutdown(); return

    ep_start = max(last_ep, args.start_ep)
    total_ep = args.total_ep

    is_new = not os.path.exists(log_path) or os.path.getsize(log_path) == 0
    csv_f  = open(log_path, 'a', newline='', encoding='utf-8')
    csv_w  = csv.writer(csv_f)
    if is_new:
        csv_w.writerow([
            'ep_global', 'maze', 'steps', 'reward',
            'avg100', 'epsilon', 'avg_loss', 'crashed',
            'total_steps', 'total_crashes', 'spawn'
        ])

    _ep = [ep_start]; _cr = [crashes]
    def _exit(sig, frame):
        print(f'\n  ⚠️  Segnale {sig}. Salvo ep={_ep[0]}...')
        save_ckpt(agent, _ep[0], rh, _cr[0], args.checkpoint, best_avg)
        csv_f.close()
        try: env.destroy_node(); rclpy.shutdown()
        except: pass
        sys.exit(0)
    signal.signal(signal.SIGTERM, _exit)
    signal.signal(signal.SIGINT,  _exit)

    print(f"\n  🏁 Blocco M{args.maze_id} | "
          f"ep {ep_start+1}→{args.end_ep} | "
          f"ε={agent.epsilon:.3f} | buffer={len(agent.memory)}")

    for offset in range(args.end_ep - ep_start):
        ep_global = ep_start + offset
        _ep[0]    = ep_global

        state  = env.reset_environment(maze_id=args.maze_id)
        sx, sy, _ = env.last_spawn
        spawn_label = f"({sx:.1f},{sy:.1f})"
        ep_rew = 0.0
        losses = []
        done   = False
        steps  = 0

        for steps in range(MAX_STEPS):
            a             = agent.act(state)
            ns, rew, done = env.step_action(a)
            agent.memory.push(state, a, rew, ns, done)
            loss = agent.learn()
            if loss is not None:
                losses.append(loss)
            agent.step_done()
            if not _prefill_done[0] and len(agent.memory) >= REPLAY_START_SIZE:
                print(f"\n  ✅ PREFILL completato: {len(agent.memory)} transizioni. Training avviato.\n")
                _prefill_done[0] = True
            state  = ns
            ep_rew += rew
            if done:
                crashes += 1; _cr[0] = crashes
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
            f"sp:{spawn_label} | "
            f"R:{ep_rew:8.1f} | avg100:{avg100:8.1f} | "
            f"ε:{agent.epsilon:.3f} | loss:{avg_loss:.4f} | "
            f"crash:{crashes} [{bar}]"
        )

        csv_w.writerow([
            ep_disp, args.maze_id, steps + 1,
            round(ep_rew, 2), round(avg100, 2),
            round(agent.epsilon, 4), round(avg_loss, 6),
            int(done), agent.total_steps, crashes, spawn_label
        ])
        csv_f.flush()

        if ep_disp % 20 == 0 or (offset + 1) == (args.end_ep - ep_start):
            save_ckpt(agent, ep_disp, rh, crashes, args.checkpoint, best_avg)

    print(f"\n  ✅ Blocco M{args.maze_id} completato. avg100={float(np.mean(rh)):.1f}")
    csv_f.close()
    env.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
