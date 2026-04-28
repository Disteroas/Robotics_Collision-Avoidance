"""
train.py  –  Training DDQN con curriculum multi-labirinto.

Argomenti CLI (gestiti da start_training_curriculum.sh):
  --start-ep    INT   Primo episodio globale del blocco (0-based, incluso)
  --end-ep      INT   Ultimo episodio globale del blocco (0-based, escluso)
  --maze-id     INT   ID labirinto corrente (solo per logging)
  --checkpoint  STR   Path file checkpoint .pkl (carica se esiste, salva sempre)

Calibrazione epsilon:
  Con BETA_DECAY=0.988 e EPISODES_PER_BLOCK=100:
    ε dopo 100 ep = 0.988^100 = 0.300  ← exploitation inizia nel blocco
    ε dopo 240 ep = 0.050               → minimo raggiunto

  Formula: β = 0.30^(1/100) = 0.988

Nota su GAMMA:
  GAMMA rimane 0.99. Orizzonte = 1/(1-0.99) = 100 step.
  NON usare 0.999: orizzonte = 1000 step > MAX_STEPS, Q-values esplodono.

Checkpoint:
  Salvato ogni 20 episodi e alla fine esatta del blocco.
  Perdita massima in caso di crash: 20 episodi.
  Risparmio I/O: 20x rispetto al salvataggio ogni episodio
  (il buffer da 100k serializzato pesa ~40 MB).
"""

import argparse
import csv
import os
import pickle
import random
import shutil
import signal
import sys
from collections import deque

import numpy as np
import rclpy
import torch
import torch.nn as nn
import torch.optim as optim

from ddqn_model import DDQN, ACTION_DIM
from usv_env import UsvEnv

# ────────────────────────────────────────────────────────────────
GAMMA               = 0.99
LR                  = 0.00025
MEMORY_CAPACITY     = 100_000
BATCH_SIZE          = 64
MAX_STEPS           = 500
BETA_DECAY          = 0.988     # ε=0.30 dopo 100 ep → exploitation nel blocco
EPSILON_START       = 1.0
EPSILON_MIN         = 0.05
TARGET_UPDATE_STEPS = 1_000
# ────────────────────────────────────────────────────────────────


# ════════════════════════════════════════════
# REPLAY BUFFER
# ════════════════════════════════════════════
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, r, s2, done):
        self.buffer.append((s, a, r, s2, done))

    def sample(self, n):
        batch = random.sample(self.buffer, n)
        s, a, r, s2, d = map(np.stack, zip(*batch))
        return s, a, r, s2, d

    def __len__(self):
        return len(self.buffer)


# ════════════════════════════════════════════
# AGENTE
# ════════════════════════════════════════════
class DDQNAgent:
    def __init__(self):
        self.q_net      = DDQN()
        self.target_net = DDQN()
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer   = optim.Adam(self.q_net.parameters(), lr=LR)
        self.memory      = ReplayBuffer(MEMORY_CAPACITY)
        self.loss_fn     = nn.MSELoss()
        self.epsilon     = EPSILON_START
        self.total_steps = 0

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, ACTION_DIM - 1)
        with torch.no_grad():
            return int(self.q_net(
                torch.FloatTensor(state).unsqueeze(0)
            ).argmax(dim=1).item())

    def learn(self):
        if len(self.memory) < BATCH_SIZE:
            return None
        s, a, r, s2, d = self.memory.sample(BATCH_SIZE)
        s  = torch.FloatTensor(s)
        a  = torch.LongTensor(a).unsqueeze(1)
        r  = torch.FloatTensor(r).unsqueeze(1)
        s2 = torch.FloatTensor(s2)
        d  = torch.FloatTensor(d.astype(np.float32)).unsqueeze(1)

        with torch.no_grad():
            a2       = self.q_net(s2).argmax(1, keepdim=True)
            target_q = r + (1 - d) * GAMMA * self.target_net(s2).gather(1, a2)

        loss = self.loss_fn(self.q_net(s).gather(1, a), target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()
        return float(loss.item())

    def step_done(self):
        self.total_steps += 1
        if self.total_steps % TARGET_UPDATE_STEPS == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(EPSILON_MIN, self.epsilon * BETA_DECAY)


# ════════════════════════════════════════════
# CHECKPOINT
# ════════════════════════════════════════════
def save_ckpt(agent, episode, rh, crashes, path):
    """Salvataggio atomico tmp→rename. Salva il buffer intero (anti-forgetting)."""
    data = {
        'episode':        episode,
        'q_net':          agent.q_net.state_dict(),
        'target_net':     agent.target_net.state_dict(),
        'optimizer':      agent.optimizer.state_dict(),
        'epsilon':        agent.epsilon,
        'total_steps':    agent.total_steps,
        'replay_buffer':  list(agent.memory.buffer),
        'reward_history': list(rh),
        'crashes':        crashes,
    }
    tmp = path + '.tmp'
    with open(tmp, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    shutil.move(tmp, path)


def load_ckpt(agent, path, rh):
    if not os.path.exists(path):
        return 0, 0
    print(f"  📂 Checkpoint: {path}")
    with open(path, 'rb') as f:
        d = pickle.load(f)
    agent.q_net.load_state_dict(d['q_net'])
    agent.target_net.load_state_dict(d['target_net'])
    agent.optimizer.load_state_dict(d['optimizer'])
    agent.epsilon       = d['epsilon']
    agent.total_steps   = d['total_steps']
    agent.memory.buffer = deque(d['replay_buffer'], maxlen=MEMORY_CAPACITY)
    rh.extend(d.get('reward_history', []))
    ep      = d['episode']
    crashes = d.get('crashes', 0)
    print(f"  ↳ Ep:{ep} | ε:{agent.epsilon:.3f} | "
          f"Buffer:{len(agent.memory.buffer)} | Crash:{crashes}")
    return ep, crashes


# ════════════════════════════════════════════
# ARGOMENTI CLI
# ════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--start-ep',   type=int, default=0)
    p.add_argument('--end-ep',     type=int, default=3000)
    p.add_argument('--maze-id',    type=int, default=1)
    p.add_argument('--checkpoint', type=str,
                   default='src/my_usv/scripts/checkpoint.pkl')
    return p.parse_args()


# ════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════
def main():
    args = parse_args()

    out_dir   = os.path.dirname(os.path.abspath(args.checkpoint))
    log_path  = os.path.join(out_dir, 'training_log.csv')
    best_path = os.path.join(out_dir, 'best_ddqn_model.pth')

    rclpy.init()
    env   = UsvEnv()
    agent = DDQNAgent()
    rh    = deque(maxlen=100)

    last_ep, crashes = load_ckpt(agent, args.checkpoint, rh)

    if last_ep >= args.end_ep:
        print(f"  Blocco {args.start_ep}-{args.end_ep} già completato.")
        env.destroy_node(); rclpy.shutdown(); return

    ep_start = max(last_ep, args.start_ep)
    best_avg = -float('inf')
    total_ep = 3000

    is_new = not os.path.exists(log_path) or os.path.getsize(log_path) == 0
    csv_f  = open(log_path, 'a', newline='', encoding='utf-8')
    csv_w  = csv.writer(csv_f)
    if is_new:
        csv_w.writerow([
            'ep_global', 'maze', 'steps', 'reward',
            'avg100', 'epsilon', 'avg_loss', 'crashed',
            'total_steps', 'total_crashes'
        ])

    # Salvataggio su Ctrl+C / SIGTERM
    _ep = [ep_start]; _cr = [crashes]
    def _exit(sig, frame):
        print(f'\n  ⚠️  Segnale {sig}. Salvo ep={_ep[0]}...')
        save_ckpt(agent, _ep[0], rh, _cr[0], args.checkpoint)
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

        state  = env.reset_environment()
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
            torch.save(agent.q_net.state_dict(), best_path)

        status = '💥 CRASH' if done else '✅ OK   '
        pct    = int(ep_disp / total_ep * 20)
        bar    = '█' * pct + '░' * (20 - pct)
        print(
            f"Ep {ep_disp:4d}/{total_ep} [M{args.maze_id}] {status} | "
            f"R:{ep_rew:8.1f} | avg100:{avg100:8.1f} | "
            f"ε:{agent.epsilon:.3f} | loss:{avg_loss:.4f} | "
            f"crash:{crashes} [{bar}]"
        )

        csv_w.writerow([
            ep_disp, args.maze_id, steps + 1,
            round(ep_rew, 2), round(avg100, 2),
            round(agent.epsilon, 4), round(avg_loss, 6),
            int(done), agent.total_steps, crashes
        ])
        csv_f.flush()

        # Salva ogni 20 episodi o alla fine esatta del blocco
        if ep_disp % 20 == 0 or (offset + 1) == (args.end_ep - ep_start):
            save_ckpt(agent, ep_disp, rh, crashes, args.checkpoint)

    print(f"\n  ✅ Blocco M{args.maze_id} completato. avg100={float(np.mean(rh)):.1f}")
    csv_f.close()
    env.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()