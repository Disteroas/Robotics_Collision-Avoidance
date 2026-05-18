"""
Logica pura del training DDQN: nessuna dipendenza da ROS2.
Importabile nei test senza rclpy attivo.
"""
import os
import pickle
import random
import shutil
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from ddqn_model import DDQN, ACTION_DIM

# ────────────────────────────────────────────────────────────────
GAMMA               = 0.99
LR                  = 0.00025
MEMORY_CAPACITY     = 100_000
BATCH_SIZE          = 64
BETA_DECAY          = 0.999
EPSILON_START       = 1.0
EPSILON_MIN         = 0.05
TARGET_UPDATE_STEPS = 5000   # <-- TORNATO A 5000
REPLAY_START_SIZE   = 10_000
# ────────────────────────────────────────────────────────────────

def set_seed(seed=42):
    """Imposta i seed per garantire la riproducibilità."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"  🌱 Seed globale impostato a: {seed}")

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
        if len(self.memory) < REPLAY_START_SIZE:
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
        # Hard Update ogni 5000 step (come da letteratura DQN/DDQN)
        if self.total_steps % TARGET_UPDATE_STEPS == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(EPSILON_MIN, self.epsilon * BETA_DECAY)


def save_ckpt(agent, episode, rh, crashes, path, best_avg=-float('inf')):
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
        'best_avg':       best_avg,
    }
    tmp = path + '.tmp'
    with open(tmp, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    shutil.move(tmp, path)


def load_ckpt(agent, path, rh):
    if not os.path.exists(path):
        return 0, 0, -float('inf')
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
    ep        = d['episode']
    crashes   = d.get('crashes', 0)
    best_avg  = d.get('best_avg', -float('inf'))
    print(f"  ↳ Ep:{ep} | ε:{agent.epsilon:.3f} | "
          f"Buffer:{len(agent.memory.buffer)} | Crash:{crashes} | "
          f"best_avg:{best_avg:.1f}")
    return ep, crashes, best_avg
