import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os
from ddqn_model import DDQN
from collections import deque

BATCH_SIZE         = 64
GAMMA              = 0.99
LR                 = 0.00025
REPLAY_BUFFER_SIZE = 50000
REPLAY_START_SIZE  = 5000
BETA_DECAY         = 0.999
EPSILON_MIN        = 0.05
GRAD_CLIP_NORM     = 1.0    # FIX: era 10.0. Lo spike a ~3500 loss a ep 1100
                             # con clip=10 dimostra che 10 non protegge abbastanza.
                             # Con clip=1 i gradienti vengono contenuti efficacemente
                             # senza rallentare la convergenza su questo task.


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class DDQNAgent:
    def __init__(self):
        self.q_net       = DDQN()
        self.target_net  = DDQN()
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer   = optim.Adam(self.q_net.parameters(), lr=LR)
        self.memory      = deque(maxlen=REPLAY_BUFFER_SIZE)
        self.epsilon     = 1.0
        self.total_steps = 0

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, 10)
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_net(state)
        return q_values.argmax().item()

    def learn(self):
        if len(self.memory) < REPLAY_START_SIZE:
            return None

        batch = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)

        states      = torch.nan_to_num(torch.FloatTensor(np.array(states)),      nan=1.0, posinf=1.0)
        actions     = torch.LongTensor(np.array(actions)).unsqueeze(1)
        rewards     = torch.FloatTensor(np.array(rewards)).unsqueeze(1)
        next_states = torch.nan_to_num(torch.FloatTensor(np.array(next_states)), nan=1.0, posinf=1.0)
        dones       = torch.FloatTensor(np.array(dones)).unsqueeze(1)

        current_q = self.q_net(states).gather(1, actions)

        with torch.no_grad():
            next_actions = self.q_net(next_states).argmax(1, keepdim=True)
            next_q       = self.target_net(next_states).gather(1, next_actions)
            target       = rewards + (GAMMA * next_q * (1 - dones))

        loss = nn.MSELoss()(current_q, target)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), GRAD_CLIP_NORM)
        self.optimizer.step()

        return loss.item()

    def step_done(self):
        self.total_steps += 1
        if self.total_steps % 500 == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(EPSILON_MIN, self.epsilon * BETA_DECAY)


def save_ckpt(agent, ep, rh, crashes, path, best_avg):
    import pickle
    ckpt = {
        'model_state':     agent.q_net.state_dict(),
        'target_state':    agent.target_net.state_dict(),
        'optimizer_state': agent.optimizer.state_dict(),
        'epsilon':         agent.epsilon,
        'total_steps':     agent.total_steps,
        'ep':              ep,
        'rh':              list(rh),
        'crashes':         crashes,
        'best_avg':        best_avg,
    }
    with open(path, 'wb') as f:
        pickle.dump(ckpt, f)


def load_ckpt(agent, path, rh):
    import pickle
    if not os.path.exists(path):
        return 0, 0, -float('inf')
    with open(path, 'rb') as f:
        ckpt = pickle.load(f)

    agent.q_net.load_state_dict(ckpt['model_state'])
    agent.target_net.load_state_dict(
        ckpt.get('target_state', ckpt['model_state'])
    )
    if 'optimizer_state' in ckpt:
        agent.optimizer.load_state_dict(ckpt['optimizer_state'])
    agent.total_steps = ckpt.get('total_steps', 0)
    agent.epsilon     = ckpt['epsilon']
    rh.extend(ckpt['rh'])
    return ckpt['ep'], ckpt['crashes'], ckpt['best_avg']
