import torch
import torch.nn as nn
import torch.nn.functional as F

# Input: 50 raggi LIDAR normalizzati in [0, 1] (divisi per 5.0 in usv_env.py)
# Output: Q-values per ciascuna delle 11 azioni discrete
STATE_DIM  = 51
ACTION_DIM = 11


class DDQN(nn.Module):
    """
    Rete DDQN come da paper:
      - 2 hidden layers da 300 neuroni con attivazione ReLU
      - Input: stato LIDAR normalizzato [0, 1]  ← FIX: normalizzazione gestita in env
      - Output: Q(s, a) per ognuna delle 11 azioni di sterzo

    Nota: la logica DDQN (Double DQN) non risiede nell'architettura della rete
    ma nel calcolo del target in train.py:
        next_action = argmax_a Q_online(s', a)   ← online net sceglie
        target      = r + γ · Q_target(s', next_action)  ← target net valuta
    Questo disaccoppiamento selection/evaluation riduce il maximization bias
    rispetto al Q-learning classico.
    """

    def __init__(self, state_dim=STATE_DIM, action_dim=ACTION_DIM):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 300)
        self.fc2 = nn.Linear(300, 300)
        self.fc3 = nn.Linear(300, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)  # Q-values (non normalizzati, nessun softmax)
