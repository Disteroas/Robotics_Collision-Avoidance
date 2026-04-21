import torch
import torch.nn as nn
import torch.nn.functional as F

# Secondo il paper: 50 raggi LIDAR in input, 11 azioni discrete in output
STATE_DIM = 50
ACTION_DIM = 11

class DDQN(nn.Module):
    def __init__(self, state_dim=STATE_DIM, action_dim=ACTION_DIM):
        super(DDQN, self).__init__()
        # Il paper usa 2 hidden layers da 300 neuroni ciascuno con attivazione ReLU
        self.fc1 = nn.Linear(state_dim, 300)
        self.fc2 = nn.Linear(300, 300)
        self.fc3 = nn.Linear(300, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        # Sputa fuori 11 numeri: il Q-Value (punteggio previsto) per ciascuna delle 11 sterzate
        q_values = self.fc3(x)
        return q_values