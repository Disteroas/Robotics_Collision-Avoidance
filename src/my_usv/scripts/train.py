import rclpy
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

from usv_env import UsvEnv
from ddqn_model import DDQN, STATE_DIM, ACTION_DIM

# ============================================================ #
#  IPERPARAMETRI                                                #
# ============================================================ #
GAMMA             = 0.99
LR                = 0.00025
MEMORY_CAPACITY   = 10_000
BATCH_SIZE        = 64
MAX_EPISODES      = 3_000

# FIX: MAX_STEPS ridotto da 3000 → 500
# Con 3000 step il reward massimo per episodio era 15.000, rendendo il
# -1000 di crash irrilevante. A 500 step il massimo è 2.500, quindi
# crashare costa molto di più relativamente all'accumulato.
MAX_STEPS         = 500

# Decadimento epsilon (formula paper: ε_{k+1} = β · ε_k)
EPSILON_START     = 1.0
BETA_DECAY        = 0.999
EPSILON_MIN       = 0.05

# FIX: aggiornamento rete target basato su STEP, non su episodi
# "ogni 5 episodi" era instabile: episodi lunghissimi → target obsoleto,
# episodi brevissimi → target aggiornato troppo spesso (overfitting target).
# 1000 step = ~100 secondi di esperienza reale → valore standard DDQN.
TARGET_UPDATE_STEPS = 1_000


# ============================================================ #
#  REPLAY BUFFER                                                #
# ============================================================ #
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


# ============================================================ #
#  AGENTE DDQN                                                  #
# ============================================================ #
class DDQNAgent:
    def __init__(self):
        self.q_net      = DDQN()
        self.target_net = DDQN()
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()  # la rete target non viene mai allenata direttamente

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=LR)
        self.memory    = ReplayBuffer(MEMORY_CAPACITY)
        self.loss_fn   = nn.MSELoss()
        self.epsilon   = EPSILON_START
        self.total_steps = 0  # contatore globale di step per target update

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, ACTION_DIM - 1)
        state_t = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            return int(self.q_net(state_t).argmax(dim=1).item())

    def train_step(self):
        if len(self.memory) < BATCH_SIZE:
            return None

        state, action, reward, next_state, done = self.memory.sample(BATCH_SIZE)

        state      = torch.FloatTensor(state)
        action     = torch.LongTensor(action).unsqueeze(1)
        reward     = torch.FloatTensor(reward).unsqueeze(1)
        next_state = torch.FloatTensor(next_state)
        done       = torch.FloatTensor(done.astype(np.float32)).unsqueeze(1)

        # Logica DDQN:
        #   1. La Q-net online sceglie l'azione migliore per next_state
        #   2. La target net VALUTA quella stessa azione (disaccoppia selection/evaluation)
        with torch.no_grad():
            next_actions = self.q_net(next_state).argmax(dim=1, keepdim=True)
            target_q     = reward + (1.0 - done) * GAMMA * \
                           self.target_net(next_state).gather(1, next_actions)

        current_q = self.q_net(state).gather(1, action)
        loss      = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()

        # FIX: gradient clipping → evita esplosione dei gradienti,
        # problema frequente con DDQN quando i reward hanno scale molto
        # diverse (+5 vs -1000).
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10.0)

        self.optimizer.step()
        return float(loss.item())

    def step_done(self):
        """Chiamato dopo ogni step di ambiente: incrementa contatore e aggiorna target se necessario."""
        self.total_steps += 1
        if self.total_steps % TARGET_UPDATE_STEPS == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
            return True  # segnala che è avvenuto un aggiornamento
        return False

    def decay_epsilon(self):
        """Formula paper: ε_{k+1} = β · ε_k (decadimento esponenziale per episodio)."""
        self.epsilon = max(EPSILON_MIN, self.epsilon * BETA_DECAY)


# ============================================================ #
#  MAIN                                                         #
# ============================================================ #
def main(args=None):
    rclpy.init(args=args)
    env   = UsvEnv()
    agent = DDQNAgent()

    # Statistiche per monitoraggio
    collision_count   = 0
    total_steps_all   = 0
    best_avg_reward   = -float('inf')
    reward_history    = deque(maxlen=100)  # finestra mobile ultimi 100 episodi

    print('\n' + '='*60)
    print('  DDQN COLLISION AVOIDANCE – TRAINING')
    print('  Bug fix: ghost collision + reward shaping + grad clip')
    print('='*60 + '\n')

    for episode in range(MAX_EPISODES):
        state         = env.reset_environment()
        episode_reward = 0.0
        episode_loss  = []
        crashed       = False

        for step in range(MAX_STEPS):
            action_idx = agent.select_action(state)
            next_state, reward, done = env.step_action(action_idx)

            agent.memory.push(state, action_idx, reward, next_state, done)
            loss = agent.train_step()
            if loss is not None:
                episode_loss.append(loss)

            # FIX: aggiornamento target basato su step totali
            target_updated = agent.step_done()

            state          = next_state
            episode_reward += reward
            total_steps_all += 1

            if done:
                crashed = True
                collision_count += 1
                break

        agent.decay_epsilon()

        reward_history.append(episode_reward)
        avg_reward = np.mean(reward_history)
        avg_loss   = np.mean(episode_loss) if episode_loss else 0.0

        # Salva il modello migliore in base alla media mobile
        if len(reward_history) == 100 and avg_reward > best_avg_reward:
            best_avg_reward = avg_reward
            torch.save(agent.q_net.state_dict(), 'best_ddqn_model.pth')

        # Log compatto ma informativo
        status = '💥 CRASH' if crashed else '✅ OK   '
        print(
            f'Ep {episode+1:4d}/{MAX_EPISODES} | {status} | '
            f'Reward: {episode_reward:8.1f} | '
            f'AvgR(100): {avg_reward:8.1f} | '
            f'ε: {agent.epsilon:.3f} | '
            f'Loss: {avg_loss:.4f} | '
            f'Crashes: {collision_count}'
        )

    print(f'\nTraining completato. Miglior media (100 ep): {best_avg_reward:.1f}')
    env.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
