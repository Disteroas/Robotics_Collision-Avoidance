import rclpy
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

from usv_env import UsvEnv
from ddqn_model import DDQN, STATE_DIM, ACTION_DIM

# --- IPERPARAMETRI DAL PAPER ---
GAMMA = 0.99       
LR = 0.00025       
MEMORY_CAPACITY = 10000
BATCH_SIZE = 64
MAX_EPISODES = 3000     # Il paper usa 3000 epoche
MAX_STEPS = 3000         

# Formula di esplorazione dal paper
EPSILON_START = 1.0
BETA_DECAY = 0.999      # Testato dagli autori come il migliore
EPSILON_MIN = 0.05      # Valore minimo di epsilon dal paper

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

class DDQNAgent:
    def __init__(self):
        self.q_net = DDQN()
        self.target_net = DDQN()
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=LR)
        self.memory = ReplayBuffer(MEMORY_CAPACITY)
        self.loss_fn = nn.MSELoss()
        self.epsilon = EPSILON_START

    def select_action(self, state):
        # Policy epsilon-greedy del paper
        if random.random() < self.epsilon:
            return random.randint(0, ACTION_DIM - 1)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        self.q_net.eval()
        with torch.no_grad():
            q_values = self.q_net(state_tensor)
            action = torch.argmax(q_values).item()
        self.q_net.train()
        return action

    def train_step(self):
        if len(self.memory) < BATCH_SIZE:
            return

        state, action, reward, next_state, done = self.memory.sample(BATCH_SIZE)
        
        state = torch.FloatTensor(state)
        action = torch.LongTensor(action).unsqueeze(1)
        reward = torch.FloatTensor(reward).unsqueeze(1)
        next_state = torch.FloatTensor(next_state)
        done = torch.FloatTensor(np.float32(done)).unsqueeze(1)

        # Logica DDQN: Azione scelta dalla rete principale, valore valutato dalla rete Target
        with torch.no_grad():
            next_actions = self.q_net(next_state).argmax(1).unsqueeze(1)
            target_q = reward + (1 - done) * GAMMA * self.target_net(next_state).gather(1, next_actions)

        current_q = self.q_net(state).gather(1, action)
        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        self.target_net.load_state_dict(self.q_net.state_dict())
        
    def decay_epsilon(self):
        # Formula del paper: epsilon_k+1 = beta * epsilon_k
        self.epsilon = max(EPSILON_MIN, self.epsilon * BETA_DECAY)

def main(args=None):
    rclpy.init(args=args)
    env = UsvEnv()
    agent = DDQNAgent()
    
    print("\n🚀 ADDESTRAMENTO DDQN UFFICIALE (COME DA PAPER) 🚀")
    
    for episode in range(MAX_EPISODES):
        state = env.reset_environment()
        episode_reward = 0
        
        for step in range(MAX_STEPS):
            action_idx = agent.select_action(state)
            next_state, reward, done = env.step_action(action_idx)
            
            agent.memory.push(state, action_idx, reward, next_state, done)
            agent.train_step()
            
            state = next_state
            episode_reward += reward
            
            if done:
                break
                
        # Decadimento di Epsilon alla fine dell'episodio
        agent.decay_epsilon()
        
        # Aggiorna la rete target ogni 5 episodi (Iperparametro standard per DDQN)
        if episode % 5 == 0:
            agent.update_target_network()
            
        print(f"Episodio: {episode+1}/{MAX_EPISODES} | Reward: {episode_reward:.1f} | Epsilon: {agent.epsilon:.3f}")

    env.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()