"""
train.py  –  Training DDQN con supporto curriculum multi-labirinto.

Argomenti CLI (gestiti dall'orchestratore start_training_curriculum.sh):
  --start-ep    INT   Episodio globale di inizio del blocco corrente
  --end-ep      INT   Episodio globale di fine del blocco corrente (escluso)
  --maze-id     INT   ID labirinto corrente (1/2/3), solo per logging
  --checkpoint  STR   Path del file checkpoint .pkl (caricato e salvato)

Il checkpoint contiene:
  - Pesi Q-net e Target-net
  - Stato optimizer
  - Epsilon corrente
  - Total steps globali
  - Replay buffer completo (CRITICO per evitare catastrophic forgetting)
  - Cronologia reward (per la media mobile a 100 ep)
  - Contatore collisioni totali

Eseguito standalone (senza args) → comportamento identico alla versione precedente
con valori di default (tutti 3000 episodi, maze 1, no checkpoint).
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

from ddqn_model import DDQN, ACTION_DIM, STATE_DIM
from usv_env import UsvEnv

# ============================================================ #
#  IPERPARAMETRI                                                #
# ============================================================ #
GAMMA               = 0.99
LR                  = 0.00025
MEMORY_CAPACITY     = 10_000
BATCH_SIZE          = 64
MAX_STEPS           = 500       # step per episodio (vedi note in usv_env.py)
EPSILON_START       = 1.0
BETA_DECAY          = 0.999     # formula paper: ε_{k+1} = β · ε_k
EPSILON_MIN         = 0.05
TARGET_UPDATE_STEPS = 1_000     # aggiornamento target ogni N step globali


# ============================================================ #
#  REPLAY BUFFER                                                #
# ============================================================ #
class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
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
        self.target_net.eval()

        self.optimizer   = optim.Adam(self.q_net.parameters(), lr=LR)
        self.memory      = ReplayBuffer(MEMORY_CAPACITY)
        self.loss_fn     = nn.MSELoss()
        self.epsilon     = EPSILON_START
        self.total_steps = 0

    def select_action(self, state: np.ndarray) -> int:
        if random.random() < self.epsilon:
            return random.randint(0, ACTION_DIM - 1)
        with torch.no_grad():
            return int(self.q_net(
                torch.FloatTensor(state).unsqueeze(0)
            ).argmax(dim=1).item())

    def train_step(self):
        if len(self.memory) < BATCH_SIZE:
            return None

        state, action, reward, next_state, done = self.memory.sample(BATCH_SIZE)

        state      = torch.FloatTensor(state)
        action     = torch.LongTensor(action).unsqueeze(1)
        reward     = torch.FloatTensor(reward).unsqueeze(1)
        next_state = torch.FloatTensor(next_state)
        done       = torch.FloatTensor(done.astype(np.float32)).unsqueeze(1)

        with torch.no_grad():
            next_actions = self.q_net(next_state).argmax(dim=1, keepdim=True)
            target_q = reward + (1.0 - done) * GAMMA * \
                       self.target_net(next_state).gather(1, next_actions)

        current_q = self.q_net(state).gather(1, action)
        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10.0)
        self.optimizer.step()
        return float(loss.item())

    def step_done(self) -> bool:
        """Incrementa il contatore e aggiorna la target net se necessario."""
        self.total_steps += 1
        if self.total_steps % TARGET_UPDATE_STEPS == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
            return True
        return False

    def decay_epsilon(self):
        self.epsilon = max(EPSILON_MIN, self.epsilon * BETA_DECAY)


# ============================================================ #
#  CHECKPOINT                                                   #
# ============================================================ #
def save_checkpoint(agent: DDQNAgent,
                    episode: int,
                    reward_history: deque,
                    collision_count: int,
                    path: str) -> None:
    """
    Salva lo stato completo dell'agente su disco.

    Il replay buffer viene salvato integralmente: questo è il meccanismo
    che previene il catastrophic forgetting nel curriculum multi-labirinto.
    Quando si passa da Maze 1 a Maze 2, il buffer contiene ancora le
    esperienze di Maze 1; il network continua ad aggiornarsi su tutti i
    labirinti precedenti tramite le mini-batch di training.

    Scrittura atomica (tmp → rename) per evitare corruzione se il processo
    viene terminato durante il salvataggio.
    """
    data = {
        'episode':          episode,
        'q_net_state':      agent.q_net.state_dict(),
        'target_net_state': agent.target_net.state_dict(),
        'optimizer_state':  agent.optimizer.state_dict(),
        'epsilon':          agent.epsilon,
        'total_steps':      agent.total_steps,
        # CRITICO: salva TUTTO il buffer, incluse esperienze dei labirinti precedenti
        'replay_buffer':    list(agent.memory.buffer),
        'reward_history':   list(reward_history),
        'collision_count':  collision_count,
    }
    tmp_path = path + '.tmp'
    with open(tmp_path, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    shutil.move(tmp_path, path)


def load_checkpoint(agent: DDQNAgent,
                    path: str,
                    reward_history: deque) -> tuple[int, int]:
    """
    Carica checkpoint. Restituisce (last_episode, collision_count).
    Se il file non esiste, restituisce (0, 0) (training from scratch).
    """
    if not os.path.exists(path):
        return 0, 0

    print(f"📂 Checkpoint trovato: {path}")
    with open(path, 'rb') as f:
        data = pickle.load(f)

    agent.q_net.load_state_dict(data['q_net_state'])
    agent.target_net.load_state_dict(data['target_net_state'])
    agent.optimizer.load_state_dict(data['optimizer_state'])
    agent.epsilon        = data['epsilon']
    agent.total_steps    = data['total_steps']
    agent.memory.buffer  = deque(data['replay_buffer'], maxlen=MEMORY_CAPACITY)
    reward_history.extend(data.get('reward_history', []))

    last_ep       = data['episode']
    collision_cnt = data.get('collision_count', 0)

    buf_size = len(agent.memory.buffer)
    print(f"   ↳ Episodio: {last_ep} | ε: {agent.epsilon:.3f} | "
          f"Buffer: {buf_size} esperienze | Collisioni: {collision_cnt}")
    return last_ep, collision_cnt


# ============================================================ #
#  ARGOMENTI CLI                                                #
# ============================================================ #
def parse_args():
    p = argparse.ArgumentParser(description='DDQN Collision Avoidance Training')
    p.add_argument('--start-ep',   type=int,   default=0,
                   help='Episodio globale di inizio del blocco (default: 0)')
    p.add_argument('--end-ep',     type=int,   default=3000,
                   help='Episodio globale di fine del blocco escluso (default: 3000)')
    p.add_argument('--maze-id',    type=int,   default=1,
                   help='ID labirinto corrente per il logging (default: 1)')
    p.add_argument('--checkpoint', type=str,
                   default='src/my_usv/scripts/checkpoint.pkl',
                   help='Path del checkpoint da caricare/salvare')
    return p.parse_args()


# ============================================================ #
#  MAIN                                                         #
# ============================================================ #
def main():
    args = parse_args()

    # Percorsi file di output (nella stessa directory del checkpoint)
    out_dir  = os.path.dirname(os.path.abspath(args.checkpoint))
    log_path = os.path.join(out_dir, 'training_log.csv')
    best_model_path = os.path.join(out_dir, 'best_ddqn_model.pth')

    # Inizializza ROS2 e ambiente
    rclpy.init()
    env   = UsvEnv()
    agent = DDQNAgent()

    # Cronologia reward (persiste nel checkpoint per avg mobile globale)
    reward_history = deque(maxlen=100)
    best_avg_reward = -float('inf')

    # -------------------------------------------------------- #
    # Carica checkpoint (se esiste)                             #
    # -------------------------------------------------------- #
    last_completed_ep, collision_count = load_checkpoint(
        agent, args.checkpoint, reward_history
    )

    # Determina da quale episodio (globale) partire in questo blocco
    # Caso 1: checkpoint già oltre end_ep → questo blocco è già finito
    if last_completed_ep >= args.end_ep:
        print(f"✅ Blocco {args.start_ep}-{args.end_ep} già completato "
              f"(checkpoint ep={last_completed_ep}). Uscita.")
        env.destroy_node()
        rclpy.shutdown()
        return

    # Caso 2: checkpoint è nel mezzo del blocco → riprendi
    global_ep_start = max(last_completed_ep, args.start_ep)
    episodes_remaining = args.end_ep - global_ep_start

    # -------------------------------------------------------- #
    # Gestione segnali per salvataggio in caso di interruzione  #
    # -------------------------------------------------------- #
    # Variabili usate nel signal handler (closure)
    _current_ep   = [global_ep_start]
    _current_rh   = reward_history
    _current_cc   = [collision_count]
    _agent        = agent
    _checkpoint   = args.checkpoint

    def _on_signal(signum, frame):
        print(f'\n⚠️  Segnale {signum} ricevuto. Salvo checkpoint ep={_current_ep[0]}...')
        save_checkpoint(_agent, _current_ep[0], _current_rh,
                        _current_cc[0], _checkpoint)
        print('   ↳ Checkpoint salvato. Uscita.')
        try:
            env.destroy_node()
            rclpy.shutdown()
        except Exception:
            pass
        sys.exit(0)

    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT,  _on_signal)

    # -------------------------------------------------------- #
    # Intestazione CSV (crea file se non esiste)                #
    # -------------------------------------------------------- #
    csv_is_new = not os.path.exists(log_path) or os.path.getsize(log_path) == 0
    log_file   = open(log_path, 'a', newline='', encoding='utf-8')
    log_writer = csv.writer(log_file)
    if csv_is_new:
        log_writer.writerow([
            'episode', 'maze_id', 'steps', 'reward',
            'avg_reward_100', 'epsilon', 'avg_loss',
            'crashed', 'total_steps', 'collision_count'
        ])

    # -------------------------------------------------------- #
    # Intestazione blocco su stdout                             #
    # -------------------------------------------------------- #
    total_blocks = (args.end_ep - args.start_ep)
    print('\n' + '═' * 66)
    print(f'  🏁 BLOCCO CURRICULUM | Labirinto {args.maze_id} | '
          f'Ep {global_ep_start + 1} → {args.end_ep}')
    print(f'  Episodi rimanenti in questo blocco: {episodes_remaining}')
    print('═' * 66)

    # -------------------------------------------------------- #
    # Loop di training                                          #
    # -------------------------------------------------------- #
    for offset in range(episodes_remaining):
        global_ep = global_ep_start + offset  # episodio globale corrente
        _current_ep[0] = global_ep

        state          = env.reset_environment()
        episode_reward = 0.0
        episode_losses = []
        crashed        = False

        for step in range(MAX_STEPS):
            action_idx = agent.select_action(state)
            next_state, reward, done = env.step_action(action_idx)

            agent.memory.push(state, action_idx, reward, next_state, done)
            loss = agent.train_step()
            if loss is not None:
                episode_losses.append(loss)

            agent.step_done()
            state          = next_state
            episode_reward += reward

            if done:
                crashed = True
                collision_count += 1
                _current_cc[0]  = collision_count
                break

        agent.decay_epsilon()

        reward_history.append(episode_reward)
        avg_reward = float(np.mean(reward_history))
        avg_loss   = float(np.mean(episode_losses)) if episode_losses else 0.0

        # Salva il miglior modello (basato sulla media mobile globale)
        if len(reward_history) >= 10 and avg_reward > best_avg_reward:
            best_avg_reward = avg_reward
            torch.save(agent.q_net.state_dict(), best_model_path)

        # ---- Stampa su console ------------------------------------ #
        status      = '💥 CRASH' if crashed else '✅ OK   '
        global_disp = global_ep + 1          # 1-based per l'utente
        total_ep    = args.end_ep            # totale campagna
        # Barra progresso globale (mini)
        pct = int(global_disp / total_ep * 20)
        bar = '█' * pct + '░' * (20 - pct)

        print(
            f'Ep {global_disp:4d}/{total_ep} [M{args.maze_id}] {status} | '
            f'R: {episode_reward:8.1f} | '
            f'AvgR(100): {avg_reward:8.1f} | '
            f'ε: {agent.epsilon:.3f} | '
            f'Loss: {avg_loss:.4f} | '
            f'Crash: {collision_count} | '
            f'[{bar}] {global_disp/total_ep*100:.1f}%'
        )

        # ---- CSV log ---------------------------------------------- #
        log_writer.writerow([
            global_disp, args.maze_id, step + 1,
            round(episode_reward, 2), round(avg_reward, 2),
            round(agent.epsilon, 4), round(avg_loss, 6),
            int(crashed), agent.total_steps, collision_count
        ])
        log_file.flush()

        # ---- Salva checkpoint ogni episodio ----------------------- #
        # Scrittura atomica: in caso di crash perdiamo AL MASSIMO 1 episodio.
        save_checkpoint(
            agent, global_disp, reward_history,
            collision_count, args.checkpoint
        )

    # -------------------------------------------------------- #
    # Fine blocco                                               #
    # -------------------------------------------------------- #
    print('\n' + '─' * 66)
    print(f'  ✅ Blocco completato | Labirinto {args.maze_id} | '
          f'Ep {global_ep_start + 1}–{args.end_ep}')
    print(f'  Media reward (ultimi 100): {float(np.mean(reward_history)):.1f}')
    print(f'  Miglior media globale: {best_avg_reward:.1f}')
    print('─' * 66 + '\n')

    log_file.close()
    env.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()