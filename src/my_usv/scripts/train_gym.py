"""
train_gym.py — DDQN training via gymnasium interface (single maze, no curriculum).

Validates UsvGymEnv and serves as the swap point for other algorithms.
Checkpoints saved to checkpoint_gym.pth (separate from train.py's checkpoint.pkl).

To swap algorithm (e.g. XinJingHao PPO):
    1. Copy algorithm file to src/my_usv/scripts/
    2. Replace the DDQNAgent import below with the new agent class
    3. Ensure the agent exposes: act(state), memory.push(...), learn(), step_done(), decay_epsilon()
"""

import argparse
import os
from collections import deque

import numpy as np

from usv_gym_env import UsvGymEnv
from train_core import DDQNAgent

# ── swap point ────────────────────────────────────────────────────────────────
# from xjh_ddqn import DQN_Agent as DDQNAgent   # XinJingHao DDQN
# from xjh_ppo   import PPO_Agent as DDQNAgent   # XinJingHao PPO
# ─────────────────────────────────────────────────────────────────────────────

SAVE_EVERY = 20


def parse_args():
    p = argparse.ArgumentParser(description='Train DDQN via gymnasium interface')
    p.add_argument('--maze-id',    type=int,  default=1,
                   help='Maze ID for per-episode random spawn (1 or 2)')
    p.add_argument('--episodes',   type=int,  default=3000)
    p.add_argument('--max-steps',  type=int,  default=1000)
    p.add_argument('--continuous', action='store_true', default=False)
    p.add_argument('--checkpoint', type=str,
                   default='src/my_usv/scripts/checkpoint_gym.pth')
    return p.parse_args()


def main():
    args    = parse_args()
    env     = UsvGymEnv(continuous=args.continuous, max_steps=args.max_steps)
    agent   = DDQNAgent()

    rh        = deque(maxlen=100)
    best_avg  = -float('inf')
    out_dir   = os.path.dirname(os.path.abspath(args.checkpoint))
    best_path = os.path.join(out_dir, 'best_gym_model.pth')

    print(f"\n  Training via gymnasium | maze={args.maze_id} | "
          f"episodes={args.episodes} | continuous={args.continuous}\n")

    for ep in range(args.episodes):
        state, _ = env.reset(options={'maze_id': args.maze_id})
        ep_reward = 0.0
        losses    = []

        while True:
            action                                           = agent.act(state)
            next_state, reward, terminated, truncated, info = env.step(action)

            # Pass terminated (not done) to replay buffer.
            # truncated=True means time limit hit — episode is NOT over in the MDP sense,
            # so the value bootstrap should NOT be zeroed.
            agent.memory.push(state, action, reward, next_state, terminated)

            loss = agent.learn()
            if loss is not None:
                losses.append(loss)
            agent.step_done()

            ep_reward += reward
            state      = next_state

            if terminated or truncated:
                break

        agent.decay_epsilon()
        rh.append(ep_reward)
        avg100   = float(np.mean(rh))
        avg_loss = float(np.mean(losses)) if losses else 0.0

        if len(rh) >= 10 and avg100 > best_avg:
            best_avg = avg100
            import torch
            torch.save(agent.q_net.state_dict(), best_path)

        status = 'CRASH' if info['crashed'] else 'OK   '
        print(
            f"Ep {ep+1:4d} [{status}] "
            f"R:{ep_reward:8.1f} | avg100:{avg100:8.1f} | "
            f"steps:{info['steps']:4d} | ε:{agent.epsilon:.3f} | "
            f"loss:{avg_loss:.4f}"
        )

        if (ep + 1) % SAVE_EVERY == 0:
            import torch
            torch.save({
                'q_net':      agent.q_net.state_dict(),
                'target_net': agent.target_net.state_dict(),
                'optimizer':  agent.optimizer.state_dict(),
                'epsilon':    agent.epsilon,
                'episode':    ep + 1,
            }, args.checkpoint)

    env.close()


if __name__ == '__main__':
    main()
