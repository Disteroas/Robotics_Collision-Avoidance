# Gymnasium Wrapper Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wrap `UsvEnv` in a `gymnasium.Env`-compatible interface and add a second training entry point (`train_gym.py`) that uses it with the existing DDQN agent, enabling future algorithm swaps with a single import change.

**Architecture:** Composition pattern — `UsvGymEnv(gymnasium.Env)` contains `UsvEnv` as attribute. `rclpy` lifecycle owned by wrapper. `train_gym.py` uses `UsvGymEnv` + existing `DDQNAgent` from `train_core.py`. No modifications to `usv_env.py`, `train.py`, or `train_core.py`.

**Tech Stack:** Python 3.10, gymnasium, numpy, pytest (inside Docker), existing ROS2/Gazebo infrastructure.

---

## File Map

| File | Type | Responsibility |
|---|---|---|
| `src/my_usv/scripts/usv_gym_env.py` | Create | `UsvGymEnv(gymnasium.Env)` wrapper |
| `src/my_usv/scripts/train_gym.py` | Create | Training entry point using gymnasium API |
| `src/my_usv/test/test_usv_gym_env.py` | Create | 10 unit tests, `UsvEnv` fully mocked |
| `Dockerfile` | Modify | Add `gymnasium` to pip install |

---

## Task 1: Add gymnasium to Dockerfile

**Files:**
- Modify: `Dockerfile`

- [ ] **Step 1: Read Dockerfile to find the pip install line**

Read `Dockerfile` and locate the line containing `pip3 install pytest`.

- [ ] **Step 2: Add gymnasium**

Change:
```dockerfile
RUN pip3 install pytest
```
To:
```dockerfile
RUN pip3 install pytest gymnasium
```

- [ ] **Step 3: Verify build**

```bash
docker build -t usv_rl_project . 2>&1 | tail -5
```

Expected: `Successfully built ...` (no errors). If build fails with network error, retry once.

- [ ] **Step 4: Verify gymnasium importable**

```bash
docker run --rm usv_rl_project python3 -c "import gymnasium; print(gymnasium.__version__)"
```

Expected: prints a version string like `0.29.1`, no ImportError.

- [ ] **Step 5: Commit**

```bash
git add Dockerfile
git commit -m "build: add gymnasium to Docker image"
```

---

## Task 2: Write all tests for UsvGymEnv (RED)

**Files:**
- Create: `src/my_usv/test/test_usv_gym_env.py`

**Context:** `UsvEnv` is a ROS2 Node — it needs Gazebo running. Tests must mock it entirely. The mock must be installed into `sys.modules` BEFORE `usv_gym_env` is imported, otherwise Python tries to import the real `rclpy` and fails.

- [ ] **Step 1: Create test file with mock setup and all 10 tests**

Create `src/my_usv/test/test_usv_gym_env.py` with this exact content:

```python
import sys
from unittest.mock import MagicMock
import numpy as np
import pytest

# ── Mock ROS2 stack and UsvEnv before importing usv_gym_env ──────────────
# These must be set before any import that triggers usv_env.py or rclpy.
for _mod in (
    'rclpy', 'rclpy.node', 'rclpy.parameter', 'rclpy.time',
    'geometry_msgs', 'geometry_msgs.msg',
    'sensor_msgs', 'sensor_msgs.msg',
    'std_srvs', 'std_srvs.srv',
):
    sys.modules[_mod] = MagicMock()

_env_instance = MagicMock()
_env_instance.reset_environment.return_value = np.ones(50, dtype=np.float32) * 0.5
_env_instance.step_action.return_value = (np.ones(50, dtype=np.float32) * 0.5, 5.0, False)

_usv_env_mod = MagicMock()
_usv_env_mod.UsvEnv = MagicMock(return_value=_env_instance)
sys.modules['usv_env'] = _usv_env_mod
# ─────────────────────────────────────────────────────────────────────────

import gymnasium
from usv_gym_env import UsvGymEnv  # noqa: E402 — must come after mocks

LIDAR_BEAMS = 50
MAX_STEPS   = 5   # small value to test truncation quickly


@pytest.fixture(autouse=True)
def _reset_mock():
    _env_instance.reset_mock()
    _env_instance.reset_environment.return_value = np.ones(LIDAR_BEAMS, dtype=np.float32) * 0.5
    _env_instance.step_action.return_value = (np.ones(LIDAR_BEAMS, dtype=np.float32) * 0.5, 5.0, False)


def _env(continuous=False):
    return UsvGymEnv(continuous=continuous, max_steps=MAX_STEPS)


# ── Spaces ────────────────────────────────────────────────────────────────

def test_observation_space_shape():
    env = _env()
    assert env.observation_space.shape == (LIDAR_BEAMS,)
    assert env.observation_space.dtype == np.float32


def test_action_space_discrete():
    env = _env(continuous=False)
    assert isinstance(env.action_space, gymnasium.spaces.Discrete)
    assert env.action_space.n == 11


def test_action_space_continuous():
    env = _env(continuous=True)
    assert isinstance(env.action_space, gymnasium.spaces.Box)
    assert env.action_space.shape == (1,)
    assert float(env.action_space.low[0])  == pytest.approx(-0.8)
    assert float(env.action_space.high[0]) == pytest.approx(0.8)


# ── reset() ───────────────────────────────────────────────────────────────

def test_reset_returns_obs_and_empty_info():
    env = _env()
    obs, info = env.reset()
    assert obs.shape == (LIDAR_BEAMS,)
    assert info == {}


def test_reset_resets_step_counter():
    env = _env()
    env.reset()
    for _ in range(MAX_STEPS):
        env.step(5)
    env.reset()
    _, _, _, truncated, info = env.step(5)
    assert info['steps'] == 1
    assert not truncated


# ── step() discrete ───────────────────────────────────────────────────────

def test_step_returns_correct_5_tuple():
    env = _env()
    env.reset()
    obs, reward, terminated, truncated, info = env.step(5)
    assert obs.shape == (LIDAR_BEAMS,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert 'steps' in info and 'crashed' in info


def test_terminated_true_on_crash():
    env = _env()
    env.reset()
    _env_instance.step_action.return_value = (np.ones(LIDAR_BEAMS) * 0.1, -1000.0, True)
    _, _, terminated, truncated, info = env.step(5)
    assert terminated is True
    assert truncated is False
    assert info['crashed'] is True


def test_truncated_true_on_step_limit():
    env = _env()
    env.reset()
    for _ in range(MAX_STEPS - 1):
        env.step(5)
    _, _, terminated, truncated, _ = env.step(5)
    assert truncated is True
    assert terminated is False


def test_terminated_false_on_truncation():
    """terminated must be False when truncated — value bootstrap depends on this."""
    env = _env()
    env.reset()
    for _ in range(MAX_STEPS - 1):
        env.step(5)
    _, _, terminated, truncated, _ = env.step(5)
    assert truncated is True
    assert terminated is False   # must stay False even at step limit


# ── step() continuous ─────────────────────────────────────────────────────

def test_continuous_action_maps_center_to_index_5():
    env = _env(continuous=True)
    env.reset()
    env.step(np.array([0.0], dtype=np.float32))
    _env_instance.step_action.assert_called_with(5)


def test_continuous_action_maps_extremes():
    env = _env(continuous=True)
    env.reset()
    env.step(np.array([-0.8], dtype=np.float32))
    _env_instance.step_action.assert_called_with(0)
    env.step(np.array([0.8], dtype=np.float32))
    _env_instance.step_action.assert_called_with(10)
```

- [ ] **Step 2: Run tests — verify RED**

```bash
docker run --rm --volume="/$(pwd):/home/usv_ws" usv_rl_project bash -c \
  "cd /home/usv_ws && python3 -m pytest src/my_usv/test/test_usv_gym_env.py -v 2>&1"
```

Expected: all 10 tests FAILED with `ModuleNotFoundError: No module named 'usv_gym_env'`. If any test PASSES, the mock setup is wrong — investigate before continuing.

- [ ] **Step 3: Commit RED tests**

```bash
git add src/my_usv/test/test_usv_gym_env.py
git commit -m "test(gym): add 10 failing tests for UsvGymEnv"
```

---

## Task 3: Implement UsvGymEnv (GREEN)

**Files:**
- Create: `src/my_usv/scripts/usv_gym_env.py`

- [ ] **Step 1: Create usv_gym_env.py**

Create `src/my_usv/scripts/usv_gym_env.py` with this exact content:

```python
import numpy as np
import rclpy
import gymnasium
from gymnasium.spaces import Box, Discrete

from usv_env import UsvEnv
from usv_logic import LIDAR_BEAMS


class UsvGymEnv(gymnasium.Env):
    """
    Gymnasium wrapper for UsvEnv.

    Exposes standard gymnasium API so any gymnasium-compatible algorithm
    (XinJingHao DRL, stable-baselines3, cleanRL, etc.) can train on this env
    without modification.

    Swap algorithm in train_gym.py — this wrapper stays unchanged.
    """

    metadata = {'render_modes': []}

    def __init__(self, continuous: bool = False, max_steps: int = 1000):
        super().__init__()
        rclpy.init()
        self._env       = UsvEnv()
        self._cont      = continuous
        self._max_steps = max_steps
        self._steps     = 0

        self.observation_space = Box(
            low=0.0, high=1.0, shape=(LIDAR_BEAMS,), dtype=np.float32
        )
        if continuous:
            self.action_space = Box(
                low=np.float32(-0.8), high=np.float32(0.8),
                shape=(1,), dtype=np.float32
            )
        else:
            self.action_space = Discrete(11)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._steps = 0
        obs = self._env.reset_environment()
        return obs, {}

    def step(self, action):
        if self._cont:
            # inverse of: angular_z = -0.8 + 0.16 * idx
            idx = int(np.clip(round((float(action[0]) + 0.8) / 0.16), 0, 10))
        else:
            idx = int(action)

        obs, reward, crashed = self._env.step_action(idx)
        self._steps += 1

        terminated = crashed
        truncated  = (not crashed) and (self._steps >= self._max_steps)

        return obs, float(reward), terminated, truncated, {
            'steps': self._steps, 'crashed': crashed
        }

    def close(self):
        self._env.destroy_node()
        rclpy.shutdown()
```

- [ ] **Step 2: Run usv_gym_env tests — verify GREEN**

```bash
docker run --rm --volume="/$(pwd):/home/usv_ws" usv_rl_project bash -c \
  "cd /home/usv_ws && python3 -m pytest src/my_usv/test/test_usv_gym_env.py -v 2>&1"
```

Expected: all 10 PASSED. If any fail, fix `usv_gym_env.py` — do NOT touch the tests.

- [ ] **Step 3: Run full suite — verify no regressions**

```bash
docker run --rm --volume="/$(pwd):/home/usv_ws" usv_rl_project bash -c \
  "cd /home/usv_ws && python3 -m pytest src/my_usv/test/ -v 2>&1"
```

Expected: 51 PASSED (41 existing + 10 new), 0 FAILED.

- [ ] **Step 4: Commit**

```bash
git add src/my_usv/scripts/usv_gym_env.py
git commit -m "feat(gym): add UsvGymEnv gymnasium wrapper with discrete and continuous action spaces"
```

---

## Task 4: Implement train_gym.py

**Files:**
- Create: `src/my_usv/scripts/train_gym.py`

No unit tests for the training script — validated by running 10 episodes in Docker with Gazebo (integration check).

- [ ] **Step 1: Create train_gym.py**

Create `src/my_usv/scripts/train_gym.py` with this exact content:

```python
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
                   help='Maze ID passed to start_gazebo_block (1 or 2)')
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
        state, _ = env.reset()
        ep_reward = 0.0
        losses    = []

        while True:
            action                               = agent.act(state)
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
```

- [ ] **Step 2: Verify syntax (no Gazebo needed)**

```bash
docker run --rm --volume="/$(pwd):/home/usv_ws" usv_rl_project bash -c \
  "cd /home/usv_ws && python3 -c 'import ast; ast.parse(open(\"src/my_usv/scripts/train_gym.py\").read()); print(\"Syntax OK\")' 2>&1"
```

Expected: `Syntax OK`.

- [ ] **Step 3: Verify --help parses correctly (no Gazebo needed)**

`train_gym.py` imports `usv_gym_env` which imports `rclpy`. In a Docker container without Gazebo, `rclpy.init()` is only called inside `UsvGymEnv.__init__` — not at import time. So `--help` should work without Gazebo:

```bash
docker run --rm --volume="/$(pwd):/home/usv_ws" usv_rl_project bash -c \
  "cd /home/usv_ws && source install/setup.bash && python3 src/my_usv/scripts/train_gym.py --help 2>&1"
```

Expected: help text with `--maze-id`, `--episodes`, `--max-steps`, `--continuous`, `--checkpoint` visible. No crash.

- [ ] **Step 4: Commit**

```bash
git add src/my_usv/scripts/train_gym.py
git commit -m "feat(gym): add train_gym.py DDQN training entry point via gymnasium"
```

---

## Task 5: Full verification and push

**Files:** none (verification only)

- [ ] **Step 1: Run full test suite**

```bash
docker run --rm --volume="/$(pwd):/home/usv_ws" usv_rl_project bash -c \
  "cd /home/usv_ws && python3 -m pytest src/my_usv/test/ -v 2>&1"
```

Expected: **51 PASSED**, 0 FAILED.

- [ ] **Step 2: Push**

```bash
git push origin gym_env_claude
```

Expected: `gym_env_claude -> gym_env_claude` on remote.
