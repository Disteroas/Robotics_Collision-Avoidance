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
