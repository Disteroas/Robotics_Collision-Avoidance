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
