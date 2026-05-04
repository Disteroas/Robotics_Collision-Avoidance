import pytest
import numpy as np
import torch
from train_core import DDQNAgent, TARGET_UPDATE_STEPS, EPSILON_MIN, BETA_DECAY

STATE_DIM = 50


def _state():
    return np.zeros(STATE_DIM, dtype=np.float32)


def test_initial_epsilon_is_one():
    agent = DDQNAgent()
    assert agent.epsilon == pytest.approx(1.0)


def test_epsilon_decays_by_beta_each_episode():
    agent = DDQNAgent()
    agent.epsilon = 1.0
    agent.decay_epsilon()
    assert agent.epsilon == pytest.approx(1.0 * BETA_DECAY)


def test_epsilon_never_goes_below_minimum():
    agent = DDQNAgent()
    agent.epsilon = 0.001
    agent.decay_epsilon()
    assert agent.epsilon == pytest.approx(EPSILON_MIN)


def test_greedy_action_is_deterministic_when_epsilon_zero():
    agent = DDQNAgent()
    agent.epsilon = 0.0
    actions = [agent.act(_state()) for _ in range(20)]
    assert len(set(actions)) == 1


def test_greedy_action_in_valid_range():
    agent = DDQNAgent()
    agent.epsilon = 0.0
    action = agent.act(_state())
    assert 0 <= action <= 10


def test_learn_returns_none_when_buffer_empty():
    agent = DDQNAgent()
    assert agent.learn() is None


def test_learn_returns_float_loss_after_enough_transitions():
    agent = DDQNAgent()
    s = _state()
    for _ in range(100):
        agent.memory.push(s, 5, 1.0, s, False)
    loss = agent.learn()
    assert isinstance(loss, float)
    assert loss >= 0.0


def test_target_net_synced_after_update_step():
    agent = DDQNAgent()
    with torch.no_grad():
        agent.q_net.fc1.weight.fill_(42.0)
    agent.total_steps = TARGET_UPDATE_STEPS - 1
    agent.step_done()
    assert torch.allclose(agent.target_net.fc1.weight,
                          agent.q_net.fc1.weight)


def test_target_net_not_synced_before_update_step():
    agent = DDQNAgent()
    original_weights = agent.target_net.fc1.weight.clone()
    with torch.no_grad():
        agent.q_net.fc1.weight.fill_(42.0)
    agent.step_done()  # total_steps=1, non triggera update
    assert torch.allclose(agent.target_net.fc1.weight, original_weights)
