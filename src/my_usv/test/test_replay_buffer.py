import pytest
import numpy as np
from train_core import ReplayBuffer

STATE_DIM = 50


def _transition(value=1.0):
    return (
        np.ones(STATE_DIM, dtype=np.float32) * value,
        3,
        float(value),
        np.zeros(STATE_DIM, dtype=np.float32),
        False,
    )


def test_empty_buffer_has_zero_length():
    buf = ReplayBuffer(100)
    assert len(buf) == 0


def test_push_increases_length():
    buf = ReplayBuffer(100)
    buf.push(*_transition())
    assert len(buf) == 1


def test_capacity_respected_fifo():
    buf = ReplayBuffer(10)
    for i in range(20):
        buf.push(*_transition(float(i)))
    assert len(buf) == 10


def test_sample_returns_correct_batch_size():
    buf = ReplayBuffer(100)
    for _ in range(20):
        buf.push(*_transition())
    s, a, r, s2, d = buf.sample(8)
    assert len(s) == 8


def test_sample_shapes_match_state_dim():
    buf = ReplayBuffer(100)
    for _ in range(20):
        buf.push(*_transition())
    s, a, r, s2, d = buf.sample(5)
    assert s.shape  == (5, STATE_DIM)
    assert s2.shape == (5, STATE_DIM)
    assert a.shape  == (5,)
    assert r.shape  == (5,)
    assert d.shape  == (5,)


def test_sample_raises_when_buffer_too_small():
    buf = ReplayBuffer(100)
    buf.push(*_transition())
    with pytest.raises(ValueError):
        buf.sample(5)
