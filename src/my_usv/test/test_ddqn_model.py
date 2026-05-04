import torch
import pytest
from ddqn_model import DDQN, STATE_DIM, ACTION_DIM


def test_single_input_output_shape():
    model = DDQN()
    x = torch.zeros(1, STATE_DIM)
    out = model(x)
    assert out.shape == (1, ACTION_DIM)


def test_batch_output_shape():
    model = DDQN()
    x = torch.zeros(64, STATE_DIM)
    out = model(x)
    assert out.shape == (64, ACTION_DIM)


def test_forward_pass_finite_for_normalized_inputs():
    # Input in [0,1] (come arriva dall'env) non deve produrre NaN o inf
    model = DDQN()
    x = torch.rand(100, STATE_DIM)
    out = model(x)
    assert torch.all(torch.isfinite(out))


def test_different_inputs_give_different_outputs():
    model = DDQN()
    x_zeros = torch.zeros(1, STATE_DIM)
    x_ones  = torch.ones(1, STATE_DIM)
    assert not torch.allclose(model(x_zeros), model(x_ones))


def test_greedy_action_is_argmax_of_q_values():
    model = DDQN()
    model.eval()
    with torch.no_grad():
        x   = torch.rand(1, STATE_DIM)
        out = model(x)
        expected_action = int(out.argmax(dim=1).item())
    # Verifica che argmax coincida con il massimo Q-value
    assert out[0, expected_action] == out[0].max()


def test_gradients_flow_through_all_layers():
    model = DDQN()
    x = torch.rand(4, STATE_DIM)
    loss = model(x).sum()
    loss.backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, f"Nessun gradiente su {name}"
        assert torch.any(param.grad != 0), f"Gradiente zero su {name}"
