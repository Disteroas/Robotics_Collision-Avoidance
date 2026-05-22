import random
import numpy as np
import torch
from seeding import set_global_seed


def test_same_seed_reproducible():
    set_global_seed(42)
    a = (random.random(), float(np.random.rand()), float(torch.rand(1)))
    set_global_seed(42)
    b = (random.random(), float(np.random.rand()), float(torch.rand(1)))
    assert a == b


def test_different_seed_differs():
    set_global_seed(1)
    a = random.random()
    set_global_seed(2)
    b = random.random()
    assert a != b
