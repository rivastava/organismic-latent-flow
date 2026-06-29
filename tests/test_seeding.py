import numpy as np
import torch

from olf.seeding import set_seed


def test_set_seed_reproducible_numpy_and_torch():
    set_seed(123)
    a_np = np.random.rand(4)
    a_torch = torch.rand(4)

    set_seed(123)
    b_np = np.random.rand(4)
    b_torch = torch.rand(4)

    assert np.allclose(a_np, b_np)
    assert torch.allclose(a_torch, b_torch)

