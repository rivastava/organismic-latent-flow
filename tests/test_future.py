import torch

from olf.future import FutureLatentControl
from olf.geometry import project_to_sphere


def test_future_latent_control_shapes_and_sphere():
    torch.manual_seed(0)
    flc = FutureLatentControl(latent_dim=8, action_dim=3, sigma_dim=10)
    h = project_to_sphere(torch.randn(2, 8))
    sigma = torch.randn(2, 10)
    self_state = torch.rand(2, 2)
    action = torch.zeros(2, 3)

    controlled, diag = flc(h, sigma, self_state, action)

    assert controlled.shape == (2, 3)
    assert torch.all(controlled <= 1.0)
    assert torch.all(controlled >= -1.0)
    assert diag["future_horizon"].shape == (2, 1)
    assert diag["flc_correction_norm"].shape == (2, 1)

