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


def test_grounded_future_hint_changes_inverse_transfer_target():
    torch.manual_seed(1)
    flc = FutureLatentControl(latent_dim=8, action_dim=3, sigma_dim=10)
    h = project_to_sphere(torch.randn(1, 8))
    hint = project_to_sphere(torch.randn(1, 8))
    sigma = torch.randn(1, 10)
    self_state = torch.rand(1, 2)
    action = torch.zeros(1, 3)

    ungrounded, diag_un = flc(h, sigma, self_state, action)
    grounded, diag_ground = flc(
        h,
        sigma,
        self_state,
        action,
        future_hint=hint,
        hint_confidence=1.0,
    )

    assert not torch.allclose(ungrounded, grounded)
    assert diag_un["future_hint_confidence"].item() == 0.0
    assert diag_ground["future_hint_confidence"].item() == 1.0
    assert not torch.allclose(
        diag_un["flc_correction_norm"], diag_ground["flc_correction_norm"]
    )


def test_event_grounded_inverse_learns_abstract_action_without_base_copy():
    torch.manual_seed(25)
    flc = FutureLatentControl(
        latent_dim=8,
        action_dim=3,
        sigma_dim=12,
        self_state_dim=2,
        hidden_dim=16,
    )
    optimizer = torch.optim.Adam(
        list(flc.grounded_transfer.parameters())
        + list(flc.grounded_motor_projection.parameters()),
        lr=0.03,
    )
    current = torch.nn.functional.normalize(torch.randn(6, 8), dim=-1)
    future = torch.nn.functional.normalize(torch.randn(6, 8), dim=-1)
    sigma = torch.randn(6, 12)
    self_state = torch.randn(6, 2)
    target = torch.tanh(torch.randn(6, 3))

    with torch.no_grad():
        before, _ = flc.grounded_inverse_action(
            current, future, sigma, self_state
        )
        before_error = (before - target).square().mean()
    for _ in range(80):
        optimizer.zero_grad()
        loss = flc.grounded_inverse_loss(
            current_latents=current,
            target_future=future,
            sigma_flat=sigma,
            self_state=self_state,
            target_actions=target,
        )
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        after, _ = flc.grounded_inverse_action(
            current, future, sigma, self_state
        )
        after_error = (after - target).square().mean()

    assert after_error < before_error * 0.1
