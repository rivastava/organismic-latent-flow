"""Future-latent control as an OLF subsystem."""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from olf.geometry import exponential_map, project_to_tangent
from olf.transfer import InverseTransferField


@dataclass
class FutureLatent:
    latent: torch.Tensor
    horizon: torch.Tensor
    correction: torch.Tensor
    abstraction: torch.Tensor


class FutureLatentField(nn.Module):
    """Forms an abstract future latent from the current organism state.

    The field does not name goals. It produces a task-conditioned latent
    direction on the unit sphere plus a soft horizon/abstraction estimate.
    """

    def __init__(
        self,
        latent_dim=32,
        sigma_dim=128,
        self_state_dim=2,
        hidden_dim=64,
        max_horizon=8.0,
    ):
        super().__init__()
        self.max_horizon = max_horizon
        in_dim = latent_dim + sigma_dim + self_state_dim
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.delta_head = nn.Linear(hidden_dim, latent_dim)
        self.horizon_head = nn.Linear(hidden_dim, 1)
        self.abstraction_head = nn.Linear(hidden_dim, 1)

        with torch.no_grad():
            self.delta_head.weight.mul_(0.05)
            self.delta_head.bias.zero_()
            self.horizon_head.bias.fill_(-1.0)
            self.abstraction_head.bias.zero_()

    def forward(self, current_latent, sigma_flat, self_state):
        x = torch.cat([current_latent, sigma_flat, self_state], dim=-1)
        z = self.encoder(x)

        raw_delta = self.delta_head(z)
        tangent_delta = project_to_tangent(current_latent, raw_delta)
        horizon = 1.0 + self.max_horizon * torch.sigmoid(self.horizon_head(z))
        abstraction = torch.sigmoid(self.abstraction_head(z))

        # Scale the latent step by a small learned horizon factor so the field
        # starts conservative but can still represent nonlocal future pressure.
        step = tangent_delta * (0.05 + 0.05 * torch.log1p(horizon))
        future_latent = exponential_map(current_latent, step)
        return FutureLatent(
            latent=future_latent,
            horizon=horizon,
            correction=tangent_delta,
            abstraction=abstraction,
        )


class FutureLatentControl(nn.Module):
    """FLC loop: current latent -> future latent -> inverse correction -> action."""

    def __init__(
        self,
        latent_dim=32,
        action_dim=3,
        sigma_dim=128,
        self_state_dim=2,
        hidden_dim=64,
    ):
        super().__init__()
        self.future_field = FutureLatentField(
            latent_dim=latent_dim,
            sigma_dim=sigma_dim,
            self_state_dim=self_state_dim,
            hidden_dim=hidden_dim,
        )
        self.transfer = InverseTransferField(latent_dim=latent_dim)
        self.motor_projection = nn.Sequential(
            nn.Linear(latent_dim * 3 + sigma_dim + self_state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )
        self.gain = nn.Parameter(torch.tensor(-2.0))

    def forward(self, current_latent, sigma_flat, self_state, base_action):
        future = self.future_field(current_latent, sigma_flat, self_state)
        present_correction = self.transfer.inverse_correction(
            current_latent, future.latent
        )
        projection_input = torch.cat(
            [
                current_latent,
                future.latent,
                present_correction,
                sigma_flat,
                self_state,
                base_action,
            ],
            dim=-1,
        )
        action_delta = self.motor_projection(projection_input)
        gain = 0.25 * torch.sigmoid(self.gain)
        action = torch.clamp(base_action + gain * action_delta, -1.0, 1.0)

        diagnostics = {
            "future_horizon": future.horizon.detach(),
            "future_abstraction": future.abstraction.detach(),
            "future_alignment": F.cosine_similarity(
                current_latent.detach(), future.latent.detach(), dim=-1, eps=1e-8
            ).unsqueeze(-1),
            "flc_correction_norm": present_correction.detach().norm(dim=-1, keepdim=True),
            "flc_action_delta_norm": action_delta.detach().norm(dim=-1, keepdim=True),
            "flc_gain": gain.detach().reshape(1, 1),
        }
        return action, diagnostics

