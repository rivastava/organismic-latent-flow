"""Event-horizon prospective consequence field for future-latent control."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from olf.geometry import exponential_map, log_map_sphere, project_to_tangent


class ProspectiveEventField(nn.Module):
    """Predict nonlocal event endpoints from situated abstract actions.

    The immediate consequence model answers what this action deforms now. This
    field answers what event-horizon latent this action participates in. It is
    trained only from observed transitions and endogenous viability, with no
    task labels or symbolic goal identities.
    """

    def __init__(
        self,
        latent_dim=32,
        sigma_dim=64,
        action_dim=3,
        hidden_dim=64,
        max_horizon=64,
    ):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.max_horizon = int(max_horizon)
        input_dim = latent_dim + sigma_dim + action_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.tangent_head = nn.Linear(hidden_dim, latent_dim)
        self.value_head = nn.Linear(hidden_dim, 1)
        self.risk_head = nn.Linear(hidden_dim, 1)
        self.horizon_head = nn.Linear(hidden_dim, 1)

        with torch.no_grad():
            self.tangent_head.weight.mul_(0.05)
            self.tangent_head.bias.zero_()
            self.value_head.weight.mul_(0.05)
            self.value_head.bias.zero_()
            self.risk_head.weight.mul_(0.05)
            self.risk_head.bias.fill_(-2.0)
            self.horizon_head.weight.mul_(0.05)
            self.horizon_head.bias.zero_()

    def forward(self, current_latent, sigma_t, action):
        if current_latent.ndim != 2 or sigma_t.ndim != 3 or action.ndim != 2:
            raise ValueError(
                "expected latent (B,D), sigma (B,N,H), action (B,A)"
            )
        num_entities = sigma_t.shape[1]
        h_expanded = current_latent.unsqueeze(1).expand(-1, num_entities, -1)
        a_expanded = action.unsqueeze(1).expand(-1, num_entities, -1)
        encoded = self.encoder(
            torch.cat([h_expanded, sigma_t, a_expanded], dim=-1)
        )
        tangent = project_to_tangent(
            h_expanded, self.tangent_head(encoded)
        )
        tangent_norm = tangent.norm(dim=-1, keepdim=True)
        max_angle = math.pi - 1e-4
        tangent = tangent * torch.clamp(
            max_angle / tangent_norm.clamp_min(1e-8), max=1.0
        )
        future_latent = exponential_map(h_expanded, tangent)
        horizon = 1.0 + (self.max_horizon - 1.0) * torch.sigmoid(
            self.horizon_head(encoded)
        )
        return {
            "future_latent": future_latent,
            "future_tangent": tangent,
            "value": self.value_head(encoded),
            "risk": torch.sigmoid(self.risk_head(encoded)),
            "horizon": horizon,
        }

    @staticmethod
    def eligibility_weights(latents, effects, endpoint):
        """Coordinate-free alignment of local deformation with later event."""
        target_tangent = log_map_sphere(
            latents, endpoint.expand_as(latents)
        )
        alignment = F.cosine_similarity(
            effects, target_tangent, dim=-1, eps=1e-8
        ).clamp_min(0.0)
        if float(alignment.sum().detach().item()) <= 1e-8:
            alignment = torch.ones_like(alignment)
        return alignment / alignment.mean().clamp_min(1e-8)

    def event_loss(
        self,
        *,
        latents,
        sigmas,
        actions,
        effects,
        endpoint,
        entity_mask,
        future_value,
        lethal,
    ):
        """Ground prior situated states in one later observation event.

        Inputs span the eligible history ending at the event. Entity slots are
        correspondence handles only within that episode; the predictor itself
        is shared across entities and therefore permutation equivariant.
        """
        length = latents.shape[0]
        if length == 0:
            return None
        if not (
            sigmas.shape[0]
            == actions.shape[0]
            == effects.shape[0]
            == length
        ):
            raise ValueError("prospective event traces must have equal length")
        entity_indices = torch.nonzero(
            entity_mask.reshape(-1), as_tuple=False
        ).flatten()
        if entity_indices.numel() == 0:
            return None

        prediction = self(latents, sigmas, actions)
        weights = self.eligibility_weights(
            latents, effects, endpoint
        ).detach()
        endpoint_target = endpoint.expand(length, -1)
        value_target = torch.as_tensor(
            future_value, dtype=latents.dtype, device=latents.device
        ).reshape(1, 1).expand(length, 1)
        risk_target = torch.as_tensor(
            lethal, dtype=latents.dtype, device=latents.device
        ).reshape(1, 1).expand(length, 1)
        horizon_target = torch.arange(
            length, 0, -1, dtype=latents.dtype, device=latents.device
        ).reshape(length, 1)
        horizon_target = horizon_target.clamp(max=float(self.max_horizon))

        terms = []
        for entity_index in entity_indices.tolist():
            predicted_endpoint = prediction["future_latent"][:, entity_index]
            endpoint_error = 1.0 - (
                predicted_endpoint * endpoint_target
            ).sum(dim=-1).clamp(-1.0, 1.0)
            value_error = (
                prediction["value"][:, entity_index] - value_target
            ).square().squeeze(-1)
            risk_error = F.binary_cross_entropy(
                prediction["risk"][:, entity_index],
                risk_target,
                reduction="none",
            ).squeeze(-1)
            horizon_error = (
                (
                    prediction["horizon"][:, entity_index]
                    - horizon_target
                )
                / float(self.max_horizon)
            ).square().squeeze(-1)
            terms.append(
                (
                    weights
                    * (
                        endpoint_error
                        + value_error
                        + risk_error
                        + 0.1 * horizon_error
                    )
                ).mean()
            )
        return torch.stack(terms).mean()
