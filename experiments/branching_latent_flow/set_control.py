"""Set-valued future-latent control for the branching-flow experiment.

The predictor learns a bounded set of possible observed probes and future
endpoints from completed trajectories. The controller never averages that set
before generating corrections: every future produces its own tangent action.
Observed flow recouples the set by a Bayesian likelihood update.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .geometry import log_map_sphere, project_to_sphere, project_to_tangent


@dataclass(frozen=True)
class BranchBelief:
    """A permutation-equivariant set of possible continuations."""

    probes: torch.Tensor
    endpoints: torch.Tensor
    weights: torch.Tensor
    scales: torch.Tensor

    def permute(self, order: torch.Tensor) -> BranchBelief:
        return BranchBelief(
            probes=self.probes[order],
            endpoints=self.endpoints[order],
            weights=self.weights[order],
            scales=self.scales[order],
        )


class BranchSetPredictor(nn.Module):
    """Conditional mixture over an observed probe and a later sphere endpoint."""

    def __init__(
        self,
        latent_dim: int,
        capacity: int = 8,
        hidden_dim: int = 64,
        depth: int = 2,
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.capacity = int(capacity)
        input_dim = 3 * self.latent_dim
        layers: list[nn.Module] = []
        for layer in range(depth):
            layers.extend(
                [
                    nn.Linear(input_dim if layer == 0 else hidden_dim, hidden_dim),
                    nn.SiLU(),
                ]
            )
        self.encoder = nn.Sequential(*layers)
        self.pair_head = nn.Linear(hidden_dim, capacity * 2 * latent_dim)
        self.logit_head = nn.Linear(hidden_dim, capacity)
        self.scale_head = nn.Linear(hidden_dim, capacity)

    def raw(self, context: torch.Tensor) -> tuple[torch.Tensor, ...]:
        encoded = self.encoder(context)
        pair = self.pair_head(encoded).reshape(
            -1, self.capacity, 2, self.latent_dim
        )
        pair = project_to_sphere(pair)
        probes = pair[:, :, 0]
        endpoints = pair[:, :, 1]
        logits = self.logit_head(encoded)
        scales = 0.025 + 0.475 * torch.sigmoid(self.scale_head(encoded))
        return probes, endpoints, logits, scales

    def component_log_prob(
        self,
        context: torch.Tensor,
        probe: torch.Tensor,
        endpoint: torch.Tensor,
    ) -> torch.Tensor:
        probes, endpoints, logits, scales = self.raw(context)
        pair_error = (probe.unsqueeze(1) - probes).square().sum(dim=-1)
        pair_error = pair_error + (
            endpoint.unsqueeze(1) - endpoints
        ).square().sum(dim=-1)
        observed_dim = 2 * self.latent_dim
        log_density = (
            -0.5 * observed_dim * math.log(2.0 * math.pi)
            - observed_dim * scales.log()
            - 0.5 * pair_error / scales.square()
        )
        return F.log_softmax(logits, dim=-1) + log_density

    def nll(
        self,
        context: torch.Tensor,
        probe: torch.Tensor,
        endpoint: torch.Tensor,
    ) -> torch.Tensor:
        return -torch.logsumexp(
            self.component_log_prob(context, probe, endpoint), dim=-1
        ).mean()

    @torch.no_grad()
    def belief(self, context: torch.Tensor) -> BranchBelief:
        if context.ndim == 1:
            context = context.unsqueeze(0)
        probes, endpoints, logits, scales = self.raw(context)
        return BranchBelief(
            probes=probes[0].detach(),
            endpoints=endpoints[0].detach(),
            weights=F.softmax(logits[0], dim=-1).detach(),
            scales=scales[0].detach(),
        )

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def recouple_weights(
    belief: BranchBelief,
    observed_probe: torch.Tensor,
) -> torch.Tensor:
    """Update branch credibility from an observed latent-flow departure."""
    probe_error = (observed_probe.unsqueeze(0) - belief.probes).square().sum(dim=-1)
    d = observed_probe.shape[-1]
    log_likelihood = (
        -0.5 * d * math.log(2.0 * math.pi)
        - d * belief.scales.log()
        - 0.5 * probe_error / belief.scales.square()
    )
    return F.softmax(belief.weights.clamp_min(1e-12).log() + log_likelihood, dim=-1)


def inverse_corrections(
    current: torch.Tensor,
    endpoints: torch.Tensor,
) -> torch.Tensor:
    """Generate one unit tangent correction for every possible future."""
    expanded = current.unsqueeze(0).expand_as(endpoints)
    corrections = log_map_sphere(expanded, endpoints)
    corrections = project_to_tangent(expanded, corrections)
    return F.normalize(corrections, p=2, dim=-1, eps=1e-8)


def centroid_correction(
    current: torch.Tensor,
    belief: BranchBelief,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Single-vector control that averages branch corrections before acting."""
    corrections = inverse_corrections(current, belief.endpoints)
    use_weights = belief.weights if weights is None else weights
    mean = (use_weights.unsqueeze(-1) * corrections).sum(dim=0)
    return F.normalize(
        project_to_tangent(current, mean), p=2, dim=-1, eps=1e-8
    )


def select_correction(
    current: torch.Tensor,
    belief: BranchBelief,
    *,
    weights: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
    stochastic: bool = False,
) -> tuple[torch.Tensor, int]:
    """Select one branch correction without collapsing branches into a mean."""
    use_weights = belief.weights if weights is None else weights
    if stochastic:
        index = int(torch.multinomial(use_weights, 1, generator=generator).item())
    else:
        index = int(use_weights.argmax().item())
    corrections = inverse_corrections(current, belief.endpoints)
    return corrections[index], index
