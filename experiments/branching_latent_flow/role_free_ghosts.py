"""Role-free spherical trajectories emitted by learned transformation memory."""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch
import torch.nn.functional as F

from .geometry import (
    angular_distance,
    exponential_map,
    log_map_sphere,
    project_to_tangent,
    slerp,
)
from .set_control import BranchBelief


@dataclass(frozen=True)
class GhostTrajectories:
    """One shared substrate; meaning comes only from relations and use."""

    points: torch.Tensor
    credibility: torch.Tensor
    grounding: torch.Tensor
    persistence: torch.Tensor
    influence: torch.Tensor
    horizon_mass: torch.Tensor
    uncertainty: torch.Tensor
    boundary_risk: torch.Tensor

    @property
    def count(self) -> int:
        return self.points.shape[0]

    def permute(self, order: torch.Tensor) -> GhostTrajectories:
        return GhostTrajectories(
            points=self.points[order],
            credibility=self.credibility[order],
            grounding=self.grounding[order],
            persistence=self.persistence[order],
            influence=self.influence[order],
            horizon_mass=self.horizon_mass[order],
            uncertainty=self.uncertainty[order],
            boundary_risk=self.boundary_risk[order],
        )


def materialize_trajectories(
    current: torch.Tensor,
    belief: BranchBelief,
    *,
    steps: int = 5,
) -> GhostTrajectories:
    """Deform a common present into several temporary latent continuations."""
    times = torch.linspace(0.0, 1.0, steps + 1).view(1, steps + 1, 1)
    starts = current.view(1, 1, -1).expand(belief.endpoints.shape[0], steps + 1, -1)
    ends = belief.endpoints.unsqueeze(1).expand_as(starts)
    points = slerp(starts, ends, times)
    horizon_mass = torch.zeros(belief.endpoints.shape[0], steps + 1)
    horizon_mass[:, -1] = 1.0
    zeros = torch.zeros_like(belief.weights)
    return GhostTrajectories(
        points=points,
        credibility=belief.weights.clone(),
        grounding=zeros.clone(),
        persistence=torch.ones_like(belief.weights),
        influence=belief.weights.clone(),
        horizon_mass=horizon_mass,
        uncertainty=belief.scales.clone(),
        boundary_risk=zeros.clone(),
    )


def future_view(population: GhostTrajectories) -> torch.Tensor:
    """Read each trajectory at its currently expressed horizon."""
    indices = population.horizon_mass.argmax(dim=-1)
    return population.points[torch.arange(population.count), indices]


def correction_view(
    population: GhostTrajectories, present: torch.Tensor
) -> torch.Tensor:
    """Read those same latents as magnitude-preserving tangent deformations."""
    futures = future_view(population)
    expanded = present.unsqueeze(0).expand_as(futures)
    return project_to_tangent(expanded, log_map_sphere(expanded, futures))


def continuation_view(
    present: torch.Tensor, deformations: torch.Tensor
) -> torch.Tensor:
    """Read tangent deformations back as points on the shared substrate."""
    expanded = present.unsqueeze(0).expand_as(deformations)
    return exponential_map(expanded, project_to_tangent(expanded, deformations))


def compatibility_view(
    population: GhostTrajectories,
    observed: torch.Tensor,
    *,
    phase: int,
) -> torch.Tensor:
    """Read trajectories as memories of an externally observed deformation."""
    predicted = population.points[:, phase]
    return angular_distance(
        predicted, observed.unsqueeze(0).expand_as(predicted)
    ).squeeze(-1)


def recouple_observation(
    population: GhostTrajectories,
    observed: torch.Tensor,
    *,
    phase: int,
) -> GhostTrajectories:
    """External evidence changes credibility and grounding; rehearsal cannot."""
    error = compatibility_view(population, observed, phase=phase)
    scale = population.uncertainty.clamp_min(0.025)
    log_likelihood = -0.5 * error.square() / scale.square() - scale.log()
    credibility = F.softmax(
        population.credibility.clamp_min(1e-12).log() + log_likelihood,
        dim=0,
    )
    evidence = torch.exp(-0.5 * error.square() / scale.square())
    grounding = 1.0 - (1.0 - population.grounding) * (1.0 - evidence)
    return replace(
        population,
        credibility=credibility,
        grounding=grounding.clamp(0.0, 1.0),
    )


def predecessor_view(
    population: GhostTrajectories,
    observed_effect: torch.Tensor,
    *,
    phase: int,
) -> tuple[torch.Tensor, int]:
    """Read the same trajectory backward from effect to its prior latent."""
    if phase <= 0:
        raise ValueError("phase must have a predecessor")
    errors = compatibility_view(population, observed_effect, phase=phase)
    score = population.credibility.clamp_min(1e-12).log() - errors / population.uncertainty.clamp_min(0.025)
    index = int(score.argmax().item())
    return population.points[index, phase - 1], index


def population_entropy(population: GhostTrajectories) -> torch.Tensor:
    weights = population.credibility.clamp_min(1e-12)
    return -(weights * weights.log()).sum()


def trajectory_identity(population: GhostTrajectories) -> int:
    """Diagnostic identity of the shared backing tensor, never semantic identity."""
    return population.points.data_ptr()
