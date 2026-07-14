"""Adaptive population of spherical transfer-law ghosts."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .geometry import exponential_map, log_map_sphere, project_to_tangent
from .set_control import BranchBelief


def _directions(current: torch.Tensor, matrices: torch.Tensor) -> torch.Tensor:
    raw = torch.einsum("kij,nj->nki", matrices, current)
    tangent = project_to_tangent(current.unsqueeze(1).expand_as(raw), raw)
    return F.normalize(tangent, dim=-1, eps=1e-8)


def _target_directions(current: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    tangent = log_map_sphere(current, target)
    return F.normalize(tangent, dim=-1, eps=1e-8)


def _errors(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.acos(
        (predicted * target.unsqueeze(1)).sum(dim=-1).clamp(-1.0, 1.0)
    )


def _fit_matrix(current: torch.Tensor, target_direction: torch.Tensor) -> torch.Tensor:
    dim = current.shape[-1]
    covariance = current.T @ current + 1e-3 * torch.eye(dim)
    regression = (target_direction.T @ current) @ torch.linalg.inv(covariance)
    skew = 0.5 * (regression - regression.T)
    return skew / skew.norm().clamp_min(1e-8)


def _sample_matrix(current: torch.Tensor, target_direction: torch.Tensor) -> torch.Tensor:
    matrix = torch.outer(target_direction, current) - torch.outer(current, target_direction)
    return matrix / matrix.norm().clamp_min(1e-8)


def _refine(
    current: torch.Tensor,
    targets: torch.Tensor,
    matrices: torch.Tensor,
    *,
    iterations: int = 30,
) -> tuple[torch.Tensor, torch.Tensor]:
    assignment = torch.zeros(len(current), dtype=torch.long)
    for _ in range(iterations):
        errors = _errors(_directions(current, matrices), targets)
        new_assignment = errors.argmin(dim=1)
        if torch.equal(new_assignment, assignment) and _ > 0:
            break
        assignment = new_assignment
        updated = []
        for index, matrix in enumerate(matrices):
            selected = assignment == index
            updated.append(
                _fit_matrix(current[selected], targets[selected])
                if int(selected.sum().item()) >= current.shape[-1]
                else matrix
            )
        matrices = torch.stack(updated)
    return matrices, assignment


def _supported_improvement(
    before: torch.Tensor,
    after: torch.Tensor,
    *,
    old_population: int,
    latent_dim: int,
    total_observations: int,
) -> tuple[bool, float, float]:
    """Require paired evidence and a BIC reduction for added transfer capacity."""
    gain = before - after
    mean = float(gain.mean().item())
    standard_error = float(gain.std(unbiased=True).item() / math.sqrt(len(gain)))
    parameters_per_skew_law = latent_dim * (latent_dim - 1) / 2
    old_parameters = old_population * parameters_per_skew_law + old_population - 1
    new_parameters = (old_population + 1) * parameters_per_skew_law + old_population
    old_mse = before.square().mean().clamp_min(1e-12)
    new_mse = after.square().mean().clamp_min(1e-12)
    # The held-out partition estimates predictive error; description length is
    # amortized over all completed observations available to the learner.
    old_bic = total_observations * torch.log(old_mse) + old_parameters * math.log(total_observations)
    new_bic = total_observations * torch.log(new_mse) + new_parameters * math.log(total_observations)
    supported = mean > 1.96 * standard_error and bool(new_bic < old_bic)
    return supported, mean, standard_error


@dataclass(frozen=True)
class TransferLifecycleEvent:
    kind: str
    accepted: bool
    heldout_gain: float
    gain_standard_error: float
    population: int


@dataclass(frozen=True)
class TransferGhostField:
    matrices: torch.Tensor
    weights: torch.Tensor
    probe_arc: torch.Tensor
    endpoint_arc: torch.Tensor
    scale: torch.Tensor
    events: tuple[TransferLifecycleEvent, ...]

    def belief(self, context: torch.Tensor) -> BranchBelief:
        if context.ndim != 1:
            raise ValueError("context must be one observable context vector")
        dim = self.matrices.shape[-1]
        current = context[:dim]
        directions = _directions(current.unsqueeze(0), self.matrices)[0]
        probes = exponential_map(
            current.unsqueeze(0).expand_as(directions),
            directions * self.probe_arc.unsqueeze(-1),
        )
        endpoints = exponential_map(
            current.unsqueeze(0).expand_as(directions),
            directions * self.endpoint_arc.unsqueeze(-1),
        )
        return BranchBelief(probes, endpoints, self.weights, self.scale)


def fit_transfer_ghosts(
    contexts: torch.Tensor,
    probes: torch.Tensor,
    endpoints: torch.Tensor,
    *,
    capacity: int = 8,
    seed: int = 0,
) -> TransferGhostField:
    """Discover transfer laws using only completed observed trajectories."""
    dim = endpoints.shape[-1]
    current = contexts[:, :dim]
    targets = _target_directions(current, endpoints)
    probe_arcs = log_map_sphere(current, probes).norm(dim=-1)
    endpoint_arcs = log_map_sphere(current, endpoints).norm(dim=-1)
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(current), generator=generator)
    fit_end = max(dim * 2, int(0.7 * len(order)))
    selection_end = max(fit_end + dim, int(0.85 * len(order)))
    fit = order[:fit_end]
    selection = order[fit_end:selection_end]
    evidence = order[selection_end:]

    matrices = _fit_matrix(current[fit], targets[fit]).unsqueeze(0)
    matrices, assignment = _refine(current[fit], targets[fit], matrices)
    events = []
    while len(matrices) < capacity:
        fit_errors = _errors(_directions(current[fit], matrices), targets[fit])
        residual = fit_errors.min(dim=1).values
        candidate_indices = residual.argsort(descending=True)[: min(12, len(residual))]
        proposals = []
        for candidate_index in candidate_indices.tolist():
            candidate = _sample_matrix(
                current[fit][candidate_index], targets[fit][candidate_index]
            )
            proposal, proposal_assignment = _refine(
                current[fit],
                targets[fit],
                torch.cat((matrices, candidate.unsqueeze(0))),
            )
            selection_error = _errors(
                _directions(current[selection], proposal), targets[selection]
            ).min(dim=1).values.square().mean()
            proposals.append((float(selection_error.item()), proposal, proposal_assignment))
        _, proposal, proposal_assignment = min(proposals, key=lambda item: item[0])
        before = _errors(
            _directions(current[evidence], matrices), targets[evidence]
        ).min(dim=1).values
        after = _errors(
            _directions(current[evidence], proposal), targets[evidence]
        ).min(dim=1).values
        accepted, gain, standard_error = _supported_improvement(
            before,
            after,
            old_population=len(matrices),
            latent_dim=dim,
            total_observations=len(current),
        )
        if accepted:
            matrices, assignment = proposal, proposal_assignment
        events.append(
            TransferLifecycleEvent(
                kind="birth",
                accepted=accepted,
                heldout_gain=gain,
                gain_standard_error=standard_error,
                population=len(matrices),
            )
        )
        if not accepted:
            break

    counts = torch.bincount(assignment, minlength=len(matrices)).float()
    weights = (counts + 0.5) / (counts.sum() + 0.5 * len(counts))
    per_probe_arc = []
    per_endpoint_arc = []
    per_scale = []
    fit_error = _errors(_directions(current[fit], matrices), targets[fit])
    for index in range(len(matrices)):
        selected = assignment == index
        per_probe_arc.append(probe_arcs[fit][selected].mean())
        per_endpoint_arc.append(endpoint_arcs[fit][selected].mean())
        angular = fit_error[selected, index]
        per_scale.append(angular.square().mean().sqrt().clamp(0.025, 0.5))
    return TransferGhostField(
        matrices=matrices,
        weights=weights,
        probe_arc=torch.stack(per_probe_arc),
        endpoint_arc=torch.stack(per_endpoint_arc),
        scale=torch.stack(per_scale),
        events=tuple(events),
    )
