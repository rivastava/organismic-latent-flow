"""Situated consequence deformation for untyped spherical trajectories."""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch

from .role_free_ghosts import GhostTrajectories, correction_view


@dataclass(frozen=True)
class ConsequenceTraces:
    before: torch.Tensor
    deformation: torch.Tensor
    after: torch.Tensor
    self_state: torch.Tensor
    viability: torch.Tensor
    viability_delta: torch.Tensor
    boundary_deformation: torch.Tensor


@dataclass(frozen=True)
class SituatedConsequenceModel:
    weights: torch.Tensor
    residual_scale: torch.Tensor
    situated: bool

    def predict(
        self,
        before: torch.Tensor,
        deformation: torch.Tensor,
        self_state: torch.Tensor,
        viability: torch.Tensor,
    ) -> torch.Tensor:
        features = consequence_features(
            before,
            deformation,
            self_state,
            viability,
            situated=self.situated,
        )
        return features @ self.weights


def consequence_features(
    before: torch.Tensor,
    deformation: torch.Tensor,
    self_state: torch.Tensor,
    viability: torch.Tensor,
    *,
    situated: bool,
) -> torch.Tensor:
    self_state = self_state.reshape(-1, 1)
    viability = viability.reshape(-1, 1)
    ones = torch.ones_like(self_state)
    base = [ones, before, deformation, self_state, viability]
    if situated:
        base.extend(
            [
                self_state * deformation,
                viability * deformation,
                before * deformation,
                self_state * before,
                self_state * viability,
            ]
        )
    return torch.cat(base, dim=-1)


def fit_consequence_model(
    traces: ConsequenceTraces,
    *,
    situated: bool = True,
    ridge: float = 1e-3,
) -> SituatedConsequenceModel:
    features = consequence_features(
        traces.before,
        traces.deformation,
        traces.self_state,
        traces.viability,
        situated=situated,
    )
    targets = torch.stack(
        (traces.viability_delta, traces.boundary_deformation), dim=-1
    )
    regularizer = ridge * torch.eye(features.shape[-1], dtype=features.dtype)
    weights = torch.linalg.solve(
        features.T @ features + regularizer,
        features.T @ targets,
    )
    residual = targets - features @ weights
    residual_scale = residual.square().mean(dim=0).sqrt().clamp_min(1e-4)
    return SituatedConsequenceModel(weights, residual_scale, situated)


def deform_from_consequence(
    model: SituatedConsequenceModel,
    population: GhostTrajectories,
    present: torch.Tensor,
    self_state: torch.Tensor | float,
    viability: torch.Tensor | float,
) -> GhostTrajectories:
    """Express temporary strengths from predicted situated consequence."""
    deformation = correction_view(population, present)
    count = population.count
    before = present.unsqueeze(0).expand(count, -1)
    self_tensor = torch.as_tensor(self_state, dtype=present.dtype).expand(count)
    viability_tensor = torch.as_tensor(viability, dtype=present.dtype).expand(count)
    prediction = model.predict(
        before, deformation, self_tensor, viability_tensor
    )
    influence = prediction[:, 0]
    boundary_risk = prediction[:, 1].clamp(0.0, 1.0)
    confidence = torch.exp(-model.residual_scale[0]).expand(count)
    persistence = confidence * (0.5 + 0.5 * population.credibility)
    uncertainty = model.residual_scale[0].expand(count)
    return replace(
        population,
        influence=influence,
        boundary_risk=boundary_risk,
        persistence=persistence,
        uncertainty=uncertainty,
    )


def release_index(population: GhostTrajectories) -> int:
    """Release the strongest viable deformation; negative influence may inhibit."""
    viable = population.boundary_risk < 0.5
    pressure = population.influence.clone()
    pressure[~viable] = -torch.inf
    if not bool(viable.any().item()):
        pressure = population.influence - population.boundary_risk
    return int(pressure.argmax().item())
