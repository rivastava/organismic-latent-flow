"""Bounded-capacity lifecycle: birth, merge, dormancy, eviction.

Rules are derived from ONLINE statistics of the ghosts themselves (their own
predictive uncertainty and signed evidence support), never from benchmark
constants or a desired population size. Every decision is returned as a
diagnostic so the organism can inspect *why* a ghost was born, merged, or
evicted. When a calibrated rule is not yet available the default is to retain
the ghost within capacity ("observe and retain") rather than destroy it.
"""

import torch

from .config import GhostConfig
from .evidence import observed_deformation, predictive_error
from .population import GhostPopulation
from .trajectory import make_ghost
from olf.geometry import (
    angular_distance,
    log_map_sphere,
    parallel_transport_sphere,
    project_to_sphere,
)


class LifecycleReport:
    """Diagnostic record of the lifecycle decisions taken this recoupling."""
    def __init__(self):
        self.born = False
        self.merged = False
        self.evicted: list[int] = []
        self.reasons: list[str] = []

    def note(self, reason: str) -> None:
        self.reasons.append(reason)


def maybe_birth(population: GhostPopulation, real_prev: torch.Tensor,
                observed_anchor: torch.Tensor, config: GhostConfig,
                report: LifecycleReport) -> GhostPopulation:
    """Birth only when the observed point lies outside every existing ghost's
    predictive interval (no ghost already predicted it within its own
    uncertainty). Capacity permitting. The new trajectory is seeded directly
    from the *observed* deformation, so it represents real external evidence.
    """
    if len(population) >= config.effective_capacity or config.effective_capacity == 0:
        return population
    if len(population) == 0:
        # No existing predictive interval covers an empty population.
        report.born = True
        report.note("birth: empty population, seed from observed deformation")
        return _seed(population, real_prev, observed_anchor, config)
    errors = [
        float(predictive_error(g, observed_anchor, config.transport_step))
        for g in population._ghosts
    ]
    uncertainties = [float(g.uncertainty) for g in population._ghosts]
    if all(
        error > uncertainty
        for error, uncertainty in zip(errors, uncertainties, strict=True)
    ):
        report.born = True
        report.note(
            "birth: observation outside every ghost's predictive interval"
        )
        return _seed(population, real_prev, observed_anchor, config)
    report.note("retain: observed point within an existing predictive interval")
    return population


def _seed(population, real_prev, observed_anchor, config):
    # observed_deformation is a tangent at real_prev. Since the new ghost's
    # anchor is the observed_anchor, parallel-transport the tangent from
    # real_prev to observed_anchor so it is a valid tangent there.
    real_prev_s = project_to_sphere(real_prev.detach().reshape(-1))
    observed_anchor_s = project_to_sphere(observed_anchor.detach().reshape(-1))
    tangent = observed_deformation(real_prev_s, observed_anchor_s)
    tangent = parallel_transport_sphere(real_prev_s, observed_anchor_s, tangent)
    new_ghost = make_ghost(
        anchor=observed_anchor.detach().reshape(-1),
        tangent=tangent.detach().reshape(-1),
        credibility=0.5,
        grounding=0.0,
        uncertainty=1.0,
        persistence=0.0,
        evidence_support=0.0,
        evidence_negative=0.0,
        boundary_compat=1.0,
        horizon_expr=1.0,
    )
    out = GhostPopulation.empty(config.latent_dim, config.effective_capacity)
    out._ghosts = list(population._ghosts) + [new_ghost]
    return out


def merge_similar(population: GhostPopulation, config: GhostConfig,
                  report: LifecycleReport) -> GhostPopulation:
    """Merge only numerically equivalent, externally grounded trajectories."""
    ghosts = population._ghosts
    if len(ghosts) < 2:
        return population
    tolerance = torch.finfo(ghosts[0].anchor.dtype).eps ** 0.5
    remaining = set(range(len(ghosts)))
    groups = []
    while remaining:
        seed = min(remaining)
        group = {seed}
        frontier = [seed]
        remaining.remove(seed)
        while frontier:
            left = frontier.pop()
            matches = [
                right
                for right in sorted(remaining)
                if _equivalent(ghosts[left], ghosts[right], config, tolerance)
            ]
            for right in matches:
                remaining.remove(right)
                group.add(right)
                frontier.append(right)
        groups.append(sorted(group))

    if all(len(group) == 1 for group in groups):
        report.note("retain: no externally equivalent ghosts")
        return population

    out = GhostPopulation.empty(population.latent_dim, population.capacity)
    for group in groups:
        members = [ghosts[index] for index in group]
        out._ghosts.append(
            members[0] if len(members) == 1 else _merge_group(members, config)
        )
    report.merged = True
    report.note(
        "merge: consolidated numerically equivalent externally grounded ghosts"
    )
    return out


def _equivalent(left, right, config, tolerance):
    if float(left.grounding) <= 0.0 or float(right.grounding) <= 0.0:
        return False
    future_distance = float(
        angular_distance(
            left.predicted_anchor(config.transport_step),
            right.predicted_anchor(config.transport_step),
        )
    )
    if future_distance > tolerance:
        return False
    left_defined = left.transfer_identifiable(config.min_action_evidence)
    right_defined = right.transfer_identifiable(config.min_action_evidence)
    if left_defined != right_defined:
        return False
    if not left_defined:
        return True
    actions = left.transfer_actions + right.transfer_actions
    for action in actions:
        left_prediction = left.transfer_predict(action, left.anchor)
        right_prediction = right.transfer_predict(action, left.anchor)
        scale = 1.0 + max(
            float(left_prediction.norm()), float(right_prediction.norm())
        )
        if float((left_prediction - right_prediction).norm()) > tolerance * scale:
            return False
    return True


def _merge_group(ghosts, config):
    anchor_sum = torch.stack([ghost.anchor for ghost in ghosts]).sum(dim=0)
    anchor = project_to_sphere(anchor_sum)
    future_sum = torch.stack(
        [ghost.predicted_anchor(config.transport_step) for ghost in ghosts]
    ).sum(dim=0)
    future = project_to_sphere(future_sum)
    tangent = log_map_sphere(anchor, future) / config.transport_step
    count = len(ghosts)
    uncertainty = torch.stack([ghost.uncertainty for ghost in ghosts])
    inverse_variance = 1.0 / uncertainty.square().clamp_min(
        torch.finfo(uncertainty.dtype).eps
    )
    combined_uncertainty = 1.0 / torch.sqrt(inverse_variance.sum())
    merged = make_ghost(
        anchor,
        tangent,
        credibility=sum(float(ghost.credibility) for ghost in ghosts) / count,
        grounding=sum(float(ghost.grounding) for ghost in ghosts) / count,
        uncertainty=float(combined_uncertainty),
        persistence=max(float(ghost.persistence) for ghost in ghosts),
        evidence_support=sum(float(ghost.evidence_support) for ghost in ghosts),
        evidence_negative=sum(float(ghost.evidence_negative) for ghost in ghosts),
        boundary_compat=(
            sum(float(ghost.boundary_compat) for ghost in ghosts) / count
        ),
        horizon_expr=sum(float(ghost.horizon_expr) for ghost in ghosts) / count,
    )
    for ghost in ghosts:
        for source, action, observed in zip(
            ghost.transfer_anchors,
            ghost.transfer_actions,
            ghost.transfer_tangents,
            strict=True,
        ):
            merged = merged.add_action_evidence(source, action, observed)
    return merged


def evict(population: GhostPopulation, config: GhostConfig,
          report: LifecycleReport) -> GhostPopulation:
    """Remove only ghosts strictly dominated by external predictive evidence."""
    dominated = []
    ghosts = population._ghosts
    for index, ghost in enumerate(ghosts):
        if float(ghost.persistence) <= 0.0:
            continue
        current = _evidence_coordinates(ghost)
        for other_index, other in enumerate(ghosts):
            if index == other_index:
                continue
            candidate = _evidence_coordinates(other)
            no_worse = (
                candidate[0] >= current[0]
                and candidate[1] <= current[1]
                and candidate[2] <= current[2]
                and candidate[3] >= current[3]
                and candidate[4] >= current[4]
            )
            strictly_better = (
                candidate[0] > current[0]
                or candidate[1] < current[1]
                or candidate[2] < current[2]
                or candidate[3] > current[3]
                or candidate[4] > current[4]
            )
            if no_worse and strictly_better:
                dominated.append(index)
                break

    if not dominated:
        report.note("retain: no ghost dominated on external evidence")
        return population
    out = GhostPopulation.empty(population.latent_dim, population.capacity)
    removed = set(dominated)
    out._ghosts = [
        ghost for index, ghost in enumerate(ghosts) if index not in removed
    ]
    report.evicted.extend(sorted(removed))
    report.note(f"evict: externally dominated ghosts {sorted(removed)}")
    return out


def _evidence_coordinates(ghost):
    return (
        float(ghost.evidence_support),
        float(ghost.evidence_negative),
        float(ghost.uncertainty),
        float(ghost.credibility),
        float(ghost.grounding),
    )
