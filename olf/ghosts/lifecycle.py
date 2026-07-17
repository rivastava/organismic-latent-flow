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
from olf.geometry import parallel_transport_sphere, project_to_sphere


class LifecycleReport:
    """Diagnostic record of the lifecycle decisions taken this recoupling."""
    def __init__(self):
        self.born = False
        self.merged = False
        self.evicted: list[int] = []
        self.reasons: list[str] = []

    def note(self, reason: str) -> None:
        self.reasons.append(reason)


def best_predictive_error(population: GhostPopulation, observed_anchor: torch.Tensor,
                          step: float) -> torch.Tensor:
    if len(population) == 0:
        return torch.tensor(float("inf"))
    errs = [predictive_error(g, observed_anchor, step) for g in population]
    return torch.stack(errs).min()


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
    best_err = float(best_predictive_error(population, observed_anchor, config.transport_step))
    max_unc = max(float(g.uncertainty) for g in population._ghosts)
    if best_err > max_unc:
        report.born = True
        report.note(
            f"birth: observed error {best_err:.3f} outside all predictive "
            f"intervals (max uncertainty {max_unc:.3f})"
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
    """Retain alternatives until cross-evidence can justify a merge."""
    report.note("retain: merge disabled pending cross-evidence")
    return population


def evict(population: GhostPopulation, config: GhostConfig,
          report: LifecycleReport) -> GhostPopulation:
    """Retain alternatives until negative evidence is calibrated for eviction."""
    report.note("retain: eviction disabled pending calibrated negative evidence")
    return population
