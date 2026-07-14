"""Constitutional tests for role-free ghost trajectories."""

from __future__ import annotations

from dataclasses import fields

import torch

from experiments.branching_latent_flow.geometry import exponential_map
from experiments.branching_latent_flow.role_free_ghosts import (
    GhostTrajectories,
    compatibility_view,
    continuation_view,
    correction_view,
    future_view,
    materialize_trajectories,
    predecessor_view,
    recouple_observation,
    trajectory_identity,
)
from experiments.branching_latent_flow.set_control import BranchBelief


def _population():
    current = torch.tensor([1.0, 0.0, 0.0, 0.0])
    tangents = torch.tensor([[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
    endpoints = exponential_map(current.expand(2, -1), tangents)
    belief = BranchBelief(
        probes=endpoints,
        endpoints=endpoints,
        weights=torch.tensor([0.5, 0.5]),
        scales=torch.tensor([0.08, 0.08]),
    )
    return current, materialize_trajectories(current, belief, steps=5)


def test_ghost_has_no_permanent_role_or_type_field():
    names = {field.name for field in fields(GhostTrajectories)}
    assert not names & {"role", "kind", "type", "future", "action", "memory", "cause"}


def test_all_ghost_points_share_the_spherical_substrate():
    _, population = _population()
    assert torch.allclose(
        population.points.norm(dim=-1), torch.ones_like(population.points[..., 0]), atol=1e-5
    )


def test_relational_reads_do_not_replace_or_mutate_the_trajectory():
    current, population = _population()
    identity = trajectory_identity(population)
    grounding = population.grounding.clone()
    future_view(population)
    correction_view(population, current)
    compatibility_view(population, population.points[0, 3], phase=3)
    predecessor_view(population, population.points[0, 5], phase=5)
    assert trajectory_identity(population) == identity
    assert torch.equal(population.grounding, grounding)


def test_only_external_recoupling_changes_grounding():
    _, population = _population()
    updated = recouple_observation(population, population.points[1, 3], phase=3)
    assert torch.equal(population.grounding, torch.zeros(2))
    assert updated.grounding[1] > updated.grounding[0]
    assert updated.credibility[1] > updated.credibility[0]
    assert trajectory_identity(updated) == trajectory_identity(population)


def test_future_and_correction_are_reversible_relational_views():
    current, population = _population()
    deformations = correction_view(population, current)
    reconstructed = continuation_view(current, deformations)
    assert torch.allclose(reconstructed, future_view(population), atol=1e-5)


def test_views_are_permutation_equivariant():
    current, population = _population()
    order = torch.tensor([1, 0])
    permuted = population.permute(order)
    assert torch.allclose(future_view(permuted)[order.argsort()], future_view(population))
    assert torch.allclose(
        correction_view(permuted, current)[order.argsort()],
        correction_view(population, current),
    )


def test_effect_can_reveal_a_predecessor_on_the_same_trajectory():
    _, population = _population()
    predecessor, index = predecessor_view(
        population, population.points[1, 5], phase=5
    )
    assert index == 1
    assert torch.allclose(predecessor, population.points[1, 4])
