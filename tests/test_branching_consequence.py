"""Constitutional tests for consequence-driven temporary ghost strengths."""

from __future__ import annotations

import inspect
from dataclasses import fields

import torch

import experiments.branching_latent_flow.consequence_deformation as consequence_deformation
from experiments.branching_latent_flow.consequence_deformation import (
    ConsequenceTraces,
    deform_from_consequence,
    fit_consequence_model,
)
from experiments.branching_latent_flow.geometry import exponential_map
from experiments.branching_latent_flow.role_free_ghosts import materialize_trajectories, trajectory_identity
from experiments.branching_latent_flow.set_control import BranchBelief


def _training_traces() -> ConsequenceTraces:
    before = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).expand(80, -1)
    positive = torch.tensor([0.0, 1.0, 0.0, 0.0])
    negative = -positive
    deformation = torch.stack([positive if index % 2 == 0 else negative for index in range(80)])
    self_state = torch.tensor([1.0 if (index // 2) % 2 == 0 else -1.0 for index in range(80)])
    viability = torch.full((80,), 0.5)
    viability_delta = 0.3 * self_state * deformation[:, 1]
    after = exponential_map(before, deformation)
    boundary = torch.zeros(80)
    return ConsequenceTraces(
        before,
        deformation,
        after,
        self_state,
        viability,
        viability_delta,
        boundary,
    )


def _population():
    present = torch.tensor([1.0, 0.0, 0.0, 0.0])
    deformation = torch.tensor(
        [[0.0, 1.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0]]
    )
    endpoints = exponential_map(present.expand(2, -1), deformation)
    belief = BranchBelief(endpoints, endpoints, torch.tensor([0.5, 0.5]), torch.full((2,), 0.08))
    return present, materialize_trajectories(present, belief)


def test_trace_schema_contains_no_functional_role_target():
    names = {field.name for field in fields(ConsequenceTraces)}
    assert not names & {"role", "action_value", "goal", "memory", "warning"}


def test_learning_core_has_no_benchmark_or_hidden_world_channel():
    source = inspect.getsource(consequence_deformation)
    for forbidden in ("mode_index", "all_mode_endpoints", "n_modes", "success"):
        assert forbidden not in source


def test_same_trajectory_flips_temporary_influence_with_self_state():
    model = fit_consequence_model(_training_traces())
    present, population = _population()
    positive = deform_from_consequence(model, population, present, 1.0, 0.5)
    negative = deform_from_consequence(model, population, present, -1.0, 0.5)
    assert positive.influence.argmax() != negative.influence.argmax()
    assert positive.influence[0] > 0 and negative.influence[0] < 0


def test_deformation_changes_strengths_not_shared_trajectory_or_grounding():
    model = fit_consequence_model(_training_traces())
    present, population = _population()
    updated = deform_from_consequence(model, population, present, 1.0, 0.5)
    assert trajectory_identity(updated) == trajectory_identity(population)
    assert torch.equal(updated.grounding, population.grounding)
    assert not torch.equal(updated.influence, population.influence)
