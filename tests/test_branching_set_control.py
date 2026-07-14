"""Constitutional and mechanical tests for Stage 1 set-valued control."""

from __future__ import annotations

import inspect
import torch

import experiments.branching_latent_flow.set_control as set_control
from experiments.branching_latent_flow.geometry import exponential_map, project_to_tangent
from experiments.branching_latent_flow.set_control import (
    BranchBelief,
    BranchSetPredictor,
    centroid_correction,
    inverse_corrections,
    recouple_weights,
    select_correction,
)


def _belief() -> tuple[torch.Tensor, BranchBelief]:
    current = torch.tensor([1.0, 0.0, 0.0, 0.0])
    tangents = torch.tensor(
        [
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    endpoints = exponential_map(current.expand(3, -1), tangents * 0.8)
    probes = exponential_map(current.expand(3, -1), tangents * 0.2)
    return current, BranchBelief(
        probes=probes,
        endpoints=endpoints,
        weights=torch.tensor([0.2, 0.7, 0.1]),
        scales=torch.full((3,), 0.08),
    )


def test_predictor_capacity_is_fixed_independently_of_world_modes():
    assert BranchSetPredictor(latent_dim=8, capacity=8).capacity == 8


def test_learner_module_has_no_oracle_or_reward_inputs():
    source = inspect.getsource(set_control)
    for forbidden in ("mode_index", "all_mode_endpoints", "n_modes", "reward"):
        assert forbidden not in source


def test_inverse_control_keeps_one_correction_per_future():
    current, belief = _belief()
    corrections = inverse_corrections(current, belief.endpoints)
    assert corrections.shape == belief.endpoints.shape
    assert torch.allclose(corrections.norm(dim=-1), torch.ones(3), atol=1e-6)
    assert torch.allclose((corrections * current).sum(dim=-1), torch.zeros(3), atol=1e-6)


def test_selected_action_is_a_branch_not_the_centroid():
    current, belief = _belief()
    selected, index = select_correction(current, belief)
    candidates = inverse_corrections(current, belief.endpoints)
    centroid = centroid_correction(current, belief)
    assert index == 1
    assert torch.allclose(selected, candidates[index])
    assert not torch.allclose(selected, centroid)


def test_recoupling_moves_mass_to_observation_consistent_branch():
    _, belief = _belief()
    posterior = recouple_weights(belief, belief.probes[0])
    assert int(posterior.argmax().item()) == 0
    assert posterior[0] > belief.weights[0]


def test_recoupling_is_permutation_equivariant():
    _, belief = _belief()
    order = torch.tensor([2, 0, 1])
    original = recouple_weights(belief, belief.probes[0])
    permuted = recouple_weights(belief.permute(order), belief.probes[0])
    assert torch.allclose(permuted[order.argsort()], original)


def test_map_selection_is_permutation_invariant_when_evidence_is_decisive():
    current, belief = _belief()
    order = torch.tensor([2, 0, 1])
    action, _ = select_correction(current, belief)
    permuted_action, _ = select_correction(current, belief.permute(order))
    assert torch.allclose(action, permuted_action)


def test_centroid_is_permutation_invariant():
    current, belief = _belief()
    order = torch.tensor([2, 0, 1])
    assert torch.allclose(
        centroid_correction(current, belief),
        centroid_correction(current, belief.permute(order)),
    )


def test_stochastic_selection_approximates_declared_belief_distribution():
    current, belief = _belief()
    generator = torch.Generator().manual_seed(9)
    counts = torch.zeros(3)
    for _ in range(6000):
        _, index = select_correction(
            current, belief, generator=generator, stochastic=True
        )
        counts[index] += 1
    frequencies = counts / counts.sum()
    assert torch.allclose(frequencies, belief.weights, atol=0.025)


def test_zero_radial_component_after_inverse_projection():
    current, belief = _belief()
    raw = inverse_corrections(current, belief.endpoints)
    assert torch.allclose(
        project_to_tangent(current.expand_as(raw), raw), raw, atol=1e-6
    )
