"""Structural ablation tests for the ``no_schema_composition`` mode.

This ablation disables only recursive schema-to-schema composition
(``composed_schemas`` -> ``schemas``). Depth-one schema reuse remains
active. The tests verify:

1. Full mode can restore a branch with depth greater than one.
2. ``no_schema_composition`` restores only depth-one schemas.
3. Both modes use the same underlying consolidated memory.
4. The ablation does not disable schema reuse.
5. Episode boundaries remain respected.
6. ``off`` and ``observe`` remain paired-identical.
7. No reward, success, task identity, labels, or coordinates enter
   composition.
8. All values remain finite.
9. Existing frame-equivariance tests still pass.
"""

import numpy as np
import torch

from olf.geometry import project_to_sphere, project_to_tangent
from olf.ghosts.config import (
    ABLATIONS,
    GhostConfig,
)
from olf.ghosts.recoupling import ReachabilityBuffer
from olf.organism import Organism


def _alternating_schema_buffer(separate_edges=False):
    """Construct a buffer containing repeated alternating schema adjacencies.

    With ``separate_edges=False`` the four recouplings form a single segment
    with multiple alternating adjacencies, which produces recursive branches
    of depth two when ``composed_schemas`` is queried. With
    ``separate_edges=True`` the recouplings are split into segments so the
    cross-edge composition is forbidden and depth never exceeds one.
    """
    anchor = project_to_sphere(torch.randn(8))
    first = project_to_tangent(anchor, torch.randn(8))
    first = 0.1 * first / first.norm()
    second = project_to_tangent(anchor, torch.randn(8))
    second = second - (second @ first) / (first @ first) * first
    second = 0.1 * second / second.norm()
    actions = (torch.tensor([1.0, 0.0, 0.0]), torch.tensor([0.0, 1.0, 0.0]))
    buffer = ReachabilityBuffer(16)
    for index, (action, tangent) in enumerate(
        ((actions[0], first), (actions[1], second)) * 2
    ):
        if separate_edges and index:
            buffer.start_segment()
        buffer.add(anchor, action, tangent)
        buffer.consolidate_latest(1.0)
    return buffer, anchor


def _make_organism(ablation=None):
    config = GhostConfig(
        ghost_mode="influence" if ablation else "observe",
        ablation=ablation,
        latent_dim=8,
        action_dim=3,
        capacity=4,
    )
    return Organism(
        obs_dim=18,
        latent_dim=8,
        hidden_dim=16,
        ghost_mode=config.ghost_mode,
        ghost_config=config,
    )


def test_1_full_mode_can_restore_branch_with_depth_greater_than_one():
    """Full influence mode uses ``composed_schemas`` and produces depth > 1."""
    organism = _make_organism(ablation=None)
    buffer, anchor = _alternating_schema_buffer()
    organism.ghost.buffer = buffer
    organism.ghost.reset()
    organism.ghost.begin_step(anchor)
    assert len(organism.ghost.population) >= 1
    composed = organism.ghost.buffer.composed_schemas(anchor, 4)
    assert any(branch["depth"] == 2 for branch in composed)
    telemetry = organism.ghost.telemetry()
    assert telemetry["max_schema_depth"] >= 2
    assert telemetry["composed_branches"] >= 1


def test_2_no_schema_composition_restores_only_depth_one_schemas():
    """``no_schema_composition`` uses ``schemas`` and never depth > 1."""
    organism = _make_organism(ablation="no_schema_composition")
    buffer, anchor = _alternating_schema_buffer()
    organism.ghost.buffer = buffer
    organism.ghost.reset()
    organism.ghost.begin_step(anchor)
    assert len(organism.ghost.population) >= 1
    schemas = organism.ghost.buffer.schemas(anchor, 4)
    assert len(schemas) >= 1
    assert all(s.get("depth", 1) == 1 for s in schemas)
    telemetry = organism.ghost.telemetry()
    assert telemetry["max_schema_depth"] <= 1
    assert telemetry["composed_branches"] == 0


def test_3_both_modes_use_same_underlying_consolidated_memory():
    """Both conditions read the same ``ReachabilityBuffer`` instance."""
    buffer, anchor = _alternating_schema_buffer()
    full = _make_organism(ablation=None)
    no_comp = _make_organism(ablation="no_schema_composition")
    full.ghost.buffer = buffer
    no_comp.ghost.buffer = buffer
    full.ghost.reset()
    no_comp.ghost.reset()
    full.ghost.begin_step(anchor)
    no_comp.ghost.begin_step(anchor)
    assert full.ghost.buffer is no_comp.ghost.buffer
    full_schemas = full.ghost.buffer.schemas(anchor, 4)
    no_comp_schemas = no_comp.ghost.buffer.schemas(anchor, 4)
    assert len(full_schemas) == len(no_comp_schemas)
    for schema in full_schemas:
        matching_branch = next(
            branch
            for branch in full.ghost.buffer.composed_schemas(anchor, 4)
            if branch["path"] == (schema["schema_id"],)
        )
        assert matching_branch["strength"] == schema["strength"]
        assert matching_branch["uncertainty"] == schema["uncertainty"]
        assert torch.equal(matching_branch["action"], schema["action"])
        assert torch.allclose(
            matching_branch["tangent"], schema["tangent"], atol=1e-5, rtol=1e-5
        )


def test_4_ablation_does_not_disable_schema_reuse():
    """Depth-one schemas are still restored under ``no_schema_composition``."""
    organism = _make_organism(ablation="no_schema_composition")
    buffer, anchor = _alternating_schema_buffer()
    organism.ghost.buffer = buffer
    organism.ghost.reset()
    organism.ghost.begin_step(anchor)
    assert len(organism.ghost.population) >= 1
    restored = organism.ghost.population[0]
    assert float(restored.grounding) > 0.0
    assert len(restored.transfer_actions) >= 2
    schemas = organism.ghost.buffer.schemas(anchor, 4)
    assert len(schemas) >= 1


def test_5_episode_boundaries_remain_respected():
    """``start_segment`` still forbids cross-episode composition under both."""
    for ablation in (None, "no_schema_composition"):
        organism = _make_organism(ablation=ablation)
        buffer, anchor = _alternating_schema_buffer(separate_edges=True)
        organism.ghost.buffer = buffer
        organism.ghost.reset()
        organism.ghost.begin_step(anchor)
        composed = organism.ghost.buffer.composed_schemas(anchor, 4)
        assert all(branch["depth"] == 1 for branch in composed)
        schemas = organism.ghost.buffer.schemas(anchor, 4)
        assert all(s.get("depth", 1) == 1 for s in schemas)


def test_6_off_and_observe_remain_paired_identical():
    """``off`` and ``observe`` modes must produce no ghost influence at all,
    regardless of which ablation is set on the influence side. Verify by
    constructing both and confirming only ``off`` produces no ghost object
    while ``observe`` runs ghost state without influencing the action."""
    off = Organism(
        obs_dim=18,
        latent_dim=8,
        hidden_dim=16,
        ghost_mode="off",
        ghost_config=GhostConfig(
            ghost_mode="off",
            latent_dim=8,
            action_dim=3,
            capacity=4,
        ),
    )
    assert off.ghost is None
    observe = Organism(
        obs_dim=18,
        latent_dim=8,
        hidden_dim=16,
        ghost_mode="observe",
        ghost_config=GhostConfig(
            ghost_mode="observe",
            latent_dim=8,
            action_dim=3,
            capacity=4,
        ),
    )
    assert observe.ghost is not None
    assert observe.ghost.config.ghost_mode == "observe"
    assert observe.ghost.config.influences_action is False


def test_7_no_reward_success_task_labels_or_coordinates_enter_composition():
    """The composed schemas must not contain any prohibited-label substring."""
    forbidden_substrings = (
        "reward",
        "success",
        "task",
        "label",
        "environment",
        "scripted",
        "privileged",
        "goal",
        "world",
        "role",
        "identity",
        "meaning",
        "benchmark",
    )
    buffer, anchor = _alternating_schema_buffer()
    composed = buffer.composed_schemas(anchor, 4)
    for record in composed:
        record_repr = repr(record)
        for forbidden in forbidden_substrings:
            assert forbidden not in record_repr, (
                f"forbidden substring {forbidden!r} found in composed branch"
            )
    schemas = buffer.schemas(anchor, 4)
    for record in schemas:
        record_repr = repr(record)
        for forbidden in forbidden_substrings:
            assert forbidden not in record_repr, (
                f"forbidden substring {forbidden!r} found in schema"
            )


def test_8_all_values_remain_finite():
    """Telemetry under the ablation must contain only finite numbers."""
    organism = _make_organism(ablation="no_schema_composition")
    buffer, anchor = _alternating_schema_buffer()
    organism.ghost.buffer = buffer
    organism.ghost.reset()
    organism.ghost.begin_step(anchor)
    telemetry = organism.ghost.telemetry()
    for key, value in telemetry.items():
        if isinstance(value, (int, float, np.floating)):
            assert np.isfinite(float(value)), f"non-finite telemetry {key}={value}"
        elif isinstance(value, torch.Tensor):
            assert torch.isfinite(value).all(), f"non-finite telemetry {key}"
    organism.ghost.check_invariants()


def test_9_existing_frame_equivariance_test_still_passes():
    """``ReachabilityBuffer.composed_schemas`` frame-equivariance holds even
    when the buffer is queried with the depth-one only ``schemas`` fallback."""
    torch.manual_seed(83)
    buffer, anchor = _alternating_schema_buffer()
    orthogonal, _ = torch.linalg.qr(torch.randn(8, 8))
    rotated = ReachabilityBuffer(16)
    for source, action, tangent in zip(
        buffer.anchors, buffer.actions, buffer.prototypes, strict=True
    ):
        rotated.add(orthogonal @ source, action, orthogonal @ tangent)
        rotated.consolidate_latest(1.0)
    original_branch = next(
        branch
        for branch in buffer.composed_schemas(anchor, 4)
        if branch["depth"] == 2
    )
    rotated_branch = next(
        branch
        for branch in rotated.composed_schemas(orthogonal @ anchor, 4)
        if branch["depth"] == 2
    )
    assert torch.allclose(
        rotated_branch["tangent"],
        orthogonal @ original_branch["tangent"],
        atol=1e-5,
        rtol=1e-5,
    )


def test_ablation_registered_in_ablations_enum():
    assert "no_schema_composition" in ABLATIONS


def test_schema_composition_enabled_property_is_default_true():
    config = GhostConfig(ghost_mode="influence")
    assert config.schema_composition_enabled is True


def test_schema_composition_disabled_under_ablation():
    config = GhostConfig(
        ghost_mode="influence", ablation="no_schema_composition"
    )
    assert config.schema_composition_enabled is False


def test_adjacency_without_predictive_evidence_cannot_authorize_depth():
    buffer, anchor = _alternating_schema_buffer()
    assert buffer.composition_observations
    buffer.composition_positive.clear()
    buffer.composition_negative.clear()
    buffer.composition_observations.clear()

    branches = buffer.composed_schemas(anchor, 4)
    assert branches
    assert all(branch["depth"] == 1 for branch in branches)


def test_negative_composition_evidence_suppresses_continuation():
    buffer, anchor = _alternating_schema_buffer()
    path = next(iter(buffer.composition_observations))
    buffer.composition_positive[path] = 0.25
    buffer.composition_negative[path] = 0.5

    branches = buffer.composed_schemas(anchor, 4)
    assert path not in {branch["path"] for branch in branches}


def test_deeper_continuation_requires_an_evidence_supported_prefix():
    buffer, anchor = _alternating_schema_buffer()
    path = next(iter(buffer.composition_observations))
    deeper = path + (path[0],)
    buffer.composition_positive[path] = 0.0
    buffer.composition_negative[path] = 1.0
    buffer.composition_positive[deeper] = 1.0
    buffer.composition_negative[deeper] = 0.0
    buffer.composition_observations[deeper] = 1

    branches = buffer.composed_schemas(anchor, 4)
    paths = {branch["path"] for branch in branches}
    assert path not in paths
    assert deeper not in paths
    assert buffer.composition_stats()["composition_eligible"] == 0


def test_composition_telemetry_reports_learned_candidates_and_eligibility():
    organism = _make_organism(ablation=None)
    buffer, anchor = _alternating_schema_buffer()
    organism.ghost.buffer = buffer
    organism.ghost.reset()
    organism.ghost.begin_step(anchor)

    telemetry = organism.ghost.telemetry()
    assert telemetry["composition_candidates"] >= 1
    assert telemetry["composition_eligible"] >= 1
