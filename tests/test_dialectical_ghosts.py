import torch

from olf.geometry import project_to_sphere, project_to_tangent
from olf.ghosts.config import GhostConfig
from olf.ghosts.evidence import update_after_recoupling
from olf.ghosts.integration import _non_dominated_indices
from olf.ghosts.lifecycle import (
    LifecycleReport,
    evict,
    maybe_birth,
    merge_similar,
)
from olf.ghosts.population import GhostPopulation
from olf.ghosts.recoupling import ReachabilityBuffer
from olf.ghosts.trajectory import make_ghost
from olf.invention import InventionGenerator
from olf.organism import Organism


def _population(*ghosts):
    population = GhostPopulation.empty(ghosts[0].anchor.numel(), len(ghosts))
    population._ghosts = list(ghosts)
    return population


def _rotation(dimension):
    orthogonal, _ = torch.linalg.qr(torch.randn(dimension, dimension))
    return orthogonal


def test_tension_requires_two_grounded_alternatives():
    anchor = project_to_sphere(torch.randn(8))
    tangent = project_to_tangent(anchor, torch.randn(8))
    ghost = make_ghost(anchor, tangent, credibility=1.0, grounding=1.0)
    assert not _population(ghost).tension(1.0)["defined"]

    ungrounded = make_ghost(anchor, -tangent, credibility=1.0, grounding=0.0)
    assert not _population(ghost, ungrounded).tension(1.0)["defined"]


def test_credibility_is_observed_positive_evidence_fraction():
    anchor = project_to_sphere(torch.randn(8))
    tangent = 0.2 * project_to_tangent(anchor, torch.randn(8))
    ghost = make_ghost(
        anchor,
        tangent,
        credibility=0.5,
        evidence_support=0.3,
        evidence_negative=0.1,
    )
    observed = ghost.predicted_anchor(1.0)
    updated = update_after_recoupling(
        ghost,
        observed,
        1.0,
        torch.tensor(0.4),
    )
    expected = updated.evidence_support / (
        updated.evidence_support + updated.evidence_negative
    )
    assert torch.allclose(updated.credibility, expected)

    contradicted = update_after_recoupling(
        updated,
        -observed,
        1.0,
        torch.tensor(0.0),
    )
    expected = contradicted.evidence_support / (
        contradicted.evidence_support + contradicted.evidence_negative
    )
    assert torch.allclose(contradicted.credibility, expected)
    assert contradicted.credibility < updated.credibility


def test_horizon_expresses_nonlocal_future_as_local_correction():
    anchor = project_to_sphere(torch.randn(8))
    tangent = 0.4 * project_to_tangent(anchor, torch.randn(8))
    ghost = make_ghost(anchor, tangent, horizon_expr=0.25)
    expected = ghost.with_updates(horizon_expr=torch.tensor(1.0)).predicted_anchor(
        0.25
    )
    torch.testing.assert_close(ghost.predicted_anchor(1.0), expected)


def test_tension_is_set_and_frame_invariant():
    torch.manual_seed(41)
    anchor = project_to_sphere(torch.randn(8))
    first = project_to_tangent(anchor, torch.randn(8))
    second = project_to_tangent(anchor, torch.randn(8))
    ghosts = (
        make_ghost(anchor, first, credibility=0.8, grounding=0.7),
        make_ghost(anchor, second, credibility=0.6, grounding=0.9),
    )
    original = _population(*ghosts).tension(1.0)
    permuted = _population(*reversed(ghosts)).tension(1.0)
    rotation = _rotation(8)
    rotated = _population(
        *(
            make_ghost(
                rotation @ ghost.anchor,
                rotation @ ghost.tangent,
                credibility=float(ghost.credibility),
                grounding=float(ghost.grounding),
                uncertainty=float(ghost.uncertainty),
            )
            for ghost in ghosts
        )
    ).tension(1.0)
    assert original["defined"]
    assert original["energy"] > 0.0
    assert abs(original["normalized"] - permuted["normalized"]) < 1e-6
    assert abs(original["normalized"] - rotated["normalized"]) < 1e-5


def test_birth_requires_each_predictive_interval_to_fail():
    anchor = project_to_sphere(torch.randn(8))
    direction = project_to_tangent(anchor, torch.randn(8))
    direction = direction / direction.norm()
    ghosts = (
        make_ghost(anchor, 0.2 * direction, uncertainty=0.1),
        make_ghost(anchor, 0.5 * direction, uncertainty=0.4),
    )
    population = _population(*ghosts)
    population.capacity = 3
    report = LifecycleReport()
    result = maybe_birth(
        population,
        anchor,
        anchor,
        GhostConfig(latent_dim=8, capacity=3),
        report,
    )
    assert report.born
    assert len(result) == 3


def test_schema_needs_repetition_and_predictive_advantage():
    anchor = project_to_sphere(torch.randn(8))
    action = torch.tensor([0.4, -0.2, 0.1])
    tangent = project_to_tangent(anchor, torch.randn(8))
    buffer = ReachabilityBuffer(8)
    buffer.add(anchor, action, tangent)
    assert not buffer.consolidate_latest(1.0)["consolidated"]
    buffer.add(anchor, action, tangent)
    assert not buffer.consolidate_latest(0.0)["consolidated"]
    assert buffer.schemas(anchor, 4) == []

    result = buffer.consolidate_latest(0.5)
    schemas = buffer.schemas(anchor, 4)
    assert result["consolidated"]
    assert len(schemas) == 1
    assert len(schemas[0]["evidence"]) == 2
    assert schemas[0]["support"] > 0.0


def test_grounded_transfer_can_invert_its_observed_deformation():
    anchor = project_to_sphere(torch.randn(8))
    actions = torch.cat([torch.zeros(1, 3), torch.eye(3)])
    action_effects = torch.stack(
        [project_to_tangent(anchor, torch.randn(8)) for _ in range(3)]
    )
    drift = project_to_tangent(anchor, torch.randn(8))
    effects = torch.cat(
        [drift.unsqueeze(0), drift.unsqueeze(0) + action_effects], dim=0
    )
    ghost = make_ghost(anchor, effects[0])
    for action, effect in zip(actions, effects, strict=True):
        ghost = ghost.add_action_evidence(anchor, action, effect)
    wanted_action = torch.tensor([0.2, -0.4, 0.3])
    wanted_effect = drift + wanted_action @ action_effects
    recovered = ghost.transfer_inverse(wanted_effect, anchor)
    reconstructed = ghost.transfer_predict(recovered, anchor)
    assert torch.allclose(recovered, wanted_action, atol=1e-4, rtol=1e-4)
    assert torch.allclose(reconstructed, wanted_effect, atol=1e-4, rtol=1e-4)


def test_consolidated_schema_reappears_after_episode_reset():
    config = GhostConfig(
        ghost_mode="influence", latent_dim=8, action_dim=3, capacity=4
    )
    organism = Organism(
        obs_dim=18,
        latent_dim=8,
        hidden_dim=16,
        ghost_mode="influence",
        ghost_config=config,
    )
    anchor = project_to_sphere(torch.randn(8))
    action = torch.tensor([0.4, -0.2, 0.1])
    tangent = project_to_tangent(anchor, torch.randn(8))
    organism.ghost.buffer.add(anchor, action, tangent)
    organism.ghost.buffer.add(anchor, action, tangent)
    organism.ghost.buffer.consolidate_latest(0.5)

    organism.ghost.reset()
    organism.ghost.begin_step(anchor)
    assert len(organism.ghost.population) == 1
    restored = organism.ghost.population[0]
    assert float(restored.grounding) > 0.0
    assert len(restored.transfer_actions) == 2


def _alternating_schema_buffer(rotation=None, separate_edges=False):
    anchor = project_to_sphere(torch.randn(8))
    first = project_to_tangent(anchor, torch.randn(8))
    first = 0.1 * first / first.norm()
    second = project_to_tangent(anchor, torch.randn(8))
    second = second - (second @ first) / (first @ first) * first
    second = 0.1 * second / second.norm()
    if rotation is not None:
        anchor = rotation @ anchor
        first = rotation @ first
        second = rotation @ second
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


def test_recursive_schema_branch_requires_external_predictive_evidence():
    buffer, anchor = _alternating_schema_buffer()
    branches = buffer.composed_schemas(anchor, 4)
    assert any(branch["depth"] == 2 for branch in branches)

    separated, separated_anchor = _alternating_schema_buffer(
        separate_edges=True
    )
    separated_branches = separated.composed_schemas(separated_anchor, 4)
    assert all(branch["depth"] == 1 for branch in separated_branches)


def test_composed_schema_restores_inverse_depth_horizon():
    buffer, anchor = _alternating_schema_buffer()
    config = GhostConfig(
        ghost_mode="influence", latent_dim=8, action_dim=3, capacity=4
    )
    organism = Organism(
        obs_dim=18,
        latent_dim=8,
        hidden_dim=16,
        ghost_mode="influence",
        ghost_config=config,
    )
    organism.ghost.buffer = buffer
    organism.ghost.reset()
    schemas = buffer.composed_schemas(anchor, config.capacity)
    organism.ghost.begin_step(anchor)
    assert len(organism.ghost.population) == len(schemas)
    for ghost, schema in zip(
        organism.ghost.population._ghosts, schemas, strict=True
    ):
        assert abs(float(ghost.horizon_expr) - 1.0 / schema["depth"]) < 1e-7


def test_recursive_schema_composition_is_frame_equivariant():
    torch.manual_seed(83)
    buffer, anchor = _alternating_schema_buffer()
    rotation = _rotation(8)

    rotated = ReachabilityBuffer(16)
    for source, action, tangent in zip(
        buffer.anchors, buffer.actions, buffer.prototypes, strict=True
    ):
        rotated.add(rotation @ source, action, rotation @ tangent)
        rotated.consolidate_latest(1.0)
    original_branch = next(
        branch
        for branch in buffer.composed_schemas(anchor, 4)
        if branch["depth"] == 2
    )
    rotated_branch = next(
        branch
        for branch in rotated.composed_schemas(rotation @ anchor, 4)
        if branch["depth"] == 2
    )
    assert torch.allclose(
        rotated_branch["tangent"],
        rotation @ original_branch["tangent"],
        atol=1e-5,
        rtol=1e-5,
    )


def test_lifecycle_evicts_only_externally_dominated_ghosts():
    anchor = project_to_sphere(torch.randn(8))
    tangent = project_to_tangent(anchor, torch.randn(8))
    weaker = make_ghost(
        anchor,
        tangent,
        credibility=0.2,
        grounding=0.1,
        uncertainty=0.8,
        persistence=1.0,
        evidence_support=0.1,
        evidence_negative=0.7,
    )
    stronger = make_ghost(
        anchor,
        -tangent,
        credibility=0.8,
        grounding=0.7,
        uncertainty=0.2,
        persistence=1.0,
        evidence_support=0.9,
        evidence_negative=0.1,
    )
    population = _population(weaker, stronger)
    report = LifecycleReport()
    result = evict(
        population, GhostConfig(latent_dim=8, capacity=2), report
    )
    assert len(result) == 1
    assert result[0] is stronger
    assert report.evicted == [0]


def test_lifecycle_merges_equivalent_but_not_distinct_futures():
    anchor = project_to_sphere(torch.randn(8))
    tangent = 0.1 * project_to_tangent(anchor, torch.randn(8))
    first = make_ghost(
        anchor,
        tangent,
        grounding=0.5,
        credibility=0.7,
        evidence_support=0.4,
    )
    duplicate = first.with_updates(
        evidence_support=first.evidence_support.clone()
    )
    distinct = make_ghost(
        anchor,
        -tangent,
        grounding=0.5,
        credibility=0.7,
        evidence_support=0.4,
    )
    population = _population(first, duplicate, distinct)
    report = LifecycleReport()
    result = merge_similar(
        population, GhostConfig(latent_dim=8, capacity=3), report
    )
    assert report.merged
    assert len(result) == 2
    torch.testing.assert_close(
        result[0].evidence_support,
        first.evidence_support + duplicate.evidence_support,
    )


def test_pareto_selection_has_no_hidden_scalarization():
    objectives = [(0.0, 1.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 0.0)]
    assert _non_dominated_indices(objectives) == [0, 1, 2]
    assert _non_dominated_indices(objectives + [(2.0, 2.0, 2.0)]) == [0, 1, 2]


def test_real_organism_reaches_cross_ghost_synthesis():
    torch.manual_seed(7)
    config = GhostConfig(
        ghost_mode="influence", latent_dim=8, action_dim=3, capacity=4
    )
    organism = Organism(
        obs_dim=18,
        latent_dim=8,
        hidden_dim=16,
        ghost_mode="influence",
        ghost_config=config,
    )
    for sign in (1.0, -1.0):
        ghost = make_ghost(
            torch.randn(8),
            sign * torch.randn(8),
            grounding=1.0,
            credibility=1.0,
            evidence_support=1.0,
        )
        ghost = ghost.add_action_evidence(
            ghost.anchor,
            torch.tensor([0.5, 0.0, 0.0]),
            sign * torch.randn(8),
        )
        ghost = ghost.add_action_evidence(
            ghost.anchor,
            torch.tensor([-0.3, 0.4, 0.1]),
            sign * torch.randn(8),
        )
        organism.ghost.population.append(ghost)

    _, info = organism.select_action(torch.randn(18).numpy(), evaluate=True)
    ghost_info = info["ghost"]
    assert ghost_info["routing"] == "pareto_synthesis"
    assert ghost_info["synthesis_model_count"] == 2
    assert info["ghost_tension"]["defined"]
    assert info["impasse_pressure"] >= info["ghost_tension"]["normalized"]


class _Memory:
    def __init__(self, reward):
        self.reward = reward
        self.trace_active = torch.ones(1)

    def retrieve_trace(self, _situated, action):
        sign = float(action[0, 0].detach())
        consequence = torch.tensor(
            [[self.reward, max(-sign, 0.0), -sign, sign]]
        )
        return torch.zeros(1, 8), consequence, 0.5


def test_invention_is_invariant_to_reward_channel():
    invention = InventionGenerator(latent_dim=8, action_dim=3, hidden_dim=16)
    with torch.no_grad():
        for parameter in invention.parameters():
            parameter.zero_()
        invention.proposal_net[-2].bias.copy_(torch.tensor([0.4, -0.2, 0.1]))
    h = torch.randn(1, 8)
    situated = torch.randn(1, 2, 16)
    positive = invention(h, situated, _Memory(1000.0), num_candidates=2)
    negative = invention(h, situated, _Memory(-1000.0), num_candidates=2)
    assert torch.equal(positive, negative)
