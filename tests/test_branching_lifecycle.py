"""Tests for active/dormant/free transfer-ghost lifecycle."""

from __future__ import annotations

import inspect

import experiments.branching_latent_flow.persistent_ghosts as persistent_ghosts
from experiments.branching_latent_flow.persistent_ghosts import PersistentTransferGhostField
from experiments.branching_latent_flow.observations import observed_examples
from experiments.branching_latent_flow.changing_world import scheduled_trials


def _update(field, seed, counts, phase):
    contexts, probes, endpoints = observed_examples(
        scheduled_trials(seed=seed, counts=counts, split_offset=phase)
    )
    field.update(contexts, probes, endpoints, seed=seed + phase)


def test_lifecycle_learner_has_no_hidden_world_or_reward_channel():
    source = inspect.getsource(persistent_ghosts)
    for forbidden in ("mode_index", "all_mode_endpoints", "n_modes", "reward", "success"):
        assert forbidden not in source


def test_disappearing_ghost_becomes_dormant_and_returns_with_same_identity():
    field = PersistentTransferGhostField(latent_dim=8, capacity=8)
    _update(field, 5150, {0: 120, 1: 120, 2: 30}, 0)
    initial = set(field.active_identities())
    assert len(initial) == 3
    _update(field, 5150, {0: 120, 1: 120, 3: 30}, 1)
    dormant = set(field.dormant_identities())
    assert len(dormant & initial) == 1
    assert len(field.active_identities()) == 3
    _update(field, 5150, {0: 100, 1: 100, 2: 35, 3: 35}, 2)
    assert dormant <= set(field.active_identities())
    assert len(field.active_identities()) == 4
    assert len(field.slots) <= field.capacity
