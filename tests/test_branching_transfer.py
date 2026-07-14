"""Tests for transfer-law ghost discovery."""

from __future__ import annotations

import inspect

import torch

import experiments.branching_latent_flow.transfer_ghosts as transfer_ghosts
from experiments.branching_latent_flow.protocol import ProtocolConfig
from experiments.branching_latent_flow.observations import observed_examples, stage1_trials
from experiments.branching_latent_flow.transfer_ghosts import fit_transfer_ghosts


def test_transfer_learner_has_no_hidden_world_or_reward_channel():
    source = inspect.getsource(transfer_ghosts)
    for forbidden in ("mode_index", "all_mode_endpoints", "n_modes", "reward", "success"):
        assert forbidden not in source


def test_transfer_population_starts_small_and_is_capacity_bounded():
    config = ProtocolConfig(n_modes=2, train_trials=120, eval_trials=20, seed=71)
    contexts, probes, endpoints = observed_examples(stage1_trials(config, split="train"))
    field = fit_transfer_ghosts(contexts, probes, endpoints, capacity=5, seed=9)
    assert 1 <= len(field.matrices) <= 5
    assert len(field.events) <= 4


def test_transfer_ghosts_produce_distinct_spherical_futures():
    config = ProtocolConfig(n_modes=2, train_trials=180, eval_trials=20, seed=72)
    train = stage1_trials(config, split="train")
    contexts, probes, endpoints = observed_examples(train)
    field = fit_transfer_ghosts(contexts, probes, endpoints, capacity=5, seed=10)
    belief = field.belief(contexts[0])
    assert torch.allclose(belief.endpoints.norm(dim=-1), torch.ones(len(belief.weights)), atol=1e-5)
    assert torch.isclose(belief.weights.sum(), torch.tensor(1.0))
    if len(belief.endpoints) > 1:
        assert not torch.allclose(belief.endpoints[0], belief.endpoints[1])


def test_two_law_world_discovers_two_ghosts_without_count_input():
    config = ProtocolConfig(n_modes=2, train_trials=300, eval_trials=20, seed=4040)
    contexts, probes, endpoints = observed_examples(stage1_trials(config, split="train"))
    field = fit_transfer_ghosts(contexts, probes, endpoints, capacity=8, seed=0)
    assert len(field.matrices) == 2
    assert field.events[0].accepted
    assert not field.events[-1].accepted
