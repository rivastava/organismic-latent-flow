"""Tests for training optimizer parameter grouping."""

from experiments.run_core import build_training_param_groups
from olf.baselines import MLPBaselineAgent
from olf.organism import Organism


def _param_ids(params):
    return {id(p) for p in params}


def test_flc_parameters_have_separate_low_lr_group():
    agent = Organism(obs_dim=18, action_dim=3)
    groups = build_training_param_groups(agent, lr=0.01)

    assert [group["lr"] for group in groups] == [0.0001, 0.001, 0.00001]

    policy_ids = _param_ids(agent.movement_policy.parameters())
    flc_ids = _param_ids(agent.flc.parameters())
    grouped = [_param_ids(group["params"]) for group in groups]

    assert grouped[0] == policy_ids
    assert grouped[2] == flc_ids
    assert not grouped[1] & policy_ids
    assert not grouped[1] & flc_ids


def test_non_flc_agent_uses_single_group():
    agent = MLPBaselineAgent(obs_dim=18, action_dim=3)
    groups = build_training_param_groups(agent, lr=0.01)

    assert len(groups) == 1
    assert groups[0]["lr"] == 0.001
    assert _param_ids(groups[0]["params"]) == _param_ids(agent.parameters())
