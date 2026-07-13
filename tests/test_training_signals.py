import numpy as np
import pytest
import torch

from experiments.run_core import ENV_MAP, _policy_learning_signal, train_agent
from olf.organism import Organism
from olf.seeding import set_seed


class _EndogenousEventEnv:
    """One-step event whose dynamics are independent of benchmark reward."""

    reward_value = 0.0

    def __init__(self, seed=None):
        self.seed = seed

    @staticmethod
    def _obs(*, after_event):
        obs = np.zeros(18, dtype=np.float32)
        obs[2:4] = [0.2, 0.1] if after_event else [0.8, 0.1]
        obs[6:8] = [0.5, 0.5]
        obs[8:12] = [0.0, 1.0, 0.0, 0.0] if after_event else [1.0, 0.0, 0.0, 0.0]
        obs[12:14] = [-0.5, -0.5]
        obs[14:18] = [0.0, 0.0, 1.0, 0.0]
        return obs

    def reset(self):
        return self._obs(after_event=False)

    def step(self, action):
        del action
        return (
            self._obs(after_event=True),
            self.reward_value,
            True,
            {"status": "success"},
        )


class _PositiveRewardEventEnv(_EndogenousEventEnv):
    reward_value = 10_000.0


class _NegativeRewardEventEnv(_EndogenousEventEnv):
    reward_value = -10_000.0


def _train_reward_blind_event_agent(env_name, env_class):
    ENV_MAP[env_name] = env_class
    try:
        set_seed(19)
        agent = Organism(
            obs_dim=18,
            action_dim=3,
            use_hierarchical_intent=True,
            hierarchical_intent_std=0.5,
            hierarchical_intent_blend=0.8,
            hierarchical_babble_probability=1.0,
            use_prospective_event_grounding=True,
        )
        train_agent(
            agent,
            env_name,
            num_episodes=3,
            seed=19,
            training_signal="terminal_homeostasis",
            credit_mode="uniform",
        )
        return agent
    finally:
        del ENV_MAP[env_name]


def test_terminal_viability_ignores_raw_reward_and_predicted_value():
    positive = _policy_learning_signal(
        1000.0,
        done=True,
        was_lethal=0.0,
        predicted_value=99.0,
        training_signal="terminal_viability",
    )
    negative = _policy_learning_signal(
        -1000.0,
        done=True,
        was_lethal=0.0,
        predicted_value=-99.0,
        training_signal="terminal_viability",
    )
    assert positive == negative == 1.0


def test_terminal_viability_is_zero_before_terminal_and_negative_on_death():
    assert _policy_learning_signal(
        1.0,
        done=False,
        was_lethal=0.0,
        training_signal="terminal_viability",
    ) == 0.0
    assert _policy_learning_signal(
        1.0,
        done=True,
        was_lethal=1.0,
        training_signal="terminal_viability",
    ) == -1.0


def test_legacy_and_raw_reward_modes_are_explicitly_distinct():
    assert _policy_learning_signal(
        2.0,
        done=False,
        was_lethal=0.0,
        predicted_value=5.0,
        training_signal="legacy_reward",
    ) == pytest.approx(2.1)
    assert _policy_learning_signal(
        2.0,
        done=False,
        was_lethal=0.0,
        predicted_value=5.0,
        training_signal="raw_reward",
    ) == 2.0


def test_homeostatic_delta_is_reward_blind_and_tracks_body_relief():
    kwargs = {
        "done": False,
        "was_lethal": 0.0,
        "training_signal": "homeostatic_delta",
        "self_state": [0.7, 0.2],
        "next_self_state": [0.2, 0.2],
    }
    assert _policy_learning_signal(1000.0, **kwargs) == pytest.approx(0.5)
    assert _policy_learning_signal(-1000.0, **kwargs) == pytest.approx(0.5)


def test_homeostatic_delta_penalizes_drift_and_lethal_collapse():
    drift = _policy_learning_signal(
        0.0,
        done=False,
        was_lethal=0.0,
        training_signal="homeostatic_delta",
        self_state=[0.5, 0.5],
        next_self_state=[0.52, 0.51],
    )
    lethal = _policy_learning_signal(
        0.0,
        done=True,
        was_lethal=1.0,
        training_signal="homeostatic_delta",
        self_state=[0.4, 0.3],
        next_self_state=[0.4, 0.3],
    )
    assert drift == pytest.approx(-0.03)
    assert lethal == pytest.approx(-1.3)


def test_terminal_homeostasis_uses_absolute_body_state_and_death_boundary():
    alive = _policy_learning_signal(
        999.0,
        done=True,
        was_lethal=0.0,
        training_signal="terminal_homeostasis",
        self_state=[0.7, 0.2],
        next_self_state=[0.2, 0.2],
    )
    running = _policy_learning_signal(
        999.0,
        done=False,
        was_lethal=0.0,
        training_signal="terminal_homeostasis",
        next_self_state=[0.2, 0.2],
    )
    dead = _policy_learning_signal(
        999.0,
        done=True,
        was_lethal=1.0,
        training_signal="terminal_homeostasis",
        next_self_state=[0.2, 0.2],
    )
    assert alive == pytest.approx(0.6)
    assert running == 0.0
    assert dead == -1.0


def test_unknown_training_signal_fails_loudly():
    with pytest.raises(ValueError, match="unknown training_signal"):
        _policy_learning_signal(
            0.0,
            done=False,
            was_lethal=0.0,
            training_signal="shortcut",
        )


def test_terminal_homeostasis_training_is_end_to_end_reward_blind():
    positive = _train_reward_blind_event_agent(
        "_positive_reward_event", _PositiveRewardEventEnv
    )
    negative = _train_reward_blind_event_agent(
        "_negative_reward_event", _NegativeRewardEventEnv
    )

    assert int(positive.prospective_events_seen.item()) > 0
    assert len(positive.prospective_event_memory) > 0
    positive_state = positive.state_dict()
    negative_state = negative.state_dict()
    assert positive_state.keys() == negative_state.keys()
    for name in positive_state:
        assert torch.equal(positive_state[name], negative_state[name]), name


def test_disabled_prospective_path_stays_dormant():
    ENV_MAP["_dormant_event"] = _PositiveRewardEventEnv
    try:
        set_seed(23)
        agent = Organism(obs_dim=18, action_dim=3)
        field_before = {
            name: value.detach().clone()
            for name, value in agent.prospective_event_field.state_dict().items()
        }

        train_agent(
            agent,
            "_dormant_event",
            num_episodes=2,
            seed=23,
            training_signal="terminal_homeostasis",
        )
    finally:
        del ENV_MAP["_dormant_event"]

    assert int(agent.prospective_events_seen.item()) == 0
    assert len(agent.prospective_event_memory) == 0
    field_after = agent.prospective_event_field.state_dict()
    for name in field_before:
        assert torch.equal(field_before[name], field_after[name]), name
