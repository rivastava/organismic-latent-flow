import torch

from benchmarks.target_threat import TargetThreatEnv
from olf.events import entity_feature_event_mask
from olf.organism import Organism
from olf.seeding import set_seed


def test_entity_events_ignore_motion_and_detect_content_change():
    before = torch.tensor(
        [[[1.0, 0.0], [0.0, 1.0]]], dtype=torch.float32
    )
    unchanged = before.clone()
    changed = before.clone()
    changed[0, 1] = 0.0

    assert not bool(entity_feature_event_mask(before, unchanged).any())
    assert entity_feature_event_mask(before, changed).tolist() == [[False, True]]


def test_organism_records_feature_transition_without_coordinates():
    set_seed(21)
    agent = Organism(obs_dim=18, action_dim=3)
    agent.eval()
    obs = TargetThreatEnv(seed=21).reset()
    agent.reset_state()
    agent.select_action(obs, evaluate=True)
    next_obs = obs.copy()
    next_obs[0:2] = [0.4, -0.4]
    next_obs[6:8] = [-0.2, 0.8]
    next_obs[8:12] = 0.0

    agent.learn_consequence(
        0.0, 0.0, 0.0, 0.0, next_obs=next_obs, store=True
    )

    assert agent.last_entity_event_mask.tolist() == [True, False]
    assert int(agent.consequence_events_seen.item()) == 1
    assert agent.last_observed_effect.norm().item() > 0.0


def test_frozen_evaluation_does_not_accumulate_event_evidence():
    set_seed(22)
    agent = Organism(obs_dim=18, action_dim=3)
    agent.eval()
    obs = TargetThreatEnv(seed=22).reset()
    agent.reset_state()
    agent.select_action(obs, evaluate=True)
    next_obs = obs.copy()
    next_obs[14:18] = 0.0

    agent.learn_consequence(
        0.0, 0.0, 0.0, 0.0, next_obs=next_obs, store=False
    )

    assert agent.last_entity_event_mask.tolist() == [False, True]
    assert int(agent.consequence_events_seen.item()) == 0
