import torch

from benchmarks.target_threat import TargetThreatEnv
from olf.organism import Organism
from olf.seeding import set_seed


def test_policy_score_does_not_train_semantic_binding_directly():
    set_seed(31)
    agent = Organism(obs_dim=18, action_dim=3)
    obs = TargetThreatEnv(seed=31).reset()
    agent.train()
    agent.reset_state()

    _, info = agent.select_action(obs, evaluate=False)
    (-info["_policy_log_prob"].sum()).backward()

    movement_grads = [
        parameter.grad for parameter in agent.movement_policy.parameters()
    ]
    semantic_grads = [
        parameter.grad for parameter in agent.semantics.parameters()
    ]
    assert any(
        grad is not None and torch.count_nonzero(grad).item() > 0
        for grad in movement_grads
    )
    assert all(grad is None for grad in semantic_grads)
