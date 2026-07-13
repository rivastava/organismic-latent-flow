import copy

import numpy as np
import torch

from benchmarks.target_threat import TargetThreatEnv
from olf.geometry import log_map_sphere
from olf.organism import Organism
from olf.seeding import set_seed


def _prepared_agents():
    set_seed(23)
    base = Organism(obs_dim=18, action_dim=3)
    return copy.deepcopy(base), copy.deepcopy(base)


def test_log_map_identity_is_zero_and_outputs_are_tangent():
    x = torch.tensor([[1.0, 0.0, 0.0]])
    assert torch.equal(log_map_sphere(x, x), torch.zeros_like(x))

    y = torch.tensor([[0.0, 1.0, 0.0]])
    tangent = log_map_sphere(x, y)
    assert torch.allclose((x * tangent).sum(dim=-1), torch.zeros(1), atol=1e-7)


def test_consequence_trace_depends_on_the_actual_next_observation():
    agent_a, agent_b = _prepared_agents()
    obs = TargetThreatEnv(seed=4).reset()

    set_seed(91)
    agent_a.select_action(obs, evaluate=True)
    set_seed(91)
    agent_b.select_action(obs, evaluate=True)
    h_action = agent_a._h_at_action.clone()
    assert torch.equal(h_action, agent_b._h_at_action)

    next_a = obs.copy()
    next_b = obs.copy()
    next_a[0:4] = np.array([0.2, 0.1, 0.2, 0.2], dtype=np.float32)
    next_b[0:4] = np.array([-0.8, 0.7, 0.95, 0.95], dtype=np.float32)

    agent_a.learn_consequence(0.0, 0.0, -0.3, -0.3, next_obs=next_a)
    agent_b.learn_consequence(0.0, 0.0, 0.05, 0.05, next_obs=next_b)

    assert torch.equal(agent_a.consequence_memory.trace_s_before[0], h_action[0])
    assert torch.equal(agent_b.consequence_memory.trace_s_before[0], h_action[0])
    assert not torch.allclose(
        agent_a.consequence_memory.trace_s_after[0],
        agent_b.consequence_memory.trace_s_after[0],
    )
    assert torch.equal(agent_a.consequence_memory.trace_s_after[0], agent_a.h[0])
    assert torch.equal(agent_a.rtcm.history[-1]["h_next"], agent_a.h)

    effect = agent_a.rtcm.history[-1]["effect"]
    before = agent_a.rtcm.history[-1]["h"]
    assert effect is not None
    assert torch.allclose((before * effect).sum(dim=-1), torch.zeros(1), atol=1e-6)


def test_prefetched_consequence_observation_is_not_integrated_twice():
    agent, _ = _prepared_agents()
    env = TargetThreatEnv(seed=5)
    obs = env.reset()
    action, _ = agent.select_action(obs, evaluate=True)
    next_obs, reward, _done, info = env.step(action)
    lethal = float(info["status"] in ("death", "starvation"))
    agent.learn_consequence(
        reward,
        lethal,
        next_obs[2] - obs[2],
        next_obs[3] - obs[3],
        next_obs=next_obs,
    )
    recoupled = agent.h.clone()

    agent.select_action(next_obs, evaluate=True)
    assert torch.equal(agent._h_at_action, recoupled)
