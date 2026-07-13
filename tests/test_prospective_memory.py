import inspect

import torch
import torch.nn.functional as F

from benchmarks.target_threat import TargetThreatEnv
from olf.organism import Organism
from olf.prospective_memory import ProspectiveEventMemory
from olf.seeding import set_seed


def _add(memory, index, entity_index=0):
    memory.add(
        observation=torch.full((18,), float(index)),
        spm_trace=F.normalize(torch.arange(8, dtype=torch.float32) + index, dim=0),
        entity_index=entity_index,
        endpoint=F.normalize(torch.arange(8, dtype=torch.float32) - index, dim=0),
        future_value=float(index),
        risk=float(index % 2),
        action=torch.tensor([index, -index, 0.0]),
        horizon=index + 1,
    )


def test_prospective_memory_is_bounded_and_has_no_reward_channel():
    memory = ProspectiveEventMemory(
        obs_dim=18, latent_dim=8, action_dim=3, max_records=3
    )
    for index in range(5):
        _add(memory, index)

    assert len(memory) == 3
    assert int(memory.write_index.item()) == 2
    assert "reward" not in inspect.signature(memory.add).parameters


def test_content_read_is_equivariant_to_query_slot_permutation():
    memory = ProspectiveEventMemory(
        obs_dim=18, latent_dim=8, action_dim=3, max_records=4
    )
    _add(memory, 1)
    _add(memory, 2)
    keys = F.normalize(torch.randn(2, 6), dim=-1)
    query = torch.stack([keys[0], keys[1]], dim=0).unsqueeze(0)

    original = memory.read(query, keys, top_k=1)
    permutation = torch.tensor([1, 0])
    permuted = memory.read(query[:, permutation], keys, top_k=1)

    for name in ("future_latent", "value", "risk", "action", "support"):
        assert torch.allclose(
            original[name][:, permutation], permuted[name]
        )


def test_organism_reencodes_memory_keys_with_current_semantics():
    set_seed(26)
    agent = Organism(
        obs_dim=18,
        action_dim=3,
        use_prospective_event_grounding=True,
    )
    obs = TargetThreatEnv(seed=26).reset()
    obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
    _, self_state, context, positions, features = agent.parse_obs(obs_t)
    sigma = agent.semantics.bind(
        agent.spm.get_trace(), positions, features, context, self_state
    )
    agent.prospective_event_memory.add(
        observation=obs,
        spm_trace=agent.spm.get_trace(),
        entity_index=0,
        endpoint=agent.h,
        future_value=0.5,
        risk=0.0,
        action=torch.tensor([0.2, 0.3, 0.0]),
        horizon=3,
    )

    first = agent._read_prospective_memory(sigma.detach())
    with torch.no_grad():
        agent.semantics.pre_binder[0].bias.add_(0.5)
    sigma_after = agent.semantics.bind(
        agent.spm.get_trace(), positions, features, context, self_state
    )
    second = agent._read_prospective_memory(sigma_after.detach())

    assert first is not None and second is not None
    assert first["support"][0, 0] > 0.0
    assert second["support"][0, 0] > 0.0
    assert torch.allclose(
        first["action"][0, 0], torch.tensor([0.2, 0.3, 0.0])
    )


def test_situated_key_separates_body_and_context_without_role_labels():
    sigma = torch.ones(1, 2, 6)
    key_a = Organism._situated_memory_key(
        sigma,
        torch.tensor([[0.9, 0.1]]),
        torch.tensor([[1.0, 0.0]]),
    )
    key_b = Organism._situated_memory_key(
        sigma,
        torch.tensor([[0.1, 0.9]]),
        torch.tensor([[1.0, 0.0]]),
    )
    key_c = Organism._situated_memory_key(
        sigma,
        torch.tensor([[0.9, 0.1]]),
        torch.tensor([[0.0, 1.0]]),
    )

    assert not torch.allclose(key_a, key_b)
    assert not torch.allclose(key_a, key_c)
    assert torch.allclose(key_a.norm(dim=-1), torch.ones(1, 2))
