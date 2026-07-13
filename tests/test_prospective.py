import torch
import torch.nn.functional as F

from benchmarks.target_threat import TargetThreatEnv
from olf.organism import Organism
from olf.prospective import ProspectiveEventField
from olf.seeding import set_seed


def _sphere(rows, dim):
    return F.normalize(torch.randn(rows, dim), p=2, dim=-1)


def test_prospective_field_outputs_sphere_endpoints_and_bounded_horizons():
    set_seed(21)
    field = ProspectiveEventField(
        latent_dim=8, sigma_dim=6, action_dim=3, max_horizon=16
    )
    h = _sphere(4, 8)
    prediction = field(h, torch.randn(4, 2, 6), torch.randn(4, 3))

    assert prediction["future_latent"].shape == (4, 2, 8)
    assert torch.allclose(
        prediction["future_latent"].norm(dim=-1),
        torch.ones(4, 2),
        atol=1e-5,
    )
    assert bool((prediction["horizon"] >= 1.0).all())
    assert bool((prediction["horizon"] <= 16.0).all())


def test_event_loss_is_entity_permutation_equivariant():
    set_seed(22)
    field = ProspectiveEventField(
        latent_dim=8, sigma_dim=6, action_dim=3, max_horizon=8
    )
    latents = _sphere(3, 8)
    endpoint = _sphere(1, 8)
    sigmas = torch.randn(3, 2, 6)
    actions = torch.randn(3, 3)
    effects = torch.randn(3, 8)
    effects = effects - (effects * latents).sum(-1, keepdim=True) * latents
    mask = torch.tensor([True, False])

    original = field.event_loss(
        latents=latents,
        sigmas=sigmas,
        actions=actions,
        effects=effects,
        endpoint=endpoint,
        entity_mask=mask,
        future_value=0.7,
        lethal=0.0,
    )
    permutation = torch.tensor([1, 0])
    permuted = field.event_loss(
        latents=latents,
        sigmas=sigmas[:, permutation],
        actions=actions,
        effects=effects,
        endpoint=endpoint,
        entity_mask=mask[permutation],
        future_value=0.7,
        lethal=0.0,
    )

    assert torch.allclose(original, permuted)


def test_event_loss_requires_an_observed_entity_transition():
    field = ProspectiveEventField(latent_dim=8, sigma_dim=6, action_dim=3)
    assert (
        field.event_loss(
            latents=_sphere(2, 8),
            sigmas=torch.randn(2, 2, 6),
            actions=torch.randn(2, 3),
            effects=torch.randn(2, 8),
            endpoint=_sphere(1, 8),
            entity_mask=torch.tensor([False, False]),
            future_value=1.0,
            lethal=0.0,
        )
        is None
    )


def test_prospective_event_model_does_not_own_the_semantic_binder():
    set_seed(23)
    agent = Organism(
        obs_dim=18,
        action_dim=3,
        latent_dim=8,
        hidden_dim=16,
        use_prospective_event_grounding=True,
    )
    obs = torch.as_tensor(
        TargetThreatEnv(seed=23).reset(), dtype=torch.float32
    ).unsqueeze(0)
    _, self_state, context, positions, features = agent.parse_obs(obs)
    spm = agent.spm.get_trace()
    sigma = agent.semantics.bind(
        spm, positions, features, context, self_state
    )
    latents = _sphere(2, 8)
    loss = agent.prospective_event_field.event_loss(
        latents=latents,
        sigmas=torch.cat([sigma, sigma], dim=0).detach(),
        actions=torch.randn(2, 3),
        effects=torch.randn(2, 8),
        endpoint=_sphere(1, 8),
        entity_mask=torch.tensor([True, False]),
        future_value=0.5,
        lethal=0.0,
    )
    loss.backward()

    assert agent.prospective_event_field.tangent_head.weight.grad is not None
    assert agent.semantics.pre_binder[0].weight.grad is None
    assert all(
        parameter.grad is None for parameter in agent.movement_policy.parameters()
    )


def test_prospective_endpoint_changes_the_flc_target_only_after_grounding():
    set_seed(24)
    agent = Organism(
        obs_dim=18,
        action_dim=3,
        use_prospective_event_grounding=True,
    )
    obs = TargetThreatEnv(seed=24).reset()
    obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
    _, self_state, context, positions, features = agent.parse_obs(obs_t)
    sigma = agent.semantics.bind(
        agent.spm.get_trace(), positions, features, context, self_state
    )
    base_action = torch.zeros(1, 3)
    with torch.no_grad():
        agent.prospective_event_field.tangent_head.weight.zero_()
        agent.prospective_event_field.tangent_head.bias.zero_()
        agent.prospective_event_field.tangent_head.bias[0] = 1.0

    agent.prospective_events_seen.zero_()
    action_ungrounded, diag_ungrounded = agent.apply_future_control(
        sigma, self_state, base_action
    )
    agent.prospective_events_seen.fill_(100)
    action_grounded, diag_grounded = agent.apply_future_control(
        sigma, self_state, base_action
    )

    assert diag_ungrounded["future_hint_confidence"].item() == 0.0
    assert diag_grounded["future_hint_confidence"].item() > 0.9
    assert not torch.allclose(action_ungrounded, action_grounded)
