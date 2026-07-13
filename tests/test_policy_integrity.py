import torch

from benchmarks.target_threat import TargetThreatEnv
from experiments.run_core import (
    _mean_one_credit_weights,
    _score_function_policy_loss,
    train_agent,
)
from olf.baselines import MLPBaselineAgent
from olf.organism import Organism
from olf.seeding import set_seed
from olf.spm import SphericalPhaseMemory


def _expected_log_prob(info):
    mean = info["_policy_distribution_mean"]
    raw_sample = info["_policy_raw_sample"]
    std = info["_policy_exploration_std"]
    return torch.distributions.Normal(mean, std).log_prob(raw_sample).sum(dim=-1)


def test_empty_spm_trace_is_stable_across_reads_and_resets():
    spm = SphericalPhaseMemory(latent_dim=8)

    first = spm.get_trace()
    second = spm.get_trace()
    spm.reset_memory()
    after_reset = spm.get_trace()

    assert torch.equal(first, second)
    assert torch.equal(first, after_reset)
    assert torch.allclose(first.norm(dim=-1), torch.ones(1))


def test_organism_reset_uses_a_stable_manifold_origin():
    set_seed(6)
    agent = Organism(obs_dim=18, action_dim=3)

    first = agent.h.clone()
    agent.reset_state()
    second = agent.h.clone()

    assert torch.equal(first, second)
    assert torch.allclose(second.norm(dim=-1), torch.ones(1))


def test_randomized_origin_remains_an_explicit_ablation():
    set_seed(6)
    agent = Organism(
        obs_dim=18, action_dim=3, randomize_initial_latent=True
    )

    first = agent.h.clone()
    agent.reset_state()

    assert not torch.equal(first, agent.h)


def test_organism_scores_the_sampled_abstract_proposal():
    set_seed(7)
    agent = Organism(obs_dim=18, action_dim=3)
    obs = TargetThreatEnv(seed=7).reset()
    agent.episode_count = 0
    agent.reset_state()

    _, info = agent.select_action(obs, evaluate=False)

    assert info["_policy_log_prob"] is not None
    assert info["_policy_log_prob"].requires_grad
    assert torch.allclose(info["_policy_log_prob"], _expected_log_prob(info))


def test_evaluation_action_has_no_policy_score():
    set_seed(8)
    agent = Organism(obs_dim=18, action_dim=3)
    obs = TargetThreatEnv(seed=8).reset()
    agent.reset_state()

    _, info = agent.select_action(obs, evaluate=True)

    assert info["_policy_log_prob"] is None
    assert info["_policy_raw_sample"] is None
    assert info["_policy_exploration_std"] == 0.0


def test_correlated_exploration_scores_its_conditional_distribution():
    set_seed(12)
    agent = Organism(
        obs_dim=18, action_dim=3, exploration_correlation=0.8
    )
    obs = TargetThreatEnv(seed=12).reset()
    agent.reset_state()

    _, first = agent.select_action(obs, evaluate=False)
    _, second = agent.select_action(obs, evaluate=False)

    assert torch.allclose(first["_policy_log_prob"], _expected_log_prob(first))
    assert torch.allclose(second["_policy_log_prob"], _expected_log_prob(second))
    assert not torch.equal(
        second["_policy_distribution_mean"], second["_policy_mean"]
    )
    assert (
        second["_policy_exploration_std"]
        < first["_policy_exploration_std"]
    )


def test_episode_intent_is_abstract_and_included_in_the_policy_score():
    set_seed(13)
    agent = Organism(
        obs_dim=18, action_dim=3, exploration_intent_scale=0.4
    )
    obs = TargetThreatEnv(seed=13).reset()
    agent.train()
    agent.reset_state()

    _, info = agent.select_action(obs, evaluate=False)

    assert torch.allclose(info["_policy_log_prob"], _expected_log_prob(info))
    assert torch.allclose(
        info["_policy_exploration_intent"].norm(dim=-1),
        torch.tensor([0.4]),
    )
    assert torch.allclose(
        info["_policy_distribution_mean"],
        info["_policy_mean"] + info["_policy_exploration_intent"],
    )


def test_hierarchical_intent_is_scored_once_and_persists():
    set_seed(14)
    agent = Organism(
        obs_dim=18,
        action_dim=3,
        use_hierarchical_intent=True,
        hierarchical_intent_std=0.7,
    )
    obs = TargetThreatEnv(seed=14).reset()
    agent.train()
    agent.reset_state()

    _, first = agent.select_action(obs, evaluate=False)
    _, second = agent.select_action(obs, evaluate=False)

    expected = torch.distributions.Normal(
        first["_episode_intent_distribution_mean"], 0.7
    ).log_prob(first["_episode_intent_raw_sample"]).sum(dim=-1)
    assert first["_intent_log_prob"] is not None
    assert first["_intent_log_prob"].requires_grad
    assert torch.allclose(first["_intent_log_prob"], expected)
    assert second["_intent_log_prob"] is None
    assert torch.equal(first["_episode_intent"], second["_episode_intent"])


def test_hierarchical_intent_score_uses_episode_return_once():
    step_scores = [
        torch.tensor([0.2], requires_grad=True),
        torch.tensor([0.3], requires_grad=True),
    ]
    intent_score = torch.tensor([0.4], requires_grad=True)
    advantages = torch.tensor([5.0, 2.0])

    loss = _score_function_policy_loss(
        step_scores,
        advantages,
        blame_weights=[1.5, 0.5],
        intent_log_prob=intent_score,
    )

    expected = -(0.2 * 5.0 * 1.5 + 0.3 * 2.0 * 0.5 + 0.4 * 5.0)
    assert torch.allclose(loss, torch.tensor(expected))
    loss.backward()
    assert torch.allclose(intent_score.grad, torch.tensor([-5.0]))


def test_renewed_intentions_use_return_from_their_own_start_time():
    first = torch.tensor([0.4], requires_grad=True)
    second = torch.tensor([0.6], requires_grad=True)
    loss = _score_function_policy_loss(
        [torch.tensor([0.0]), torch.tensor([0.0])],
        torch.tensor([5.0, 2.0]),
        intent_scores=[(0, first), (1, second)],
    )

    loss.backward()
    assert torch.allclose(first.grad, torch.tensor([-5.0]))
    assert torch.allclose(second.grad, torch.tensor([-2.0]))


def test_policy_independent_babbling_has_no_high_level_policy_score():
    set_seed(15)
    agent = Organism(
        obs_dim=18,
        action_dim=3,
        use_hierarchical_intent=True,
        hierarchical_babble_probability=1.0,
    )
    obs = TargetThreatEnv(seed=15).reset()
    agent.train()
    agent.reset_state()

    _, first = agent.select_action(obs, evaluate=False)
    _, second = agent.select_action(obs, evaluate=False)

    assert first["_episode_intent_source"] == "babble"
    assert first["_intent_log_prob"] is None
    assert second["_intent_log_prob"] is None
    assert torch.equal(first["_episode_intent"], second["_episode_intent"])
    assert bool((first["_episode_intent"].abs() <= 1.0).all())


def test_intention_can_renew_at_an_event_boundary():
    set_seed(16)
    agent = Organism(
        obs_dim=18,
        action_dim=3,
        use_hierarchical_intent=True,
        hierarchical_babble_probability=1.0,
    )
    obs = TargetThreatEnv(seed=16).reset()
    agent.train()
    agent.reset_state()
    _, first = agent.select_action(obs, evaluate=False)

    agent.renew_episode_intent()
    _, renewed = agent.select_action(obs, evaluate=False)

    assert first["_episode_intent_source"] == "babble"
    assert renewed["_episode_intent_source"] == "babble"
    assert not torch.equal(
        first["_episode_intent"], renewed["_episode_intent"]
    )


def test_mlp_baseline_scores_its_actual_exploration_distribution():
    set_seed(9)
    agent = MLPBaselineAgent(obs_dim=18, action_dim=3)
    obs = TargetThreatEnv(seed=9).reset()

    _, info = agent.select_action(obs, evaluate=False)

    assert info["_policy_log_prob"].requires_grad
    assert info["_policy_exploration_std"] == 0.3
    assert torch.allclose(info["_policy_log_prob"], _expected_log_prob(info))


def test_train_agent_preserves_rtcm_history_until_training(monkeypatch):
    set_seed(10)
    agent = Organism(obs_dim=18, action_dim=3)
    reset_calls = 0
    history_lengths = []
    original_reset = agent.reset_state
    original_train_step = agent.rtcm.train_step

    def counted_reset():
        nonlocal reset_calls
        reset_calls += 1
        return original_reset()

    def observed_train_step(*args, **kwargs):
        history_lengths.append(len(agent.rtcm.history))
        return original_train_step(*args, **kwargs)

    monkeypatch.setattr(agent, "reset_state", counted_reset)
    monkeypatch.setattr(agent.rtcm, "train_step", observed_train_step)

    train_agent(agent, "target_threat", num_episodes=1, seed=10)

    assert reset_calls == 1
    assert history_lengths and history_lengths[0] > 0


def test_rtcm_credit_reweighting_preserves_policy_gradient_scale():
    weights = _mean_one_credit_weights([0.1, 0.2, 0.7], length=3)

    assert len(weights) == 3
    assert all(weight >= 0.0 for weight in weights)
    assert abs(sum(weights) - 3.0) < 1e-8
    assert _mean_one_credit_weights([], length=3) == [1.0, 1.0, 1.0]


def test_frozen_evaluation_recouples_without_persistent_memory_writes():
    set_seed(11)
    agent = Organism(obs_dim=18, action_dim=3)
    env = TargetThreatEnv(seed=11)
    obs = env.reset()
    agent.eval()
    agent.reset_state()
    consequence_size = agent.consequence_memory.size()
    motor_size = agent.motor_memory.size()
    attractors_before = agent.attractor_field.attractors.detach().clone()
    weights_before = agent.attractor_field.weights.detach().clone()

    action, _ = agent.select_action(obs, evaluate=True)
    h_at_action = agent.h.detach().clone()
    next_obs, reward, _, info = env.step(action)
    agent.learn_consequence(
        reward,
        float(info["status"] in ("death", "starvation")),
        next_obs[2] - obs[2],
        next_obs[3] - obs[3],
        next_obs=next_obs,
        store=False,
    )

    assert not torch.equal(agent.h, h_at_action)
    assert not agent.h.requires_grad
    assert agent.rtcm.history[-1]["h_next"] is not None
    assert agent.consequence_memory.size() == consequence_size
    assert agent.motor_memory.size() == motor_size
    assert torch.equal(agent.attractor_field.attractors, attractors_before)
    assert torch.equal(agent.attractor_field.weights, weights_before)
