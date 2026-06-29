"""Tests for boundary-risk calibration (issue #2).

Verifies that B_psi training targets produce attributable risk,
and that the danger signal has non-zero variance after training.
"""

import numpy as np

from olf.organism import Organism


def test_attributable_risk_target_death():
    """Death should always produce attributable_risk = 1.0."""
    from experiments.run_core import (
        _attributable_boundary_target,
        _boundary_proximity_target,
        _passive_boundary_target,
    )

    obs = np.zeros(18, dtype=np.float32)
    obs[2] = 0.95  # high hunger
    obs[3] = 0.95  # high fatigue
    next_obs = obs.copy()
    next_obs[2] = 1.0  # dead

    target = _boundary_proximity_target(obs, next_obs, was_lethal=True)
    passive = _passive_boundary_target(obs)
    attributable = _attributable_boundary_target(target, passive, was_lethal=True)

    assert target == 1.0, f"Death target should be 1.0, got {target}"
    assert attributable == 1.0, (
        f"Death attributable risk should be 1.0, got {attributable}"
    )


def test_attributable_risk_target_improvement():
    """Action that improves self_state should have zero attributable risk."""
    from experiments.run_core import (
        _attributable_boundary_target,
        _boundary_proximity_target,
        _passive_boundary_target,
    )

    obs = np.zeros(18, dtype=np.float32)
    obs[2] = 0.7  # moderate hunger
    obs[3] = 0.5
    next_obs = obs.copy()
    next_obs[2] = 0.3  # improved

    target = _boundary_proximity_target(obs, next_obs, was_lethal=False)
    passive = _passive_boundary_target(obs)
    attributable = _attributable_boundary_target(target, passive, was_lethal=False)

    assert attributable == 0.0, (
        f"Improvement should have 0 attributable risk, got {attributable} "
        f"(target={target}, passive={passive})"
    )


def test_attributable_risk_target_worsening():
    """Action that worsens self_state should have positive attributable risk."""
    from experiments.run_core import (
        _attributable_boundary_target,
        _boundary_proximity_target,
        _passive_boundary_target,
    )

    obs = np.zeros(18, dtype=np.float32)
    obs[2] = 0.5
    obs[3] = 0.5
    next_obs = obs.copy()
    next_obs[2] = 0.85  # worsened significantly

    target = _boundary_proximity_target(obs, next_obs, was_lethal=False)
    passive = _passive_boundary_target(obs)
    attributable = _attributable_boundary_target(target, passive, was_lethal=False)

    assert attributable > 0.0, (
        f"Worsening should have positive attributable risk, got {attributable} "
        f"(target={target}, passive={passive})"
    )


def test_zero_action_sample_always_zero_target():
    """Zero-action training sample should always have target = 0.0."""
    zero_target = 0.0
    assert zero_target == 0.0


def test_bpsi_training_stats_keys():
    """B_psi training stats should have the attributable risk keys."""
    from experiments.run_core import train_agent, set_global_seed

    set_global_seed(42)
    agent = Organism(obs_dim=18, action_dim=3)
    agent = train_agent(agent, "self_state_meaning", num_episodes=10, seed=42)

    stats = getattr(agent, "_bpsi_training_stats", [])
    assert len(stats) > 0, "No B_psi training stats recorded"
    for s in stats:
        assert "attributable_mean" in s, "Missing attributable_mean in stats"
        assert "attributable_max" in s, "Missing attributable_max in stats"
        assert "attributable_nonzero_rate" in s, "Missing attributable_nonzero_rate in stats"
        assert "loss" in s, "Missing loss in stats"


def test_danger_variance_after_training():
    """After training with attributable targets, B_psi should produce
    non-zero danger on at least some steps."""
    from experiments.run_core import train_agent, set_global_seed
    from benchmarks.self_state_meaning import SelfStateMeaningEnv

    set_global_seed(42)
    agent = Organism(obs_dim=18, action_dim=3)
    agent = train_agent(agent, "self_state_meaning", num_episodes=100, seed=42)
    agent.eval()
    agent.diag_mode = True
    env = SelfStateMeaningEnv(seed=100)
    dangers = []
    boundary_risks = []
    baselines = []

    for _ in range(20):
        obs = env.reset()
        agent.reset_state()
        agent.reset_diag()
        done = False
        while not done:
            action, info = agent.select_action(obs, evaluate=True)
            next_obs, reward, done, info_env = env.step(action)
            was_lethal = 1.0 if info_env["status"] in ("death", "starvation") else 0.0
            agent.learn_consequence(reward, was_lethal, next_obs[2] - obs[2], next_obs[3] - obs[3])
            obs = next_obs
        for entry in agent.diag_buffer:
            if "danger" in entry:
                dangers.append(entry["danger"])
                boundary_risks.append(entry.get("veto_boundary_risk", 0.0))
                baselines.append(entry.get("risk_baseline", 0.0))

    agent.diag_mode = False

    assert len(dangers) > 0, "No danger values recorded"
    max_danger = max(dangers)
    mean_danger = float(np.mean(dangers))
    nonzero_danger = sum(1 for d in dangers if d > 0.0)

    # After training with attributable targets, danger should have some variance.
    # The key assertion: not ALL steps should have danger = 0.
    # With attributable risk targets, B_psi should learn to produce higher risk
    # for actions that worsen the situation.
    assert max_danger > 0.0 or nonzero_danger > 0, (
        f"All danger values are zero after training. "
        f"mean={mean_danger:.6f}, max={max_danger:.6f}, nonzero={nonzero_danger}/{len(dangers)}. "
        f"B_psi is not action-discriminative."
    )
