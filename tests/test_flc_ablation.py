"""Tests for the FLC ablation suite.

Verifies that the no_future_latent ablation runs, that the runner
produces the expected structure, and that FLC actually changes behavior.
"""

import numpy as np
import torch

from olf.organism import Organism
from olf.baselines import AblatedOrganism
from olf.seeding import set_seed


def test_no_future_latent_ablation_runs():
    """AblatedOrganism with no_future_latent must produce valid actions."""
    set_seed(42)
    agent = AblatedOrganism(obs_dim=18, action_dim=3, ablation_type="no_future_latent")
    agent.reset_state()
    obs = np.random.randn(18).astype(np.float32)

    for _ in range(5):
        action, info = agent.select_action(obs)
        agent.learn_consequence(0.1, 0.0, 0.01, 0.005)
        obs = np.random.randn(18).astype(np.float32)

    assert action.shape == (3,)
    assert np.all(action >= -1.0) and np.all(action <= 1.0)
    assert "verdict" in info
    assert "danger" in info


def test_full_olf_runs():
    """Full OLF organism must also run without error."""
    set_seed(42)
    agent = Organism(obs_dim=18, action_dim=3)
    agent.reset_state()
    obs = np.random.randn(18).astype(np.float32)

    for _ in range(5):
        action, info = agent.select_action(obs)
        agent.learn_consequence(0.1, 0.0, 0.01, 0.005)
        obs = np.random.randn(18).astype(np.float32)

    assert action.shape == (3,)
    assert "verdict" in info


def test_flc_changes_behavior():
    """FLC and no_future_latent should produce different actions on the same input."""
    set_seed(42)
    obs = np.random.randn(18).astype(np.float32)

    full_agent = Organism(obs_dim=18, action_dim=3)
    full_agent.reset_state()
    full_agent.eval()
    with torch.no_grad():
        action_full, _ = full_agent.select_action(obs, evaluate=True)

    set_seed(42)
    abl_agent = AblatedOrganism(
        obs_dim=18, action_dim=3, ablation_type="no_future_latent"
    )
    abl_agent.reset_state()
    abl_agent.eval()
    with torch.no_grad():
        action_abl, _ = abl_agent.select_action(obs, evaluate=True)

    # They should differ because FLC modifies the action proposal
    diff = np.linalg.norm(action_full - action_abl)
    assert diff > 1e-6, f"FLC had no effect: diff={diff}"


def test_flc_ablation_runner_structure():
    """run_flc_ablation returns the expected dict structure."""
    from experiments.run_core import run_flc_ablation

    results = run_flc_ablation(seeds=[42], num_train=10, num_eval=3)

    assert isinstance(results, dict)
    for _task, data in results.items():
        assert "conditions" in data
        assert "delta" in data
        assert "olf" in data["conditions"]
        assert "no_future_latent" in data["conditions"]
        for cond in ("olf", "no_future_latent"):
            c = data["conditions"][cond]
            assert "success_mean" in c
            assert "success_std" in c
            assert "danger_mean" in c
            assert "danger_std" in c
            assert "rollback_mean" in c
            assert "rollback_std" in c
            assert "per_seed" in c
            assert len(c["per_seed"]) >= 1
            seed_row = c["per_seed"][0]
            assert "mean_danger" in seed_row
            assert "max_danger" in seed_row
            assert "rollback_rate" in seed_row
        d = data["delta"]
        assert "success_delta" in d
        assert "danger_delta" in d
        assert "rollback_delta" in d
