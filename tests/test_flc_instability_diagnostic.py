"""Tests for the target_threat FLC instability diagnostic."""

from experiments.isolate_flc_instability import run_target_threat_diagnostic


def test_target_threat_diagnostic_schema():
    results = run_target_threat_diagnostic(
        seeds=[0],
        train_episodes=1,
        eval_episodes=1,
    )

    assert "olf" in results
    assert "no_future_latent" in results
    assert "delta" in results

    for condition in ("olf", "no_future_latent"):
        row = results[condition]
        assert "per_seed" in row
        assert len(row["per_seed"]) == 1
        assert "success_mean" in row
        assert "danger_mean" in row
        assert "rollback_mean" in row
        assert "flc_correction_mean" in row
        assert "flc_gain_mean" in row

        seed_row = row["per_seed"][0]
        assert seed_row["seed"] == 0
        assert "success_rate" in seed_row
        assert "total_steps" in seed_row
        assert "flc_action_delta_max" in seed_row

    assert "success_delta" in results["delta"]
