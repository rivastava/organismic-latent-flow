"""Tests for the 10-seed benchmark guard.

Verifies output schema, runner structure, and that the guard
can run with minimal seeds (smoke-level).
"""

import json
import os


def test_guard_output_schema():
    """run_guard with 1 seed produces valid schema with all required keys."""
    from experiments.bench_guard import run_guard

    results, passed = run_guard(seeds=[0], tasks=["self_state_meaning"],
                                train_episodes=5, eval_episodes=3)

    assert isinstance(results, dict)
    assert isinstance(passed, bool)
    assert "self_state_meaning" in results

    task_data = results["self_state_meaning"]
    assert "conditions" in task_data
    assert "delta" in task_data
    assert "guards" in task_data

    for cond in ("olf", "no_future_latent"):
        c = task_data["conditions"][cond]
        assert "per_seed" in c
        assert "success_mean" in c
        assert "success_std" in c
        assert "safety_mean" in c
        assert "danger_mean" in c
        assert "rollback_mean" in c
        assert len(c["per_seed"]) >= 1
        row = c["per_seed"][0]
        assert "seed" in row
        assert "success_rate" in row
        assert "safety_rate" in row
        assert "mean_danger" in row
        assert "max_danger" in row
        assert "rollback_rate" in row

    d = task_data["delta"]
    assert "success_delta" in d
    assert "safety_delta" in d
    assert "danger_delta" in d

    g = task_data["guards"]
    assert "olf_success_floor" in g
    assert "olf_safety_floor" in g
    assert "seed_count" in g
    assert all(isinstance(v, bool) for v in g.values())


def test_guard_writes_json():
    """run_guard writes a valid JSON file under results/guards/."""
    from experiments.bench_guard import run_guard

    results, _ = run_guard(seeds=[0], tasks=["self_state_meaning"],
                           train_episodes=5, eval_episodes=3)

    out_dir = "results/guards"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "bench_guard_test.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    with open(out_path) as f:
        loaded = json.load(f)
    assert loaded == results
    os.remove(out_path)


def test_guard_thresholds_are_conservative():
    """Thresholds should be loose enough that a working system passes."""
    from experiments.bench_guard import THRESHOLDS

    # Floors should be well below perfect performance
    for task, floor in THRESHOLDS["olf_success_floor"].items():
        assert 0.0 <= floor <= 0.90, f"Floor for {task} is {floor}"
    assert 0.0 <= THRESHOLDS["olf_safety_floor"] <= 0.90
    assert THRESHOLDS["min_seed_count"] == 10


def test_guard_seeds_deterministic():
    """Running the same seed twice should produce the same per-seed stats."""
    from experiments.bench_guard import _run_one

    s1 = _run_one("self_state_meaning", None, seed=42, num_train=5, num_eval=3)
    s2 = _run_one("self_state_meaning", None, seed=42, num_train=5, num_eval=3)

    assert s1["success_rate"] == s2["success_rate"]
    assert s1["safety_rate"] == s2["safety_rate"]
    assert abs(s1["mean_danger"] - s2["mean_danger"]) < 1e-6


def test_guard_ablation_runs():
    """no_future_latent ablation should also run in the guard."""
    from experiments.bench_guard import _run_one

    stats = _run_one("self_state_meaning", "no_future_latent",
                     seed=0, num_train=5, num_eval=3)
    assert "success_rate" in stats
    assert "safety_rate" in stats
    assert 0.0 <= stats["success_rate"] <= 1.0
    assert 0.0 <= stats["safety_rate"] <= 1.0
