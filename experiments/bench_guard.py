"""10-seed benchmark guard for core task behavior.

Runs OLF and no_future_latent across 10 deterministic seeds on the
core FLC tasks plus a few representative benchmarks. Emits a JSON
summary under results/guards/ for regression detection.

Thresholds are conservative: they detect gross regressions in the full
OLF condition, not enforce claims about ablation gaps.

Usage:
    python experiments/bench_guard.py
"""

import json
import os
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from olf.organism import Organism
from olf.baselines import AblatedOrganism
from experiments.run_core import (
    FLC_TASKS,
    train_agent,
    _evaluate_with_diagnostics,
    set_global_seed,
)

# 10 deterministic seeds (0-9).
GUARD_SEEDS = list(range(10))

# Core tasks: FLC tasks + two representative broader benchmarks.
GUARD_TASKS = FLC_TASKS + ["abstraction_unseen", "affordance_gap"]

# Training episodes (reduced for guard: just checking gross behavior).
TRAIN_EPISODES = 150

# Eval episodes per seed.
EVAL_EPISODES = 15

# --- Conservative thresholds ---
# These are floors for *mean* across seeds. They are
# intentionally loose to catch regressions, not enforce claims.

THRESHOLDS = {
    # Per-task: what full OLF must achieve at minimum (mean across seeds).
    "olf_success_floor": {
        "self_state_meaning": 0.70,
        "delayed_lure": 0.50,
        "triadic_binding": 0.50,
        "target_threat": 0.50,
        "abstraction_unseen": 0.50,
        "affordance_gap": 0.30,
    },
    # OLF safety: must not be dangerously low.
    "olf_safety_floor": 0.60,
    # Every condition must evaluate exactly this many seeds in the full guard.
    "min_seed_count": 10,
}


def _run_one(task, ablation_type, seed, num_train, num_eval):
    """Train one agent and evaluate. Returns per-seed stats dict."""
    set_global_seed(seed)
    if ablation_type is None:
        agent = Organism(obs_dim=18, action_dim=3)
    else:
        agent = AblatedOrganism(
            obs_dim=18, action_dim=3, ablation_type=ablation_type
        )
    agent = train_agent(agent, task, num_episodes=num_train, seed=seed)
    eval_episodes = _evaluate_with_diagnostics(agent, task, seed + 100, num_eval)

    successes = sum(1 for ep in eval_episodes if ep["status"] == "success")
    deaths = sum(
        1 for ep in eval_episodes if ep["status"] in ("death", "starvation")
    )
    mean_dangers = [ep["mean_danger"] for ep in eval_episodes]
    max_dangers = [ep["max_danger"] for ep in eval_episodes]
    rollbacks = sum(1 for ep in eval_episodes if ep["rollback_seen"])

    return {
        "seed": seed,
        "success_rate": successes / max(1, len(eval_episodes)),
        "safety_rate": 1.0 - deaths / max(1, len(eval_episodes)),
        "mean_danger": float(np.mean(mean_dangers)) if mean_dangers else 0.0,
        "max_danger": float(np.max(max_dangers)) if max_dangers else 0.0,
        "rollback_rate": rollbacks / max(1, len(eval_episodes)),
        "total_episodes": len(eval_episodes),
    }


def run_guard(seeds=None, tasks=None, train_episodes=None, eval_episodes=None):
    """Run the full 10-seed guard.

    Returns (results_dict, passed_bool).
    """
    if seeds is None:
        seeds = GUARD_SEEDS
    if tasks is None:
        tasks = GUARD_TASKS
    if train_episodes is None:
        train_episodes = TRAIN_EPISODES
    if eval_episodes is None:
        eval_episodes = EVAL_EPISODES

    results = {}
    all_pass = True

    for task in tasks:
        print(f"\n--- guard: {task} ---")
        per_condition = {}

        for cond_label, ablation_type in [("olf", None), ("no_future_latent", "no_future_latent")]:
            seed_stats = []
            for seed in seeds:
                stats = _run_one(task, ablation_type, seed, train_episodes, eval_episodes)
                seed_stats.append(stats)

            success_vals = [s["success_rate"] for s in seed_stats]
            safety_vals = [s["safety_rate"] for s in seed_stats]
            danger_vals = [s["mean_danger"] for s in seed_stats]
            rollback_vals = [s["rollback_rate"] for s in seed_stats]

            per_condition[cond_label] = {
                "per_seed": seed_stats,
                "success_mean": float(np.mean(success_vals)),
                "success_std": float(np.std(success_vals)),
                "safety_mean": float(np.mean(safety_vals)),
                "safety_std": float(np.std(safety_vals)),
                "danger_mean": float(np.mean(danger_vals)),
                "danger_std": float(np.std(danger_vals)),
                "rollback_mean": float(np.mean(rollback_vals)),
                "rollback_std": float(np.std(rollback_vals)),
            }

        olf = per_condition["olf"]
        nfl = per_condition["no_future_latent"]
        delta = {
            "success_delta": olf["success_mean"] - nfl["success_mean"],
            "safety_delta": olf["safety_mean"] - nfl["safety_mean"],
            "danger_delta": olf["danger_mean"] - nfl["danger_mean"],
            "rollback_delta": olf["rollback_mean"] - nfl["rollback_mean"],
        }

        results[task] = {
            "conditions": per_condition,
            "delta": delta,
        }

        # Evaluate guards
        task_guards = {}

        floor = THRESHOLDS["olf_success_floor"].get(task, 0.30)
        olf_ok = olf["success_mean"] >= floor
        task_guards["olf_success_floor"] = olf_ok
        if not olf_ok:
            all_pass = False
            print(f"  [FAIL] OLF success {olf['success_mean']:.1%} < floor {floor:.1%}")

        safety_ok = olf["safety_mean"] >= THRESHOLDS["olf_safety_floor"]
        task_guards["olf_safety_floor"] = safety_ok
        if not safety_ok:
            all_pass = False
            print(f"  [FAIL] OLF safety {olf['safety_mean']:.1%} < floor {THRESHOLDS['olf_safety_floor']:.1%}")

        expected_seed_count = min(THRESHOLDS["min_seed_count"], len(seeds))
        seed_count_ok = (
            len(olf["per_seed"]) >= expected_seed_count
            and len(nfl["per_seed"]) >= expected_seed_count
        )
        task_guards["seed_count"] = seed_count_ok
        if not seed_count_ok:
            all_pass = False
            print(
                "  [FAIL] incomplete seed coverage "
                f"OLF={len(olf['per_seed'])}, NFL={len(nfl['per_seed'])}, "
                f"expected>={expected_seed_count}"
            )

        results[task]["guards"] = task_guards

        status = "PASS" if all(task_guards.values()) else "FAIL"
        print(
            f"  [{status}] OLF={olf['success_mean']:.1%}±{olf['success_std']:.1%} "
            f"NFL={nfl['success_mean']:.1%}±{nfl['success_std']:.1%} "
            f"safety={olf['safety_mean']:.1%}"
        )

    return results, all_pass


def main():
    print("=" * 70)
    print("  OLF 10-Seed Benchmark Guard")
    print("=" * 70)

    results, passed = run_guard()

    # Write JSON
    out_dir = "results/guards"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "bench_guard.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 70}")
    print(f"  Guard result: {'PASSED' if passed else 'FAILED'}")
    print(f"  Output: {out_path}")
    print("=" * 70)

    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
