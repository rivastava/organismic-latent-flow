"""Inspect target_threat FLC telemetry without duplicating training logic.

This diagnostic compares full OLF against the no_future_latent ablation on
target_threat. It uses the canonical train_agent path, then runs evaluation
episodes with diagnostics enabled and writes aggregate FLC/action/boundary
telemetry to results/diagnostics/flc_instability.json.

Usage:
    python -m experiments.isolate_flc_instability
    python experiments/isolate_flc_instability.py
"""

import json
import os
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from benchmarks.target_threat import TargetThreatEnv
from experiments.run_core import train_agent
from olf.baselines import AblatedOrganism
from olf.organism import Organism
from olf.seeding import set_seed


DEFAULT_SEEDS = list(range(10))


def _mean(values):
    return float(np.mean(values)) if values else 0.0


def _std(values):
    return float(np.std(values)) if values else 0.0


def _run_eval_episode(agent, seed):
    env = TargetThreatEnv(seed=seed)
    obs = env.reset()
    agent.reset_state()
    if hasattr(agent, "reset_diag"):
        agent.reset_diag()
    agent.diag_mode = True

    steps = []
    done = False
    info = {"status": "running"}
    while not done:
        action, action_info = agent.select_action(obs, evaluate=True)
        next_obs, reward, done, info = env.step(action)
        was_lethal = 1.0 if info["status"] in ("death", "starvation") else 0.0
        agent.learn_consequence(
            reward,
            was_lethal,
            next_obs[2] - obs[2],
            next_obs[3] - obs[3],
            next_obs=next_obs,
            store=False,
        )
        steps.append({
            "danger": float(action_info.get("danger", 0.0)),
            "verdict": action_info.get("verdict", "release"),
            "flc_correction_norm": float(action_info.get("flc_correction_norm", 0.0)),
            "flc_action_delta_norm": float(action_info.get("flc_action_delta_norm", 0.0)),
            "flc_gain": float(action_info.get("flc_gain", 0.0)),
            "future_alignment": float(action_info.get("future_alignment", 0.0)),
            "action_norm": float(np.linalg.norm(action)),
        })
        obs = next_obs

    agent.diag_mode = False
    return {
        "status": info["status"],
        "steps": steps,
    }


def _summarize_episodes(episodes):
    statuses = [episode["status"] for episode in episodes]
    step_rows = [step for episode in episodes for step in episode["steps"]]
    rollbacks = [step["verdict"] == "rollback" for step in step_rows]
    return {
        "success_rate": statuses.count("success") / max(1, len(statuses)),
        "death_rate": sum(s in ("death", "starvation") for s in statuses) / max(1, len(statuses)),
        "total_steps": len(step_rows),
        "danger_mean": _mean([s["danger"] for s in step_rows]),
        "danger_max": float(np.max([s["danger"] for s in step_rows])) if step_rows else 0.0,
        "rollback_rate": _mean(rollbacks),
        "flc_correction_mean": _mean([s["flc_correction_norm"] for s in step_rows]),
        "flc_correction_max": (
            float(np.max([s["flc_correction_norm"] for s in step_rows]))
            if step_rows else 0.0
        ),
        "flc_action_delta_mean": _mean([s["flc_action_delta_norm"] for s in step_rows]),
        "flc_action_delta_max": (
            float(np.max([s["flc_action_delta_norm"] for s in step_rows]))
            if step_rows else 0.0
        ),
        "flc_gain_mean": _mean([s["flc_gain"] for s in step_rows]),
        "future_alignment_mean": _mean([s["future_alignment"] for s in step_rows]),
        "action_norm_mean": _mean([s["action_norm"] for s in step_rows]),
    }


def _condition_agent(condition):
    if condition == "olf":
        return Organism(obs_dim=18, action_dim=3)
    if condition == "no_future_latent":
        return AblatedOrganism(obs_dim=18, action_dim=3, ablation_type="no_future_latent")
    raise ValueError(f"Unknown condition: {condition}")


def run_target_threat_diagnostic(seeds=None, train_episodes=150, eval_episodes=15):
    if seeds is None:
        seeds = DEFAULT_SEEDS

    results = {}
    for condition in ("olf", "no_future_latent"):
        per_seed = []
        for seed in seeds:
            set_seed(seed)
            agent = _condition_agent(condition)
            agent = train_agent(
                agent,
                "target_threat",
                num_episodes=train_episodes,
                seed=seed,
                agent_type=condition,
            )
            episodes = [
                _run_eval_episode(agent, seed + 100 + i)
                for i in range(eval_episodes)
            ]
            summary = _summarize_episodes(episodes)
            summary["seed"] = seed
            per_seed.append(summary)

        results[condition] = {
            "per_seed": per_seed,
            "success_mean": _mean([s["success_rate"] for s in per_seed]),
            "success_std": _std([s["success_rate"] for s in per_seed]),
            "death_mean": _mean([s["death_rate"] for s in per_seed]),
            "danger_mean": _mean([s["danger_mean"] for s in per_seed]),
            "rollback_mean": _mean([s["rollback_rate"] for s in per_seed]),
            "flc_correction_mean": _mean([s["flc_correction_mean"] for s in per_seed]),
            "flc_correction_max": (
                float(np.max([s["flc_correction_max"] for s in per_seed]))
                if per_seed else 0.0
            ),
            "flc_action_delta_mean": _mean([s["flc_action_delta_mean"] for s in per_seed]),
            "flc_gain_mean": _mean([s["flc_gain_mean"] for s in per_seed]),
        }

    olf = results["olf"]
    nfl = results["no_future_latent"]
    results["delta"] = {
        "success_delta": olf["success_mean"] - nfl["success_mean"],
        "danger_delta": olf["danger_mean"] - nfl["danger_mean"],
        "rollback_delta": olf["rollback_mean"] - nfl["rollback_mean"],
    }
    return results


def main():
    results = run_target_threat_diagnostic()
    out_dir = os.path.join("results", "diagnostics")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "flc_instability.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print("target_threat FLC diagnostic")
    for condition in ("olf", "no_future_latent"):
        row = results[condition]
        print(
            f"{condition}: success={row['success_mean']:.1%} "
            f"danger={row['danger_mean']:.4f} "
            f"rollback={row['rollback_mean']:.1%} "
            f"flc_correction_max={row['flc_correction_max']:.4f}"
        )
    print(f"delta success={results['delta']['success_delta']:+.1%}")
    print(f"Output: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
