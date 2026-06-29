"""Focused boundary/FLC diagnostic runner.

The runner trains OLF on a small task set, evaluates with diagnostics enabled,
and writes one consistent JSON object:

{
  "config": {...},
  "tasks": {
    "<task>": {
      "summary": {...},
      "per_seed": [...]
    }
  }
}
"""

import argparse
import json
import os

import numpy as np

from experiments.run_core import ENV_MAP, set_global_seed, train_agent
from olf.organism import Organism


DEFAULT_TASKS = [
    "self_state_meaning",
    "delayed_lure",
    "triadic_binding",
    "target_threat",
]


def _mean(values):
    return float(np.mean(values)) if values else 0.0


def _std(values):
    return float(np.std(values)) if values else 0.0


def train_agent_for_diagnostics(task_name, seed, num_episodes):
    set_global_seed(seed)
    agent = Organism(obs_dim=18, action_dim=3)
    agent.diag_mode = False
    return train_agent(agent, task_name, num_episodes=num_episodes, seed=seed)


def evaluate_with_diagnostics(agent, task_name, seed, num_episodes):
    set_global_seed(seed)
    agent.eval()
    agent.diag_mode = True
    env = ENV_MAP[task_name](seed=seed)

    episodes = []
    for _ in range(num_episodes):
        obs = env.reset()
        agent.reset_state()
        agent.reset_diag()

        done = False
        info = {"status": "running"}
        while not done:
            action, _info_dict = agent.select_action(obs, evaluate=True)
            next_obs, reward, done, info = env.step(action)
            was_lethal = 1.0 if info["status"] in ("death", "starvation") else 0.0
            hunger_delta = next_obs[2] - obs[2]
            fatigue_delta = next_obs[3] - obs[3]
            agent.learn_consequence(reward, was_lethal, hunger_delta, fatigue_delta)
            obs = next_obs

        episodes.append({
            "status": info["status"],
            "steps": list(agent.diag_buffer),
        })

    return episodes


def run_single_config(task_name, seed, num_episodes=150, num_eval=20):
    print(f"  [{task_name}] seed={seed} training...")
    agent = train_agent_for_diagnostics(task_name, seed, num_episodes)

    print(f"  [{task_name}] seed={seed} evaluating...")
    eval_diag = evaluate_with_diagnostics(agent, task_name, seed + 100, num_eval)
    bpsi_stats = getattr(agent, "_bpsi_training_stats", [])
    recent_bpsi = bpsi_stats[-50:] if bpsi_stats else []

    statuses = [episode["status"] for episode in eval_diag]
    all_entries = [entry for episode in eval_diag for entry in episode["steps"]]
    step_entries = [entry for entry in all_entries if "step" in entry]
    veto_entries = [entry for entry in all_entries if "veto_verdict" in entry]

    dangers = [entry.get("danger", 0.0) for entry in veto_entries]
    boundary_risks = [entry.get("veto_boundary_risk", 0.0) for entry in veto_entries]
    baselines = [entry.get("risk_baseline", 0.0) for entry in veto_entries]
    verdicts = [entry.get("veto_verdict", "release") for entry in veto_entries]

    total_verdicts = len(verdicts)
    verdict_dist = {
        name: (verdicts.count(name) / total_verdicts if total_verdicts else 0.0)
        for name in ("release", "hold", "recouple", "rollback")
    }

    return {
        "task": task_name,
        "seed": seed,
        "success_rate": statuses.count("success") / len(statuses) if statuses else 0.0,
        "status_counts": {status: statuses.count(status) for status in sorted(set(statuses))},
        "mean_danger": _mean(dangers),
        "max_danger": float(np.max(dangers)) if dangers else 0.0,
        "mean_veto_boundary_risk": _mean(boundary_risks),
        "mean_risk_baseline": _mean(baselines),
        "verdict_dist": verdict_dist,
        "total_steps": len(step_entries),
        "total_diag_entries": len(all_entries),
        "bpsi_target_mean": _mean([item["target_mean"] for item in recent_bpsi]),
        "bpsi_target_max": float(np.max([item["target_max"] for item in recent_bpsi])) if recent_bpsi else 0.0,
        "bpsi_target_nonzero_rate": _mean([item["target_nonzero_rate"] for item in recent_bpsi]),
        "bpsi_loss_last": float(recent_bpsi[-1]["loss"]) if recent_bpsi else 0.0,
    }


def summarize(per_seed):
    return {
        "success_mean": _mean([run["success_rate"] for run in per_seed]),
        "success_std": _std([run["success_rate"] for run in per_seed]),
        "danger_mean": _mean([run["mean_danger"] for run in per_seed]),
        "danger_std": _std([run["mean_danger"] for run in per_seed]),
        "max_danger_mean": _mean([run["max_danger"] for run in per_seed]),
        "release_mean": _mean([run["verdict_dist"]["release"] for run in per_seed]),
        "hold_mean": _mean([run["verdict_dist"]["hold"] for run in per_seed]),
        "recouple_mean": _mean([run["verdict_dist"]["recouple"] for run in per_seed]),
        "rollback_mean": _mean([run["verdict_dist"]["rollback"] for run in per_seed]),
        "veto_boundary_risk_mean": _mean([run["mean_veto_boundary_risk"] for run in per_seed]),
        "risk_baseline_mean": _mean([run["mean_risk_baseline"] for run in per_seed]),
        "bpsi_target_nonzero_rate_mean": _mean([run["bpsi_target_nonzero_rate"] for run in per_seed]),
        "bpsi_loss_last_mean": _mean([run["bpsi_loss_last"] for run in per_seed]),
        "total_steps": int(sum(run["total_steps"] for run in per_seed)),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", default=",".join(DEFAULT_TASKS))
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--episodes", type=int, default=150)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--out", default="results/boundary_signal/results.json")
    return parser.parse_args()


def main():
    args = parse_args()
    tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]
    seeds = [int(seed.strip()) for seed in args.seeds.split(",") if seed.strip()]

    result = {
        "config": {
            "tasks": tasks,
            "seeds": seeds,
            "train_episodes": args.episodes,
            "eval_episodes": args.eval_episodes,
        },
        "tasks": {},
    }

    for task in tasks:
        print(f"\n--- {task} ---")
        per_seed = []
        for seed in seeds:
            run = run_single_config(
                task,
                seed,
                num_episodes=args.episodes,
                num_eval=args.eval_episodes,
            )
            per_seed.append(run)
            print(
                f"  seed={seed}: success={run['success_rate']:.1%}, "
                f"danger={run['mean_danger']:.4f}, "
                f"release={run['verdict_dist']['release']:.1%}, "
                f"rollback={run['verdict_dist']['rollback']:.1%}"
            )

        summary = summarize(per_seed)
        result["tasks"][task] = {
            "summary": summary,
            "per_seed": per_seed,
        }
        print(
            f"  summary: success={summary['success_mean']:.1%} "
            f"+/- {summary['success_std']:.1%}, "
            f"danger={summary['danger_mean']:.4f}, "
            f"rollback={summary['rollback_mean']:.1%}"
        )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
