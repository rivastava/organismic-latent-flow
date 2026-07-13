"""Causal ablations for event-based prospective memory and FLC control."""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch

from experiments.run_core import (
    _evaluate_with_diagnostics,
    set_global_seed,
    train_agent,
)
from olf.organism import Organism


CONDITIONS = {
    "babble_only": {"use_prospective_event_grounding": False},
    "full": {"use_prospective_event_grounding": True},
    "no_memory": {
        "use_prospective_event_grounding": True,
        "use_prospective_event_memory": False,
    },
    "sigma_key": {
        "use_prospective_event_grounding": True,
        "use_situated_prospective_keys": False,
    },
    "event_only": {
        "use_prospective_event_grounding": True,
        "prospective_max_horizon": 1,
    },
    "no_action_retrieval": {
        "use_prospective_event_grounding": True,
        "use_prospective_action_retrieval": False,
    },
}


def _run_one(
    task,
    condition,
    seed,
    train_episodes,
    eval_episodes,
    babble_probability,
):
    torch.set_num_threads(1)
    set_global_seed(seed)
    agent = Organism(
        obs_dim=18,
        action_dim=3,
        use_hierarchical_intent=True,
        hierarchical_intent_std=0.5,
        hierarchical_intent_blend=0.8,
        hierarchical_babble_probability=babble_probability,
        **CONDITIONS[condition],
    )
    train_agent(
        agent,
        task,
        num_episodes=train_episodes,
        seed=seed,
        agent_type=f"prospective_{condition}",
        training_signal="terminal_homeostasis",
        credit_mode="uniform",
    )
    episodes = _evaluate_with_diagnostics(
        agent, task, seed + 10_000, eval_episodes
    )
    success = [episode["status"] == "success" for episode in episodes]
    death = [
        episode["status"] in ("death", "starvation")
        for episode in episodes
    ]
    return {
        "task": task,
        "condition": condition,
        "seed": seed,
        "success_rate": float(np.mean(success)),
        "safety_rate": float(1.0 - np.mean(death)),
        "event_count": int(agent.prospective_events_seen.item()),
        "memory_records": len(agent.prospective_event_memory),
    }


def run_probe(
    seeds,
    *,
    tasks,
    conditions,
    train_episodes=150,
    eval_episodes=15,
    max_workers=4,
    babble_probability=1.0,
):
    jobs = [
        (task, condition, seed)
        for task in tasks
        for condition in conditions
        for seed in seeds
    ]
    rows = []

    def record(row):
        rows.append(row)
        print(
            f"{row['task']:30s} {row['condition']:20s} "
            f"seed={row['seed']:2d} success={row['success_rate']:.1%} "
            f"events={row['event_count']:3d} memory={row['memory_records']:4d}",
            flush=True,
        )

    if max_workers == 1:
        for task, condition, seed in jobs:
            record(
                _run_one(
                    task,
                    condition,
                    seed,
                    train_episodes,
                    eval_episodes,
                    babble_probability,
                )
            )
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(
                    _run_one,
                    task,
                    condition,
                    seed,
                    train_episodes,
                    eval_episodes,
                    babble_probability,
                ): (task, condition, seed)
                for task, condition, seed in jobs
            }
            for future in as_completed(futures):
                record(future.result())

    summary = {}
    for task in tasks:
        summary[task] = {}
        for condition in conditions:
            selected = sorted(
                (
                    row
                    for row in rows
                    if row["task"] == task
                    and row["condition"] == condition
                ),
                key=lambda row: row["seed"],
            )
            summary[task][condition] = {
                "success_mean": float(
                    np.mean([row["success_rate"] for row in selected])
                ),
                "success_std": float(
                    np.std([row["success_rate"] for row in selected])
                ),
                "safety_mean": float(
                    np.mean([row["safety_rate"] for row in selected])
                ),
                "event_mean": float(
                    np.mean([row["event_count"] for row in selected])
                ),
                "per_seed": selected,
            }
    return {
        "configuration": {
            "seeds": list(seeds),
            "tasks": list(tasks),
            "conditions": list(conditions),
            "train_episodes": train_episodes,
            "eval_episodes": eval_episodes,
            "training_signal": "terminal_homeostasis",
            "babble_probability": float(babble_probability),
        },
        "summary": summary,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--tasks", nargs="+", default=["triadic_binding"])
    parser.add_argument(
        "--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS)
    )
    parser.add_argument("--train-episodes", type=int, default=150)
    parser.add_argument("--eval-episodes", type=int, default=15)
    parser.add_argument("--babble-probability", type=float, default=1.0)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument(
        "--out",
        default="results/diagnostics/prospective_ablation_3seed.json",
    )
    args = parser.parse_args()
    result = run_probe(
        range(args.seeds),
        tasks=args.tasks,
        conditions=args.conditions,
        train_episodes=args.train_episodes,
        eval_episodes=args.eval_episodes,
        max_workers=args.workers,
        babble_probability=args.babble_probability,
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
