"""Stage 2 experiment for adaptive spherical transfer ghosts."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from .protocol import ProtocolConfig
from .set_experiment import evaluate_predictor
from .observations import observed_examples, stage1_trials
from .transfer_ghosts import fit_transfer_ghosts


def run_seed(
    *, seed: int, n_modes: int, capacity: int, train_trials: int, eval_trials: int
) -> dict:
    config = ProtocolConfig(
        n_modes=n_modes,
        train_trials=train_trials,
        eval_trials=eval_trials,
        seed=4040 + 31 * seed,
    )
    train = stage1_trials(config, split="train")
    evaluate = stage1_trials(config, split="eval")
    contexts, probes, endpoints = observed_examples(train)
    adaptive = fit_transfer_ghosts(
        contexts, probes, endpoints, capacity=capacity, seed=seed + 10_000
    )
    single = fit_transfer_ghosts(
        contexts, probes, endpoints, capacity=1, seed=seed + 10_000
    )
    shuffle = torch.randperm(
        len(contexts), generator=torch.Generator().manual_seed(seed + 40_000)
    )
    shuffled = fit_transfer_ghosts(
        contexts,
        probes[shuffle],
        endpoints[shuffle],
        capacity=capacity,
        seed=seed + 10_000,
    )
    adaptive_metrics = evaluate_predictor(adaptive, evaluate, seed=seed + 20_000)
    noisy_metrics = evaluate_predictor(
        adaptive,
        evaluate,
        seed=seed + 30_000,
        probe_noise_arc=0.12,
    )
    return {
        "seed": seed,
        "n_modes": n_modes,
        "capacity": capacity,
        "active_ghosts": len(adaptive.matrices),
        "count_error": abs(len(adaptive.matrices) - n_modes),
        "events": [event.__dict__ for event in adaptive.events],
        "adaptive": adaptive_metrics,
        "adaptive_noisy": noisy_metrics,
        "single": evaluate_predictor(single, evaluate, seed=seed + 20_000),
        "shuffled": evaluate_predictor(shuffled, evaluate, seed=seed + 20_000),
    }


def aggregate(rows: list[dict]) -> dict:
    result = {}
    for n_modes in sorted({row["n_modes"] for row in rows}):
        selected = [row for row in rows if row["n_modes"] == n_modes]
        summary = {}
        scalar_paths = {
            "active_ghosts": lambda row: row["active_ghosts"],
            "count_error": lambda row: row["count_error"],
            "coverage": lambda row: row["adaptive"]["coverage"],
            "set_plausibility": lambda row: row["adaptive"]["ambiguous_set_plausibility"],
            "centroid_plausibility": lambda row: row["adaptive"]["ambiguous_centroid_plausibility"],
            "single_coverage": lambda row: row["single"]["coverage"],
            "shuffled_coverage": lambda row: row["shuffled"]["coverage"],
            "revealed_success": lambda row: row["adaptive"]["revealed_set_success"],
            "noisy_revealed_success": lambda row: row["adaptive_noisy"]["revealed_set_success"],
        }
        for name, getter in scalar_paths.items():
            values = [float(getter(row)) for row in selected]
            mean = sum(values) / len(values)
            variance = sum((value - mean) ** 2 for value in values) / len(values)
            summary[name] = {"mean": mean, "std": math.sqrt(variance), "n": len(values)}
        result[str(n_modes)] = summary
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("results/transfer_ghost_stage2.json"))
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--n-modes", type=int, nargs="+", default=[2, 3, 5])
    parser.add_argument("--capacity", type=int, default=8)
    parser.add_argument("--train-trials", type=int, default=600)
    parser.add_argument("--eval-trials", type=int, default=240)
    args = parser.parse_args()
    rows = []
    for n_modes in args.n_modes:
        for seed in range(args.seeds):
            print(f"transfer ghosts n_modes={n_modes} seed={seed}", flush=True)
            row = run_seed(
                seed=seed,
                n_modes=n_modes,
                capacity=args.capacity,
                train_trials=args.train_trials,
                eval_trials=args.eval_trials,
            )
            rows.append(row)
            print(
                f"active={row['active_ghosts']} coverage={row['adaptive']['coverage']:.3f} "
                f"single={row['single']['coverage']:.3f}",
                flush=True,
            )
    payload = {"config": vars(args), "rows": rows, "aggregate": aggregate(rows)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
