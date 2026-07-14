"""Role-fluid trajectory experiment: one ghost substrate, several temporary relations."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from .geometry import angular_distance, slerp
from .protocol import ProtocolConfig, prefix_context
from .role_free_ghosts import (
    correction_view,
    future_view,
    materialize_trajectories,
    predecessor_view,
    recouple_observation,
)
from .set_control import centroid_correction, inverse_corrections
from .observations import observed_examples, stage1_trials
from .transfer_ghosts import fit_transfer_ghosts


def _near_any(
    candidates: torch.Tensor, references: torch.Tensor, tolerance: float
) -> float:
    errors = angular_distance(
        candidates.unsqueeze(1), references.unsqueeze(0)
    ).squeeze(-1)
    return float((errors.min(dim=1).values < tolerance).float().mean().item())


def run_seed(
    *, seed: int, n_modes: int, capacity: int = 8, steps: int = 5
) -> dict:
    config = ProtocolConfig(
        n_modes=n_modes,
        train_trials=600,
        eval_trials=240,
        seed=7070 + 41 * seed,
    )
    train = stage1_trials(config, split="train")
    evaluate = stage1_trials(config, split="eval")
    contexts, probes, endpoints = observed_examples(train)
    field = fit_transfer_ghosts(
        contexts, probes, endpoints, capacity=capacity, seed=seed + 10_000
    )
    single = fit_transfer_ghosts(
        contexts, probes, endpoints, capacity=1, seed=seed + 10_000
    )
    records: dict[str, list[float]] = {
        "future_coverage": [],
        "action_plausibility": [],
        "centroid_plausibility": [],
        "single_future_coverage": [],
        "probe_recoupled_future": [],
        "trajectory_memory_error": [],
        "backward_predecessor_error": [],
    }
    times = torch.linspace(0.0, 1.0, steps + 1).view(1, steps + 1, 1)
    for trial in evaluate:
        current = trial.ambiguous_prefix[-1]
        context = prefix_context(trial.ambiguous_prefix)
        belief = field.belief(context)
        population = materialize_trajectories(current, belief, steps=steps)
        futures = future_view(population)
        records["future_coverage"].append(
            _near_any(trial.all_mode_endpoints, futures, 0.30)
        )

        corrections = correction_view(population, current)
        mode_corrections = inverse_corrections(current, trial.all_mode_endpoints)
        records["action_plausibility"].append(
            _near_any(corrections, mode_corrections, 0.30)
        )
        centroid = centroid_correction(current, belief)
        records["centroid_plausibility"].append(
            _near_any(centroid.unsqueeze(0), mode_corrections, 0.30)
        )
        single_future = future_view(
            materialize_trajectories(current, single.belief(context), steps=steps)
        )
        records["single_future_coverage"].append(
            _near_any(trial.all_mode_endpoints, single_future, 0.30)
        )

        recoupled = recouple_observation(
            population, trial.revealed_prefix[-1], phase=1
        )
        selected = int(recoupled.credibility.argmax().item())
        selected_future = future_view(recoupled)[selected]
        records["probe_recoupled_future"].append(
            float(
                angular_distance(selected_future, trial.endpoint).item() < 0.30
            )
        )

        actual = slerp(
            current.view(1, 1, -1).expand(1, steps + 1, -1),
            trial.endpoint.view(1, 1, -1).expand(1, steps + 1, -1),
            times,
        )[0]
        path_error = angular_distance(
            recoupled.points[selected], actual
        ).squeeze(-1).mean()
        records["trajectory_memory_error"].append(float(path_error.item()))

        predecessor, _ = predecessor_view(
            recoupled, trial.endpoint, phase=steps
        )
        predecessor_error = angular_distance(predecessor, actual[-2])
        records["backward_predecessor_error"].append(
            float(predecessor_error.item())
        )
    metrics = {
        name: sum(values) / len(values) for name, values in records.items()
    }
    return {
        "seed": seed,
        "n_modes": n_modes,
        "active_ghosts": len(field.matrices),
        "metrics": metrics,
    }


def aggregate(rows: list[dict]) -> dict:
    result: dict[str, dict] = {}
    for n_modes in sorted({row["n_modes"] for row in rows}):
        selected = [row for row in rows if row["n_modes"] == n_modes]
        result[str(n_modes)] = {}
        for metric in selected[0]["metrics"]:
            values = [row["metrics"][metric] for row in selected]
            mean = sum(values) / len(values)
            variance = sum((value - mean) ** 2 for value in values) / len(values)
            result[str(n_modes)][metric] = {
                "mean": mean,
                "std": math.sqrt(variance),
                "n": len(values),
            }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("results/role_fluid_stage4.json"))
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--n-modes", type=int, nargs="+", default=[2, 3, 5])
    parser.add_argument("--capacity", type=int, default=8)
    args = parser.parse_args()
    rows = []
    for n_modes in args.n_modes:
        for seed in range(args.seeds):
            print(f"role-fluid n_modes={n_modes} seed={seed}", flush=True)
            row = run_seed(seed=seed, n_modes=n_modes, capacity=args.capacity)
            rows.append(row)
            print(json.dumps(row["metrics"], sort_keys=True), flush=True)
    payload = {"config": vars(args), "rows": rows, "aggregate": aggregate(rows)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
