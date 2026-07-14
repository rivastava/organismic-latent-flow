"""Lifecycle experiment: ghost dormancy and return in a changing world."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from .persistent_ghosts import PersistentTransferGhostField
from .set_experiment import evaluate_predictor
from .observations import observed_examples
from .changing_world import phase_evaluation_trials, scheduled_trials
from .transfer_ghosts import fit_transfer_ghosts

PHASES = (
    ({0: 270, 1: 270, 2: 60}, (0, 1, 2)),
    ({0: 270, 1: 270, 3: 60}, (0, 1, 3)),
    ({0: 100, 1: 100, 2: 8, 3: 40}, (0, 1, 2, 3)),
    ({0: 240, 1: 240, 2: 60, 3: 60}, (0, 1, 2, 3)),
)


def run_seed(seed: int, *, capacity: int = 8) -> dict:
    world_seed = 5150 + 37 * seed
    field = PersistentTransferGhostField(latent_dim=8, capacity=capacity)
    phase_rows = []
    initial_ids = set()
    disappeared_ids = set()
    for phase, (counts, modes) in enumerate(PHASES):
        train = scheduled_trials(
            seed=world_seed, counts=counts, split_offset=phase
        )
        contexts, probes, endpoints = observed_examples(train)
        field.update(contexts, probes, endpoints, seed=world_seed + phase)
        if phase == 0:
            initial_ids = set(field.active_identities())
        if phase == 1:
            disappeared_ids = initial_ids & set(field.dormant_identities())
        evaluate = phase_evaluation_trials(seed=world_seed, modes=modes)
        persistent_metrics = evaluate_predictor(
            field, evaluate, seed=world_seed + 100 + phase
        )
        amnesic = fit_transfer_ghosts(
            contexts, probes, endpoints, capacity=capacity, seed=world_seed + phase
        )
        amnesic_metrics = evaluate_predictor(
            amnesic, evaluate, seed=world_seed + 100 + phase
        )
        phase_rows.append(
            {
                "phase": phase,
                "modes": list(modes),
                "active_ids": list(field.active_identities()),
                "dormant_ids": list(field.dormant_identities()),
                "persistent_coverage": persistent_metrics["coverage"],
                "persistent_plausibility": persistent_metrics[
                    "ambiguous_set_plausibility"
                ],
                "amnesic_coverage": amnesic_metrics["coverage"],
                "amnesic_active": len(amnesic.matrices),
            }
        )
    returned_same_identity = bool(
        disappeared_ids and disappeared_ids <= set(field.active_identities())
    )
    return {
        "seed": seed,
        "phases": phase_rows,
        "returned_same_identity": returned_same_identity,
        "population": len(field.slots),
        "events": [event.__dict__ for event in field.events],
    }


def aggregate(rows: list[dict]) -> dict:
    result: dict[str, object] = {}
    for phase in range(len(PHASES)):
        selected = [row["phases"][phase] for row in rows]
        phase_summary: dict[str, dict[str, float | int]] = {}
        for metric in (
            "persistent_coverage",
            "persistent_plausibility",
            "amnesic_coverage",
            "amnesic_active",
        ):
            values = [float(row[metric]) for row in selected]
            mean = sum(values) / len(values)
            variance = sum((value - mean) ** 2 for value in values) / len(values)
            phase_summary[metric] = {
                "mean": mean,
                "std": math.sqrt(variance),
                "n": len(values),
            }
        result[str(phase)] = phase_summary
    result["same_identity_return_rate"] = sum(
        row["returned_same_identity"] for row in rows
    ) / len(rows)
    result["max_population"] = max(row["population"] for row in rows)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("results/persistent_ghost_stage3.json"))
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--capacity", type=int, default=8)
    args = parser.parse_args()
    rows = []
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for seed in range(args.seed_start, args.seed_start + args.seeds):
        print(f"persistent ghosts seed={seed}", flush=True)
        row = run_seed(seed, capacity=args.capacity)
        rows.append(row)
        final = row["phases"][-1]
        print(
            f"return={row['returned_same_identity']} population={row['population']} "
            f"coverage={final['persistent_coverage']:.3f} "
            f"amnesic={final['amnesic_coverage']:.3f}",
            flush=True,
        )
        checkpoint = {"config": vars(args), "rows": rows}
        args.out.write_text(json.dumps(checkpoint, indent=2, default=str) + "\n")
    payload = {"config": vars(args), "rows": rows, "aggregate": aggregate(rows)}
    args.out.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
