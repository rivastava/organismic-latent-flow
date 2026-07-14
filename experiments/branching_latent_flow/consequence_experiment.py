"""Consequence experiment: temporary ghost influence learned from endogenous body consequence."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from .consequence_deformation import (
    ConsequenceTraces,
    deform_from_consequence,
    fit_consequence_model,
    release_index,
)
from .geometry import log_map_sphere, project_to_tangent
from .protocol import ProtocolConfig, prefix_context
from .role_free_ghosts import materialize_trajectories
from .observations import observed_examples, stage1_trials
from .transfer_ghosts import fit_transfer_ghosts

CONDITIONS = (
    "situated",
    "situated_no_boundary",
    "fixed_context",
    "scrambled",
    "no_consequence",
)


def body_consequence(
    deformation: torch.Tensor,
    self_state: torch.Tensor,
    viability: torch.Tensor,
    coupling_axis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    alignment = (deformation * coupling_axis).sum(dim=-1)
    raw_delta = 0.35 * self_state * alignment
    after = (viability + raw_delta).clamp(0.0, 1.0)
    delta = after - viability
    before_pressure = ((0.20 - viability) / 0.20).clamp(0.0, 1.0)
    after_pressure = ((0.20 - after) / 0.20).clamp(0.0, 1.0)
    boundary_deformation = (after_pressure - before_pressure).clamp(0.0, 1.0)
    lethal = after <= 0.05
    return delta, boundary_deformation, lethal


def build_traces(trials, *, seed: int, coupling_axis: torch.Tensor) -> ConsequenceTraces:
    generator = torch.Generator().manual_seed(seed)
    before = torch.stack([trial.ambiguous_prefix[-1] for trial in trials])
    after = torch.stack([trial.trajectory[-1] for trial in trials])
    deformation = project_to_tangent(before, log_map_sphere(before, after))
    self_state = torch.where(
        torch.rand(len(trials), generator=generator) < 0.5,
        -torch.ones(len(trials)),
        torch.ones(len(trials)),
    )
    viability = 0.08 + 0.84 * torch.rand(len(trials), generator=generator)
    delta, boundary, _ = body_consequence(
        deformation, self_state, viability, coupling_axis
    )
    return ConsequenceTraces(
        before,
        deformation,
        after,
        self_state,
        viability,
        delta,
        boundary,
    )


def _scramble(traces: ConsequenceTraces, *, seed: int) -> ConsequenceTraces:
    order = torch.randperm(
        len(traces.before), generator=torch.Generator().manual_seed(seed)
    )
    return ConsequenceTraces(
        traces.before,
        traces.deformation,
        traces.after,
        traces.self_state,
        traces.viability,
        traces.viability_delta[order],
        traces.boundary_deformation[order],
    )


def _select(
    condition: str,
    models: dict,
    population,
    present: torch.Tensor,
    self_state: float,
    viability: float,
) -> int:
    if condition == "no_consequence":
        return int(population.credibility.argmax().item())
    model_name = "situated" if condition == "situated_no_boundary" else condition
    deformed = deform_from_consequence(
        models[model_name], population, present, self_state, viability
    )
    if condition == "situated_no_boundary":
        return int(deformed.influence.argmax().item())
    return release_index(deformed)


def run_seed(*, seed: int, n_modes: int, capacity: int = 8) -> dict:
    config = ProtocolConfig(
        n_modes=n_modes,
        train_trials=800,
        eval_trials=300,
        seed=9090 + 43 * seed,
    )
    train = stage1_trials(config, split="train")
    evaluate = stage1_trials(config, split="eval")
    contexts, probes, endpoints = observed_examples(train)
    field = fit_transfer_ghosts(
        contexts, probes, endpoints, capacity=capacity, seed=seed + 10_000
    )
    axis_generator = torch.Generator().manual_seed(config.seed + 77_777)
    coupling_axis = F.normalize(
        torch.randn(config.latent_dim, generator=axis_generator), dim=0
    )
    traces = build_traces(train, seed=seed + 20_000, coupling_axis=coupling_axis)
    models = {
        "situated": fit_consequence_model(traces, situated=True),
        "fixed_context": fit_consequence_model(traces, situated=False),
        "scrambled": fit_consequence_model(
            _scramble(traces, seed=seed + 30_000), situated=True
        ),
    }
    records: dict[str, dict[str, list[float]]] = {
        condition: {
            "optimal": [],
            "safe": [],
            "viability_regret": [],
            "selected_delta": [],
            "self_state_flip": [],
            "safe_when_risk_present": [],
        }
        for condition in CONDITIONS
    }
    eval_generator = torch.Generator().manual_seed(seed + 40_000)
    for trial in evaluate:
        present = trial.ambiguous_prefix[-1]
        belief = field.belief(prefix_context(trial.ambiguous_prefix))
        population = materialize_trajectories(present, belief)
        deformation = project_to_tangent(
            present.unsqueeze(0).expand(population.count, -1),
            log_map_sphere(
                present.unsqueeze(0).expand(population.count, -1),
                population.points[:, -1],
            ),
        )
        self_state = -1.0 if torch.rand((), generator=eval_generator) < 0.5 else 1.0
        viability = float(0.08 + 0.84 * torch.rand((), generator=eval_generator))
        delta, boundary, lethal = body_consequence(
            deformation,
            torch.full((population.count,), self_state),
            torch.full((population.count,), viability),
            coupling_axis,
        )
        viable_delta = delta.clone()
        viable_delta[lethal] = -torch.inf
        oracle = int(
            (viable_delta if bool((~lethal).any().item()) else delta).argmax().item()
        )
        for condition in CONDITIONS:
            selected = _select(
                condition,
                models,
                population,
                present,
                self_state,
                viability,
            )
            opposite = _select(
                condition,
                models,
                population,
                present,
                -self_state,
                viability,
            )
            records[condition]["optimal"].append(float(selected == oracle))
            records[condition]["safe"].append(float(not lethal[selected].item()))
            records[condition]["viability_regret"].append(
                float((delta[oracle] - delta[selected]).item())
            )
            records[condition]["selected_delta"].append(float(delta[selected].item()))
            records[condition]["self_state_flip"].append(float(selected != opposite))
            if bool(lethal.any().item()):
                records[condition]["safe_when_risk_present"].append(
                    float(not lethal[selected].item())
                )
    metrics = {
        condition: {
            name: sum(values) / len(values) if values else float("nan")
            for name, values in condition_records.items()
        }
        for condition, condition_records in records.items()
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
        for condition in CONDITIONS:
            result[str(n_modes)][condition] = {}
            for metric in selected[0]["metrics"][condition]:
                values = [row["metrics"][condition][metric] for row in selected]
                mean = sum(values) / len(values)
                variance = sum((value - mean) ** 2 for value in values) / len(values)
                result[str(n_modes)][condition][metric] = {
                    "mean": mean,
                    "std": math.sqrt(variance),
                    "n": len(values),
                }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("results/consequence_role_stage5.json"))
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--n-modes", type=int, nargs="+", default=[2, 3, 5])
    parser.add_argument("--capacity", type=int, default=8)
    args = parser.parse_args()
    rows = []
    for n_modes in args.n_modes:
        for seed in range(args.seeds):
            print(f"consequence deformation n_modes={n_modes} seed={seed}", flush=True)
            row = run_seed(seed=seed, n_modes=n_modes, capacity=args.capacity)
            rows.append(row)
            print(json.dumps(row["metrics"], sort_keys=True), flush=True)
    payload = {"config": vars(args), "rows": rows, "aggregate": aggregate(rows)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
