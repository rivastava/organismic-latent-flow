"""Set-control experiment: keep future hypotheses distinct through action choice."""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Protocol

import torch
import torch.nn.functional as F

from .geometry import (
    angular_distance,
    exponential_map,
    log_map_sphere,
    project_to_tangent,
)
from .protocol import ProtocolConfig, prefix_context
from .seeding import set_seed
from .set_control import (
    BranchBelief,
    BranchSetPredictor,
    centroid_correction,
    inverse_corrections,
    recouple_weights,
    select_correction,
)
from .observations import observed_examples, stage1_trials


class BeliefProvider(Protocol):
    def belief(self, context: torch.Tensor) -> BranchBelief: ...


def train_predictor(
    model: BranchSetPredictor,
    contexts: torch.Tensor,
    probes: torch.Tensor,
    endpoints: torch.Tensor,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> list[float]:
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    generator = torch.Generator().manual_seed(seed)
    history: list[float] = []
    for _ in range(epochs):
        order = torch.randperm(contexts.shape[0], generator=generator)
        total = 0.0
        batches = 0
        for start in range(0, contexts.shape[0], batch_size):
            index = order[start : start + batch_size]
            loss = model.nll(contexts[index], probes[index], endpoints[index])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += float(loss.item())
            batches += 1
        history.append(total / max(1, batches))
    return history


def _true_correction(trial) -> torch.Tensor:
    current = trial.ambiguous_prefix[-1]
    correction = log_map_sphere(current, trial.endpoint)
    correction = project_to_tangent(current, correction)
    return F.normalize(correction, p=2, dim=-1, eps=1e-8)


def _correct(action: torch.Tensor, target: torch.Tensor, tolerance: float) -> float:
    return float(angular_distance(action, target).item() < tolerance)


@torch.no_grad()
def evaluate_predictor(
    model: BeliefProvider,
    trials,
    *,
    seed: int,
    action_tolerance: float = 0.30,
    probe_noise_arc: float = 0.0,
) -> dict[str, float]:
    generator = torch.Generator().manual_seed(seed)
    records: dict[str, list[float]] = {
        "coverage": [],
        "effective_branches": [],
        "ambiguous_set_success": [],
        "ambiguous_set_expected_success": [],
        "ambiguous_centroid_success": [],
        "ambiguous_set_plausibility": [],
        "ambiguous_centroid_plausibility": [],
        "revealed_set_success": [],
        "revealed_centroid_success": [],
        "posterior_true_mass": [],
        "posterior_entropy": [],
    }
    for trial in trials:
        current = trial.ambiguous_prefix[-1]
        belief = model.belief(prefix_context(trial.ambiguous_prefix))
        candidates = inverse_corrections(current, belief.endpoints)
        target = _true_correction(trial)
        candidate_target_errors = angular_distance(
            candidates, target.expand_as(candidates)
        ).squeeze(-1)
        correct_components = candidate_target_errors < action_tolerance

        mode_corrections = inverse_corrections(current, trial.all_mode_endpoints)
        covered = []
        for mode_action in mode_corrections:
            errors = angular_distance(candidates, mode_action.expand_as(candidates))
            covered.append(float(errors.min().item() < action_tolerance))
        records["coverage"].append(sum(covered) / len(covered))

        prior_entropy = -(
            belief.weights.clamp_min(1e-12) * belief.weights.clamp_min(1e-12).log()
        ).sum()
        records["effective_branches"].append(float(torch.exp(prior_entropy).item()))

        sampled, _ = select_correction(
            current,
            belief,
            generator=generator,
            stochastic=True,
        )
        records["ambiguous_set_success"].append(
            _correct(sampled, target, action_tolerance)
        )
        records["ambiguous_set_expected_success"].append(
            float(belief.weights[correct_components].sum().item())
        )
        records["ambiguous_centroid_success"].append(
            _correct(centroid_correction(current, belief), target, action_tolerance)
        )
        plausible_components = (
            angular_distance(
                candidates.unsqueeze(1),
                mode_corrections.unsqueeze(0),
            ).squeeze(-1).min(dim=1).values
            < action_tolerance
        )
        records["ambiguous_set_plausibility"].append(
            float(belief.weights[plausible_components].sum().item())
        )
        centroid = centroid_correction(current, belief)
        records["ambiguous_centroid_plausibility"].append(
            float(
                angular_distance(
                    mode_corrections,
                    centroid.expand_as(mode_corrections),
                ).min().item()
                < action_tolerance
            )
        )

        observed_probe = trial.revealed_prefix[-1]
        if probe_noise_arc > 0.0:
            noise = torch.randn(
                observed_probe.shape, generator=generator, dtype=observed_probe.dtype
            )
            noise = F.normalize(
                project_to_tangent(observed_probe, noise), p=2, dim=-1, eps=1e-8
            )
            observed_probe = exponential_map(
                observed_probe, noise * probe_noise_arc
            )
        posterior = recouple_weights(belief, observed_probe)
        selected, _ = select_correction(current, belief, weights=posterior)
        records["revealed_set_success"].append(
            _correct(selected, target, action_tolerance)
        )
        records["revealed_centroid_success"].append(
            _correct(
                centroid_correction(current, belief, posterior),
                target,
                action_tolerance,
            )
        )
        records["posterior_true_mass"].append(
            float(posterior[correct_components].sum().item())
        )
        records["posterior_entropy"].append(
            float(
                -(
                    posterior.clamp_min(1e-12)
                    * posterior.clamp_min(1e-12).log()
                ).sum().item()
            )
        )
    return {name: sum(values) / len(values) for name, values in records.items()}


def run_seed(
    *,
    seed: int,
    n_modes: int,
    capacity: int,
    train_trials: int,
    eval_trials: int,
    epochs: int,
) -> dict:
    config = ProtocolConfig(
        n_modes=n_modes,
        train_trials=train_trials,
        eval_trials=eval_trials,
        seed=2026 + 17 * seed,
    )
    train_trials_data = stage1_trials(config, split="train")
    eval_trials_data = stage1_trials(config, split="eval")
    contexts, probes, endpoints = observed_examples(train_trials_data)
    set_seed(seed)
    model = BranchSetPredictor(config.latent_dim, capacity=capacity)
    loss = train_predictor(
        model,
        contexts,
        probes,
        endpoints,
        epochs=epochs,
        batch_size=64,
        learning_rate=2e-3,
        seed=seed + 10_000,
    )
    clean = evaluate_predictor(model, eval_trials_data, seed=seed + 20_000)
    noisy = evaluate_predictor(
        model,
        eval_trials_data,
        seed=seed + 30_000,
        probe_noise_arc=0.12,
    )
    metrics = {
        **{f"clean_{name}": value for name, value in clean.items()},
        **{f"noisy_{name}": value for name, value in noisy.items()},
    }
    return {
        "seed": seed,
        "n_modes": n_modes,
        "capacity": capacity,
        "parameter_count": model.parameter_count(),
        "train_nll_first": loss[0],
        "train_nll_last": loss[-1],
        "metrics": metrics,
    }


def aggregate(rows: list[dict]) -> dict:
    result: dict[str, dict] = {}
    for n_modes in sorted({row["n_modes"] for row in rows}):
        selected = [row for row in rows if row["n_modes"] == n_modes]
        metrics = selected[0]["metrics"]
        result[str(n_modes)] = {}
        for metric in metrics:
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
    parser.add_argument("--out", default="results/set_control_stage1")
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--n-modes", type=int, nargs="+", default=[2, 3, 5])
    parser.add_argument("--capacity", type=int, default=8)
    parser.add_argument("--train-trials", type=int, default=600)
    parser.add_argument("--eval-trials", type=int, default=240)
    parser.add_argument("--epochs", type=int, default=60)
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rows = []
    for n_modes in args.n_modes:
        for seed in range(args.seeds):
            print(f"stage1 n_modes={n_modes} seed={seed}", flush=True)
            row = run_seed(
                seed=seed,
                n_modes=n_modes,
                capacity=args.capacity,
                train_trials=args.train_trials,
                eval_trials=args.eval_trials,
                epochs=args.epochs,
            )
            rows.append(row)
            print(json.dumps(row["metrics"], sort_keys=True), flush=True)
    payload = {
        "config": vars(args),
        "rows": rows,
        "aggregate": aggregate(rows),
    }
    with open(os.path.join(args.out, "aggregate.json"), "w") as handle:
        json.dump(payload, handle, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
