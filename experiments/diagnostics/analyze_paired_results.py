"""Paired uncertainty analysis for multi-seed OLF diagnostics."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np


def _exact_sign_flip_pvalues(differences):
    values = np.asarray(differences, dtype=np.float64)
    nonzero = values[values != 0.0]
    if nonzero.size == 0:
        return 1.0, 1.0
    if nonzero.size > 20:
        raise ValueError("exact sign-flip test is limited to 20 nonzero pairs")

    observed = float(values.mean())
    null_means = np.asarray(
        [
            (nonzero * np.asarray(signs)).sum() / values.size
            for signs in itertools.product((-1.0, 1.0), repeat=nonzero.size)
        ]
    )
    tolerance = 1e-12
    greater = float(np.mean(null_means >= observed - tolerance))
    two_sided = float(
        np.mean(np.abs(null_means) >= abs(observed) - tolerance)
    )
    return greater, two_sided


def _holm_adjust(pvalues):
    adjusted = [0.0] * len(pvalues)
    running_max = 0.0
    for rank, index in enumerate(np.argsort(pvalues)):
        running_max = max(
            running_max, (len(pvalues) - rank) * float(pvalues[index])
        )
        adjusted[index] = min(1.0, running_max)
    return adjusted


def analyze_paired_results(
    result,
    *,
    baseline="babble_only",
    treatment="full",
    bootstrap_samples=100_000,
    seed=20260713,
):
    rng = np.random.default_rng(seed)
    comparisons = {}
    two_sided_pvalues = []

    for task, conditions in result["summary"].items():
        baseline_rows = {
            row["seed"]: row["success_rate"]
            for row in conditions[baseline]["per_seed"]
        }
        treatment_rows = {
            row["seed"]: row["success_rate"]
            for row in conditions[treatment]["per_seed"]
        }
        if baseline_rows.keys() != treatment_rows.keys():
            raise ValueError(f"unpaired seeds for task {task!r}")
        seeds = sorted(baseline_rows)
        differences = np.asarray(
            [treatment_rows[item] - baseline_rows[item] for item in seeds],
            dtype=np.float64,
        )
        bootstrap_means = np.asarray(
            [
                rng.choice(differences, size=len(differences), replace=True).mean()
                for _ in range(bootstrap_samples)
            ]
        )
        greater_p, two_sided_p = _exact_sign_flip_pvalues(differences)
        comparisons[task] = {
            "seeds": seeds,
            "paired_differences": differences.tolist(),
            "mean_difference": float(differences.mean()),
            "bootstrap_95_ci": [
                float(np.quantile(bootstrap_means, 0.025)),
                float(np.quantile(bootstrap_means, 0.975)),
            ],
            "exact_greater_p": greater_p,
            "exact_two_sided_p": two_sided_p,
            "positive_pairs": int(np.sum(differences > 0.0)),
            "negative_pairs": int(np.sum(differences < 0.0)),
            "tied_pairs": int(np.sum(differences == 0.0)),
        }
        two_sided_pvalues.append(two_sided_p)

    adjusted = _holm_adjust(two_sided_pvalues)
    for comparison, adjusted_p in zip(
        comparisons.values(), adjusted, strict=True
    ):
        comparison["holm_two_sided_p"] = adjusted_p

    return {
        "baseline": baseline,
        "treatment": treatment,
        "bootstrap_samples": int(bootstrap_samples),
        "bootstrap_seed": int(seed),
        "multiplicity_family": list(comparisons),
        "comparisons": comparisons,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result")
    parser.add_argument("--baseline", default="babble_only")
    parser.add_argument("--treatment", default="full")
    parser.add_argument("--bootstrap-samples", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--out")
    args = parser.parse_args()

    input_path = Path(args.result)
    result = json.loads(input_path.read_text())
    analysis = analyze_paired_results(
        result,
        baseline=args.baseline,
        treatment=args.treatment,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    output_path = (
        Path(args.out)
        if args.out
        else input_path.with_name(f"{input_path.stem}_paired.json")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(analysis, indent=2))
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
