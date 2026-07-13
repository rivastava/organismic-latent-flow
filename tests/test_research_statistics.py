import pytest

from experiments.diagnostics.analyze_paired_results import (
    _exact_sign_flip_pvalues,
    _holm_adjust,
    analyze_paired_results,
)


def _result(baseline, treatment):
    def rows(values):
        return [
            {"seed": seed, "success_rate": value}
            for seed, value in enumerate(values)
        ]

    return {
        "summary": {
            "task": {
                "babble_only": {"per_seed": rows(baseline)},
                "full": {"per_seed": rows(treatment)},
            }
        }
    }


def test_exact_sign_flip_test_handles_ties_and_known_extreme():
    greater, two_sided = _exact_sign_flip_pvalues([1.0, 1.0, 1.0, 1.0])
    assert greater == pytest.approx(1.0 / 16.0)
    assert two_sided == pytest.approx(2.0 / 16.0)
    assert _exact_sign_flip_pvalues([0.0, 0.0]) == (1.0, 1.0)


def test_holm_adjustment_is_monotone_in_sorted_order():
    adjusted = _holm_adjust([0.04, 0.01, 0.03])
    assert adjusted == pytest.approx([0.06, 0.03, 0.06])


def test_paired_analysis_reports_effect_and_deterministic_interval():
    result = _result([0.0, 0.2, 0.4, 0.6], [0.2, 0.4, 0.6, 0.8])
    first = analyze_paired_results(
        result, bootstrap_samples=1_000, seed=7
    )
    second = analyze_paired_results(
        result, bootstrap_samples=1_000, seed=7
    )

    comparison = first["comparisons"]["task"]
    assert comparison["mean_difference"] == pytest.approx(0.2)
    assert comparison["positive_pairs"] == 4
    assert comparison["negative_pairs"] == 0
    assert comparison["bootstrap_95_ci"] == pytest.approx([0.2, 0.2])
    assert first == second


def test_paired_analysis_rejects_unmatched_seeds():
    result = _result([0.0, 0.2], [0.2])
    with pytest.raises(ValueError, match="unpaired seeds"):
        analyze_paired_results(result, bootstrap_samples=10)
