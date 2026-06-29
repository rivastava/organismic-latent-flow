"""Smoke test for the OLF v0.1 architecture.

Runs each of the 13 modular benchmarks for one episode with the full OLF
organism. Asserts that no crash occurs and that the constitutional signal
channels (mode, veto verdict, readiness, diagnostic decay) are all present.
"""

import os
import sys
import traceback

# Ensure repo root is on sys.path so `olf` and `benchmarks` import.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np

from olf.organism import Organism
from olf.baselines import AblatedOrganism, MLPBaselineAgent

import benchmarks.target_threat as target_threat
import benchmarks.delayed_lure as delayed_lure
import benchmarks.abstraction_unseen as abstraction_unseen
import benchmarks.negative_control as negative_control
import benchmarks.context_flip as context_flip
import benchmarks.self_state_meaning as self_state_meaning
import benchmarks.triadic_binding as triadic_binding
import benchmarks.role_transformation as role_transformation
import benchmarks.affordance_gap as affordance_gap
import benchmarks.situated_gap as situated_gap
import benchmarks.code_body as code_body
import benchmarks.code_body_real as code_body_real
import benchmarks.randomized_consequence as randomized_consequence

BENCHMARKS = [
    ("target_threat", target_threat.TargetThreatEnv),
    ("delayed_lure", delayed_lure.DelayedLureEnv),
    ("abstraction_unseen", abstraction_unseen.AbstractionUnseenEnv),
    ("negative_control", negative_control.NegativeControlEnv),
    ("context_flip", context_flip.ContextFlipEnv),
    ("self_state_meaning", self_state_meaning.SelfStateMeaningEnv),
    ("triadic_binding", triadic_binding.TriadicBindingEnv),
    ("role_transformation", role_transformation.RoleTransformationEnv),
    ("affordance_gap", affordance_gap.AffordanceGapEnv),
    ("situated_gap", situated_gap.SituatedGapEnv),
    ("code_body", code_body.CodeBodyEnv),
    ("code_body_real", code_body_real.CodeBodyRealEnv),
    ("randomized_consequence", randomized_consequence.RandomizedConsequenceEnv),
]

REQUIRED_INFO_KEYS = {"mode", "verdict", "risk", "readiness", "diagnostic_decay"}


def smoke_test_full_olf():
    print("=" * 60)
    print("OLF v0.1 Smoke Test — Full Organism")
    print("=" * 60)
    failures = []
    for name, EnvClass in BENCHMARKS:
        try:
            env = EnvClass()
            agent = Organism()
            obs = env.reset()
            agent.reset_state()
            done = False
            steps = 0
            while not done and steps < env.max_steps + 1:
                action, info = agent.select_action(obs)
                missing = REQUIRED_INFO_KEYS - set(info.keys())
                if missing:
                    raise AssertionError(f"missing info keys: {missing}")
                obs, reward, done, _ = env.step(action)
                agent.learn_consequence(reward, 0.0, 0.0, 0.0)
                steps += 1
            print(f"  [OK] {name:30s} steps={steps}")
        except Exception:
            tb = traceback.format_exc()
            print(f"  [FAIL] {name}")
            failures.append((name, tb))
    return failures


def smoke_test_ablations():
    print()
    print("=" * 60)
    print("OLF Smoke Test — Required And Internal Ablations")
    print("=" * 60)
    required_ablations = [
        "no_memory", "no_spm", "last_observation_only", "no_rtcm",
        "no_self_state", "no_abstraction", "exact_episodic_memory_only",
        "no_veto_boundary", "soft_risk_only", "no_mode_arbitration",
        "no_invention", "ungated_invention", "no_recoupling_constraint",
        "no_closure_pressure", "no_diagnostic_decay",
        "no_future_latent",
        "no_motor_memory",  # v3 internal ablation
    ]
    failures = []
    for ab in required_ablations:
        try:
            agent = AblatedOrganism(ablation_type=ab)
            agent.reset_state()
            obs = np.random.randn(18).astype(np.float32)
            for _ in range(5):
                action, _ = agent.select_action(obs)
                agent.learn_consequence(0.1, 0.0, 0.01, 0.005)
                obs = np.random.randn(18).astype(np.float32)
            print(f"  [OK] {ab}")
        except Exception:
            tb = traceback.format_exc()
            print(f"  [FAIL] {ab}")
            failures.append((ab, tb))
    return failures


def smoke_test_mlp_baseline():
    print()
    print("=" * 60)
    print("OLF v0.1 Smoke Test — MLP Baseline")
    print("=" * 60)
    try:
        agent = MLPBaselineAgent()
        obs = np.random.randn(18).astype(np.float32)
        for _ in range(5):
            action, info = agent.select_action(obs)
            assert action.shape == (3,), "MLP bad action shape"
            obs = np.random.randn(18).astype(np.float32)
        print("  [OK] MLP baseline runs")
        return []
    except Exception:
        tb = traceback.format_exc()
        print("  [FAIL] MLP baseline")
        return [("MLP", tb)]


def main():
    failures = []
    failures.extend(smoke_test_full_olf())
    failures.extend(smoke_test_ablations())
    failures.extend(smoke_test_mlp_baseline())

    print()
    print("=" * 60)
    if failures:
        print(f"FAILED: {len(failures)} smoke test failures")
        for name, tb in failures:
            print(f"\n--- {name} ---")
            print(tb)
        sys.exit(1)
    else:
        print("ALL SMOKE TESTS PASSED")
        sys.exit(0)


if __name__ == "__main__":
    main()
