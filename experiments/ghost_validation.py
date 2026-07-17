"""Paired validation runner for ghost influence in the complete OLF loop."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from experiments.run_core import ENV_MAP, train_agent
from olf.ghosts.config import GhostConfig
from olf.organism import Organism
from olf.seeding import set_seed

PROTOCOL = "ghost_stage10"
PRIMARY_TASKS = (
    "delayed_lure",
    "triadic_binding",
    "target_threat",
    "self_state_meaning",
)
CONDITIONS = (
    "off",
    "observe",
    "influence",
    "no_ghosts",
    "single_ghost",
    "centroid_before_inverse",
    "no_persistence",
    "no_recoupling",
    "random_routing",
)
ABLATION_CONDITIONS = CONDITIONS[3:]
OBS_DIM = 18
LATENT_DIM = 32
HIDDEN_DIM = 64
ACTION_DIM = 3
BOOTSTRAP_SEED = 12345
BOOTSTRAP_SAMPLES = 10_000


@dataclass(frozen=True)
class EpisodeRecord:
    task: str
    condition: str
    seed: int
    episode: int
    status: str
    success: bool
    death: bool
    steps: int
    reward_reporting_only: float
    action_norm_mean: float
    action_divergence_from_off: float
    episode_length_difference_from_off: int
    viability_mean: float
    danger_mean: float
    boundary_verdict_counts: dict[str, int]
    rollback_rate: float
    ghost_influence_rate: float
    population_mean: float
    evidence_support_mean: float
    grounding_mean: float
    reachability_residual_mean: float | None
    transfer_support: int
    births: int
    merges: int
    evictions: int
    recouplings: int


def _condition_spec(condition: str) -> tuple[str, str | None]:
    if condition in ("off", "observe", "influence"):
        return condition, None
    if condition in ABLATION_CONDITIONS:
        return "influence", condition
    raise ValueError(f"unknown condition: {condition}")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    set_seed(seed)


def _commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        check=False,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def _protocol_config(args) -> dict:
    return {
        "protocol": PROTOCOL,
        "conditions": list(args.conditions),
        "tasks": list(args.tasks),
        "seeds": list(args.seeds),
        "train_episodes": args.train_episodes,
        "eval_episodes": args.eval_episodes,
        "learning_rate": args.learning_rate,
        "latent_dim": LATENT_DIM,
        "hidden_dim": HIDDEN_DIM,
        "action_dim": ACTION_DIM,
        "training_signal": args.training_signal,
        "credit_mode": args.credit_mode,
        "optimizer_profile": args.optimizer_profile,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
    }


def _protocol_hash(config: dict) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _agent(condition: str, seed: int = 0) -> Organism:
    mode, ablation = _condition_spec(condition)
    config = GhostConfig(
        ghost_mode=mode,
        ablation=ablation,
        latent_dim=LATENT_DIM,
        action_dim=ACTION_DIM,
        seed=seed,
    )
    return Organism(
        obs_dim=OBS_DIM,
        latent_dim=LATENT_DIM,
        hidden_dim=HIDDEN_DIM,
        action_dim=ACTION_DIM,
        ghost_mode=mode,
        ghost_config=config,
    )


def _train(seed: int, task: str, args) -> Organism:
    _set_seed(seed)
    agent = _agent("off", seed)
    return train_agent(
        agent,
        task,
        num_episodes=args.train_episodes,
        lr=args.learning_rate,
        seed=seed,
        training_signal=args.training_signal,
        credit_mode=args.credit_mode,
        optimizer_profile=args.optimizer_profile,
    )


def _telemetry_delta(before: dict, after: dict, key: str) -> int:
    return int(after[key]) - int(before[key])


def _evaluate(
    trained: Organism,
    task: str,
    condition: str,
    seed: int,
    episodes: int,
    off_actions: dict[tuple[int, int], np.ndarray] | None,
) -> tuple[list[EpisodeRecord], dict[tuple[int, int], np.ndarray]]:
    _set_seed(seed)
    agent = _agent(condition, seed)
    agent.load_state_dict(trained.state_dict())
    agent.eval()
    agent.episode_count = max(agent.episode_count, agent.warmup_episodes)
    agent.veto.warmup = False
    env = ENV_MAP[task](seed=seed)
    records = []
    traces = {}

    for episode in range(episodes):
        obs = env.reset()
        agent.reset_state()
        agent.episode_count = max(agent.episode_count, agent.warmup_episodes)
        agent.veto.warmup = False
        before = agent.ghost.telemetry() if agent.ghost is not None else None
        done = False
        step = 0
        rewards = 0.0
        actions = []
        dangers = []
        viabilities = []
        verdict_counts = {verdict: 0 for verdict in ("release", "hold", "recouple", "rollback")}
        rollbacks = 0
        influenced = 0
        populations = []
        residuals = []

        while not done:
            # Boundary steering differentiates risk with respect to the candidate
            # action even during evaluation. No optimizer step or backward pass
            # over model parameters occurs here.
            with torch.enable_grad():
                action, action_info = agent.select_action(obs, evaluate=True)
            action = np.asarray(action, dtype=np.float32)
            if not np.isfinite(action).all() or not torch.isfinite(agent.h).all():
                raise FloatingPointError(
                    f"non-finite state task={task} condition={condition} "
                    f"seed={seed} episode={episode} step={step}"
                )
            next_obs, reward, done, env_info = env.step(action)
            status = str(env_info["status"])
            lethal = float(status in ("death", "starvation"))
            agent.learn_consequence(
                float(reward),
                lethal,
                float(next_obs[2] - obs[2]),
                float(next_obs[3] - obs[3]),
                next_obs=next_obs,
                store=False,
            )

            ghost_info = action_info.get("ghost") or {}
            influenced += int(bool(ghost_info.get("ghost_influenced", False)))
            populations.append(int(ghost_info.get("population", 0)))
            for candidate in ghost_info.get("candidates", []):
                residual = candidate.get("reachability_residual")
                if residual is not None and math.isfinite(float(residual)):
                    residuals.append(float(residual))
            actions.append(action)
            dangers.append(float(action_info.get("danger", 0.0)))
            viabilities.append(float(action_info.get("viability", 1.0)))
            verdict = str(action_info.get("verdict", "release"))
            if verdict not in verdict_counts:
                raise ValueError(f"unknown boundary verdict: {verdict}")
            verdict_counts[verdict] += 1
            rollbacks += int(verdict == "rollback")
            rewards += float(reward)
            obs = next_obs
            step += 1

        action_array = np.stack(actions) if actions else np.empty((0, ACTION_DIM))
        traces[(episode, seed)] = action_array
        divergence = 0.0
        length_difference = 0
        if off_actions is not None:
            baseline = off_actions.get((episode, seed))
            if baseline is None:
                raise ValueError("paired off-action trace is missing")
            common = min(len(action_array), len(baseline))
            paired = (
                float(np.linalg.norm(action_array[:common] - baseline[:common], axis=1).mean())
                if common
                else 0.0
            )
            divergence = paired
            length_difference = len(action_array) - len(baseline)

        after = agent.ghost.telemetry() if agent.ghost is not None else None
        if after is None:
            evidence = grounding = 0.0
            transfer_support = births = merges = evictions = recouplings = 0
        else:
            evidence = float(after["evidence_support_mean"])
            grounding = float(after["grounding_mean"])
            transfer_support = int(after["transfer_support"])
            births = _telemetry_delta(before, after, "births_total")
            merges = _telemetry_delta(before, after, "merges_total")
            evictions = _telemetry_delta(before, after, "evictions_total")
            recouplings = _telemetry_delta(before, after, "recouplings_total")

        records.append(
            EpisodeRecord(
                task=task,
                condition=condition,
                seed=seed,
                episode=episode,
                status=status,
                success=status == "success",
                death=status in ("death", "starvation"),
                steps=step,
                reward_reporting_only=rewards,
                action_norm_mean=float(np.linalg.norm(action_array, axis=1).mean()),
                action_divergence_from_off=divergence,
                episode_length_difference_from_off=length_difference,
                viability_mean=float(np.mean(viabilities)),
                danger_mean=float(np.mean(dangers)),
                boundary_verdict_counts=verdict_counts,
                rollback_rate=rollbacks / max(step, 1),
                ghost_influence_rate=influenced / max(step, 1),
                population_mean=float(np.mean(populations)),
                evidence_support_mean=evidence,
                grounding_mean=grounding,
                reachability_residual_mean=(
                    float(np.mean(residuals)) if residuals else None
                ),
                transfer_support=transfer_support,
                births=births,
                merges=merges,
                evictions=evictions,
                recouplings=recouplings,
            )
        )
    return records, traces


def _seed_summaries(records: list[dict]) -> list[dict]:
    groups = {}
    for record in records:
        groups.setdefault(
            (record["task"], record["condition"], record["seed"]), []
        ).append(record)
    summaries = []
    for (task, condition, seed), rows in sorted(groups.items()):
        summaries.append(
            {
                "task": task,
                "condition": condition,
                "seed": seed,
                "episodes": len(rows),
                "success_rate": float(np.mean([row["success"] for row in rows])),
                "survival_rate": 1.0 - float(np.mean([row["death"] for row in rows])),
                "danger_mean": float(np.mean([row["danger_mean"] for row in rows])),
                "rollback_mean": float(np.mean([row["rollback_rate"] for row in rows])),
                "influence_mean": float(
                    np.mean([row["ghost_influence_rate"] for row in rows])
                ),
                "action_divergence_mean": float(
                    np.mean([row["action_divergence_from_off"] for row in rows])
                ),
            }
        )
    return summaries


def _paired_values(summaries: list[dict], task: str, a: str, b: str) -> np.ndarray:
    indexed = {
        (row["condition"], row["seed"]): row
        for row in summaries
        if row["task"] == task
    }
    seeds_a = {seed for condition, seed in indexed if condition == a}
    seeds_b = {seed for condition, seed in indexed if condition == b}
    seeds = sorted(seeds_a & seeds_b)
    return np.asarray(
        [indexed[(a, seed)]["success_rate"] - indexed[(b, seed)]["success_rate"] for seed in seeds],
        dtype=np.float64,
    )


def _bootstrap_interval(values: np.ndarray) -> tuple[float, float]:
    if not len(values):
        return math.nan, math.nan
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = rng.choice(values, size=(BOOTSTRAP_SAMPLES, len(values)), replace=True)
    means = samples.mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _sign_flip_p(values: np.ndarray) -> float:
    nonzero = values[values != 0.0]
    if not len(nonzero):
        return 1.0
    observed = abs(float(nonzero.mean()))
    if len(nonzero) <= 20:
        count = 0
        total = 1 << len(nonzero)
        for mask in range(total):
            signs = np.asarray([1.0 if mask & (1 << i) else -1.0 for i in range(len(nonzero))])
            count += abs(float((nonzero * signs).mean())) >= observed - 1e-15
        return count / total
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    count = 0
    for _ in range(BOOTSTRAP_SAMPLES):
        signs = rng.choice((-1.0, 1.0), size=len(nonzero))
        count += abs(float((nonzero * signs).mean())) >= observed - 1e-15
    return (count + 1) / (BOOTSTRAP_SAMPLES + 1)


def _holm_adjusted(p_values: list[float]) -> list[float]:
    if not p_values:
        return []
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=np.float64)
    running = 0.0
    count = len(p_values)
    for rank, index in enumerate(order):
        candidate = min(1.0, (count - rank) * p_values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted.tolist()


def _comparisons(summaries: list[dict], tasks: list[str], conditions: list[str]) -> list[dict]:
    results = []
    for task in tasks:
        pairs = [("influence", "off"), ("influence", "observe")]
        pairs += [(condition, "influence") for condition in ABLATION_CONDITIONS if condition in conditions]
        for a, b in pairs:
            values = _paired_values(summaries, task, a, b)
            if not len(values):
                continue
            low, high = _bootstrap_interval(values)
            results.append(
                {
                    "task": task,
                    "comparison": f"{a}_vs_{b}",
                    "n_seeds": len(values),
                    "mean_success_difference": float(values.mean()),
                    "median_success_difference": float(np.median(values)),
                    "std_success_difference": float(values.std()),
                    "ci95": [low, high],
                    "sign_flip_p": _sign_flip_p(values),
                    "helped": int((values > 0).sum()),
                    "unchanged": int((values == 0).sum()),
                    "harmed": int((values < 0).sum()),
                }
            )
    controlled = [
        index
        for index, result in enumerate(results)
        if result["comparison"] in ("influence_vs_off", "influence_vs_observe")
        and result["task"] in PRIMARY_TASKS
    ]
    adjusted = _holm_adjusted([results[index]["sign_flip_p"] for index in controlled])
    for index, p_value in zip(controlled, adjusted, strict=True):
        results[index]["holm_adjusted_p"] = p_value
    return results


def _gate(records, summaries, comparisons, config, reproducible=False) -> dict:
    expected = {
        (task, condition, seed, episode)
        for task in config["tasks"]
        for condition in config["conditions"]
        for seed in config["seeds"]
        for episode in range(config["eval_episodes"])
    }
    actual = {
        (row["task"], row["condition"], row["seed"], row["episode"])
        for row in records
    }
    complete = expected == actual and len(actual) == len(records)
    primary = [task for task in config["tasks"] if task in PRIMARY_TASKS]
    improvements = []
    safety_ok = True
    for task in primary:
        off = next(
            (row for row in comparisons if row["task"] == task and row["comparison"] == "influence_vs_off"),
            None,
        )
        observe = next(
            (row for row in comparisons if row["task"] == task and row["comparison"] == "influence_vs_observe"),
            None,
        )
        if (
            off
            and observe
            and off["ci95"][0] > 0
            and observe["ci95"][0] > 0
            and off.get("holm_adjusted_p", 1.0) < 0.05
            and observe.get("holm_adjusted_p", 1.0) < 0.05
        ):
            if off["mean_success_difference"] >= 0.10 and observe["mean_success_difference"] >= 0.10:
                improvements.append(task)
        indexed = {
            (row["condition"], row["seed"]): row
            for row in summaries
            if row["task"] == task
        }
        paired_seeds = sorted(
            {seed for condition, seed in indexed if condition == "influence"}
            & {seed for condition, seed in indexed if condition == "off"}
        )
        if paired_seeds:
            survival_difference = np.mean(
                [
                    indexed[("influence", seed)]["survival_rate"]
                    - indexed[("off", seed)]["survival_rate"]
                    for seed in paired_seeds
                ]
            )
            safety_ok = safety_ok and survival_difference >= -0.05
    negative_controls_present = all(
        task in config["tasks"] for task in ("negative_control", "randomized_consequence")
    )
    negative_controls_clean = negative_controls_present
    for task in ("negative_control", "randomized_consequence"):
        comparison = next(
            (
                row
                for row in comparisons
                if row["task"] == task and row["comparison"] == "influence_vs_off"
            ),
            None,
        )
        negative_controls_clean = bool(
            negative_controls_clean
            and comparison is not None
            and comparison["mean_success_difference"] <= 0.05
        )
    required_tasks_present = set(PRIMARY_TASKS).issubset(config["tasks"])
    required_ablations_present = set(ABLATION_CONDITIONS).issubset(config["conditions"])
    ablation_support = False
    for task in improvements:
        ablation_support = ablation_support or any(
            row["task"] == task
            and row["comparison"].endswith("_vs_influence")
            and row["mean_success_difference"] <= -0.05
            for row in comparisons
        )
    return {
        "complete": complete,
        "reproduced": bool(reproducible),
        "required_primary_tasks_present": required_tasks_present,
        "required_ablations_present": required_ablations_present,
        "primary_material_improvements": improvements,
        "negative_controls_present": negative_controls_present,
        "negative_controls_clean": negative_controls_clean,
        "safety_regression_within_5pp": bool(safety_ok),
        "structural_ablation_support": ablation_support,
        "pass": bool(
            complete
            and reproducible
            and required_tasks_present
            and required_ablations_present
            and improvements
            and negative_controls_clean
            and safety_ok
            and ablation_support
        ),
    }


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def run(args) -> dict:
    config = _protocol_config(args)
    protocol_hash = _protocol_hash(config)
    start = time.time()
    records: list[dict] = []

    for task in args.tasks:
        for seed in args.seeds:
            trained = _train(seed, task, args)
            off_records, off_actions = _evaluate(
                trained, task, "off", seed, args.eval_episodes, None
            )
            records.extend(asdict(record) for record in off_records)
            for condition in args.conditions:
                if condition == "off":
                    continue
                condition_records, _ = _evaluate(
                    trained,
                    task,
                    condition,
                    seed,
                    args.eval_episodes,
                    off_actions,
                )
                records.extend(asdict(record) for record in condition_records)

    summaries = _seed_summaries(records)
    comparisons = _comparisons(summaries, list(args.tasks), list(args.conditions))
    payload = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "protocol_hash": protocol_hash,
        "repository_commit": _commit(),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "platform": sys.platform,
        },
        "configuration": config,
        "raw_episodes": records,
        "seed_summaries": summaries,
        "paired_comparisons": comparisons,
        "gate_results": _gate(records, summaries, comparisons, config),
        "failures": [],
        "wallclock_seconds": time.time() - start,
    }
    _atomic_json(Path(args.output), payload)
    return payload


def _parse_seeds(value: str) -> list[int]:
    if ":" in value:
        start, stop = (int(part) for part in value.split(":", maxsplit=1))
        seeds = list(range(start, stop))
    else:
        seeds = [int(part) for part in value.split(",") if part]
    if not seeds or len(seeds) != len(set(seeds)):
        raise argparse.ArgumentTypeError("seeds must be a non-empty unique list")
    return seeds


def _validate_args(args) -> None:
    unknown_tasks = set(args.tasks) - set(ENV_MAP)
    unknown_conditions = set(args.conditions) - set(CONDITIONS)
    if unknown_tasks:
        raise ValueError(f"unknown tasks: {sorted(unknown_tasks)}")
    if unknown_conditions:
        raise ValueError(f"unknown conditions: {sorted(unknown_conditions)}")
    if "off" not in args.conditions:
        raise ValueError("paired validation requires the off condition")
    output = Path(args.output).resolve()
    results_root = (Path.cwd() / "results" / "ghost_validation").resolve()
    if results_root not in output.parents:
        raise ValueError(f"output must be below {results_root}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--conditions", nargs="+", default=["off", "observe", "influence"])
    parser.add_argument("--tasks", nargs="+", default=list(PRIMARY_TASKS))
    parser.add_argument("--seeds", type=_parse_seeds, default=_parse_seeds("0:2"))
    parser.add_argument("--train-episodes", type=int, default=20)
    parser.add_argument("--eval-episodes", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--training-signal", default="legacy_reward")
    parser.add_argument("--credit-mode", default="uniform")
    parser.add_argument("--optimizer-profile", default="legacy")
    parser.add_argument(
        "--output",
        default="results/ghost_validation/exploratory.json",
    )
    args = parser.parse_args()
    _validate_args(args)
    result = run(args)
    print(
        f"wrote {args.output} protocol={result['protocol_hash'][:16]} "
        f"complete={result['gate_results']['complete']} "
        f"pass={result['gate_results']['pass']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
