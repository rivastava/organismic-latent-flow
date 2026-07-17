"""Strict merger for independently written ghost-validation shards."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

from experiments.ghost_validation import (
    _atomic_json,
    _comparisons,
    _gate,
    _protocol_hash,
    _seed_summaries,
)


def _shard_key(record: dict) -> tuple:
    return (
        record["task"],
        record["condition"],
        int(record["seed"]),
        int(record["episode"]),
    )


def merge(paths: list[Path], output: Path) -> dict:
    if not paths:
        raise ValueError("at least one shard is required")
    shards = [json.loads(path.read_text()) for path in paths]
    reference = shards[0]
    reference_config = reference["configuration"]
    fixed_keys = set(reference_config) - {"seeds"}
    records = []
    seen = set()
    seeds = set()

    for path, shard in zip(paths, shards, strict=True):
        if shard.get("schema_version") != 1:
            raise ValueError(f"unsupported schema in {path}")
        if shard.get("protocol") != reference.get("protocol"):
            raise ValueError(f"protocol mismatch in {path}")
        if shard.get("repository_commit") != reference.get("repository_commit"):
            raise ValueError(f"repository commit mismatch in {path}")
        if shard.get("environment") != reference.get("environment"):
            raise ValueError(f"environment mismatch in {path}")
        if shard.get("failures"):
            raise ValueError(f"failed shard cannot be merged: {path}")
        config = shard["configuration"]
        if shard.get("protocol_hash") != _protocol_hash(config):
            raise ValueError(f"invalid protocol hash in {path}")
        for key in fixed_keys:
            if config.get(key) != reference_config.get(key):
                raise ValueError(f"configuration mismatch for {key} in {path}")
        expected = {
            (task, condition, seed, episode)
            for task in config["tasks"]
            for condition in config["conditions"]
            for seed in config["seeds"]
            for episode in range(config["eval_episodes"])
        }
        actual = {_shard_key(record) for record in shard["raw_episodes"]}
        if actual != expected or len(actual) != len(shard["raw_episodes"]):
            raise ValueError(f"incomplete or duplicate records in {path}")
        overlap = seen & actual
        if overlap:
            raise ValueError(f"duplicate records across shards: {path}")
        seen.update(actual)
        records.extend(shard["raw_episodes"])
        seeds.update(int(seed) for seed in config["seeds"])

    config = deepcopy(reference_config)
    config["seeds"] = sorted(seeds)
    summaries = _seed_summaries(records)
    comparisons = _comparisons(summaries, config["tasks"], config["conditions"])
    payload = {
        "schema_version": 1,
        "protocol": reference["protocol"],
        "protocol_hash": _protocol_hash(config),
        "repository_commit": reference["repository_commit"],
        "environment": reference["environment"],
        "configuration": config,
        "raw_episodes": records,
        "seed_summaries": summaries,
        "paired_comparisons": comparisons,
        "gate_results": _gate(records, summaries, comparisons, config),
        "failures": [],
        "merged_shards": [str(path) for path in paths],
        "wallclock_seconds": sum(float(shard.get("wallclock_seconds", 0.0)) for shard in shards),
    }
    _atomic_json(output, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("shards", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    results_root = (Path.cwd() / "results" / "ghost_validation").resolve()
    if results_root not in output.parents:
        raise ValueError(f"output must be below {results_root}")
    payload = merge(args.shards, output)
    print(
        f"wrote {output} shards={len(args.shards)} "
        f"seeds={len(payload['configuration']['seeds'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
