import json
from argparse import Namespace
from copy import deepcopy

import pytest

from experiments.ghost_validation import _gate, _protocol_config, _protocol_hash
from experiments.ghost_validation_merge import merge


def _config(seeds):
    args = Namespace(
        conditions=["off", "observe", "influence"],
        tasks=["delayed_lure"],
        seeds=seeds,
        train_episodes=2,
        eval_episodes=2,
        learning_rate=0.01,
        training_signal="legacy_reward",
        credit_mode="uniform",
        optimizer_profile="legacy",
    )
    return _protocol_config(args)


def _record(condition, seed, episode):
    return {
        "task": "delayed_lure",
        "condition": condition,
        "seed": seed,
        "episode": episode,
        "success": False,
        "death": False,
        "danger_mean": 0.0,
        "rollback_rate": 0.0,
        "ghost_influence_rate": float(condition == "influence"),
        "action_divergence_from_off": 0.0,
        "episode_length_difference_from_off": 0,
        "viability_mean": 1.0,
        "boundary_verdict_counts": {"release": 1, "hold": 0, "recouple": 0, "rollback": 0},
    }


def _shard(seeds):
    config = _config(seeds)
    records = [
        _record(condition, seed, episode)
        for condition in config["conditions"]
        for seed in seeds
        for episode in range(config["eval_episodes"])
    ]
    return {
        "schema_version": 1,
        "protocol": "ghost_stage10",
        "protocol_hash": _protocol_hash(config),
        "repository_commit": "abc123",
        "environment": {"python": "test"},
        "configuration": config,
        "raw_episodes": records,
        "failures": [],
        "wallclock_seconds": 1.0,
    }


def test_gate_completeness_is_computed_from_records():
    config = _config([0])
    records = _shard([0])["raw_episodes"]
    complete = _gate(records, [], [], config)
    assert complete["complete"] is True
    assert complete["reproduced"] is False
    assert complete["required_primary_tasks_present"] is False
    assert complete["pass"] is False

    incomplete = _gate(records[:-1], [], [], config)
    assert incomplete["complete"] is False


def test_merge_rebuilds_configuration_and_hash(tmp_path):
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    output = tmp_path / "merged.json"
    first.write_text(json.dumps(_shard([0])))
    second.write_text(json.dumps(_shard([1])))

    result = merge([first, second], output)

    assert result["configuration"]["seeds"] == [0, 1]
    assert len(result["raw_episodes"]) == 12
    assert result["gate_results"]["complete"] is True
    assert result["protocol_hash"] == _protocol_hash(result["configuration"])


def test_merge_rejects_incompatible_or_duplicate_shards(tmp_path):
    first_payload = _shard([0])
    incompatible = _shard([1])
    incompatible["configuration"]["eval_episodes"] = 3
    incompatible["protocol_hash"] = _protocol_hash(incompatible["configuration"])
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text(json.dumps(first_payload))
    second.write_text(json.dumps(incompatible))

    with pytest.raises(ValueError, match="configuration mismatch"):
        merge([first, second], tmp_path / "merged.json")

    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(json.dumps(deepcopy(first_payload)))
    with pytest.raises(ValueError, match="duplicate records"):
        merge([first, duplicate], tmp_path / "merged.json")


def test_merge_rejects_invalid_hash_or_environment(tmp_path):
    first_payload = _shard([0])
    invalid_hash = _shard([1])
    invalid_hash["protocol_hash"] = "invalid"
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text(json.dumps(first_payload))
    second.write_text(json.dumps(invalid_hash))
    with pytest.raises(ValueError, match="invalid protocol hash"):
        merge([first, second], tmp_path / "merged.json")

    other_environment = _shard([1])
    other_environment["environment"] = {"python": "different"}
    second.write_text(json.dumps(other_environment))
    with pytest.raises(ValueError, match="environment mismatch"):
        merge([first, second], tmp_path / "merged.json")


def test_merge_rejects_failed_or_incomplete_shards(tmp_path):
    failed_payload = _shard([0])
    failed_payload["failures"] = ["failure"]
    failed = tmp_path / "failed.json"
    failed.write_text(json.dumps(failed_payload))
    with pytest.raises(ValueError, match="failed shard"):
        merge([failed], tmp_path / "merged.json")

    incomplete_payload = _shard([0])
    incomplete_payload["raw_episodes"].pop()
    incomplete = tmp_path / "incomplete.json"
    incomplete.write_text(json.dumps(incomplete_payload))
    with pytest.raises(ValueError, match="incomplete"):
        merge([incomplete], tmp_path / "merged.json")
