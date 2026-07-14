"""Changing-world protocol; hidden annotations are scheduler/evaluator only."""

from __future__ import annotations

import copy
import random

import torch

from .protocol import CompletedTrajectory, ProtocolConfig, generate_completed_trajectories


def scheduled_trials(
    *, seed: int, counts: dict[int, int], split_offset: int
) -> list[CompletedTrajectory]:
    total = sum(counts.values())
    config = ProtocolConfig(
        n_modes=4,
        train_trials=max(1200, total * 3),
        eval_trials=max(1200, total * 3),
        seed=seed,
    )
    split = "train" if split_offset == 0 else "eval"
    pool = generate_completed_trajectories(config, split=split)
    selected = []
    for mode, count in counts.items():
        candidates = [trial for trial in pool if trial.mode_index == mode]
        selected.extend(candidates[:count])
    random.Random(seed + split_offset).shuffle(selected)
    if len(selected) != total:
        raise RuntimeError("world scheduler could not produce requested phase")
    return selected


def phase_evaluation_trials(
    *, seed: int, modes: tuple[int, ...], per_mode: int = 60
) -> list[CompletedTrajectory]:
    trials = scheduled_trials(
        seed=seed,
        counts={mode: per_mode for mode in modes},
        split_offset=10_000 + len(modes),
    )
    out = []
    indices = torch.tensor(modes)
    for trial in trials:
        item = copy.copy(trial)
        item.all_mode_endpoints = trial.all_mode_endpoints[indices]
        out.append(item)
    return out
