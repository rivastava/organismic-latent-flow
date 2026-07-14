"""Leakage-controlled data interface for Stage 1 set-valued control."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .protocol import (
    CompletedTrajectory,
    ProtocolConfig,
    generate_completed_trajectories,
    prefix_context,
)


@dataclass(frozen=True)
class ObservedBranchExample:
    """Only tensors that would have been observable after a completed flow."""

    context: torch.Tensor
    probe: torch.Tensor
    endpoint: torch.Tensor


def observed_examples(
    trials: list[CompletedTrajectory],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    examples = [
        ObservedBranchExample(
            context=prefix_context(trial.ambiguous_prefix),
            probe=trial.revealed_prefix[-1],
            endpoint=trial.trajectory[-1],
        )
        for trial in trials
    ]
    return (
        torch.stack([example.context for example in examples]),
        torch.stack([example.probe for example in examples]),
        torch.stack([example.endpoint for example in examples]),
    )


def stage1_trials(
    config: ProtocolConfig,
    *,
    split: str,
) -> list[CompletedTrajectory]:
    return generate_completed_trajectories(config, split=split)
