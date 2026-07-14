"""Causal held-out protocol for branching latent flow.

The learner receives observation prefixes only at evaluation time. Training
targets come from completed *past* trajectories, and model parameters persist
across those trajectories. Generator metadata is retained solely for offline
scoring.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .geometry import exponential_map, project_to_sphere, slerp
from .seeding import RNG


@dataclass(frozen=True)
class ProtocolConfig:
    latent_dim: int = 8
    n_modes: int = 3
    history_steps: int = 6
    future_steps: int = 12
    endpoint_arc: float = 1.0
    probe_arc: float = 0.18
    train_trials: int = 600
    eval_trials: int = 240
    seed: int = 2026


@dataclass
class CompletedTrajectory:
    """One completed experience.

    Only ``ambiguous_prefix``, ``revealed_prefix`` and ``trajectory`` are
    available to training code. The remaining fields are offline annotations.
    """

    ambiguous_prefix: torch.Tensor
    revealed_prefix: torch.Tensor
    trajectory: torch.Tensor
    endpoint: torch.Tensor
    mode_index: int
    all_mode_endpoints: torch.Tensor


def _skew_generators(cfg: ProtocolConfig) -> torch.Tensor:
    """Create deterministic, distinct tangent transformations."""
    rng = RNG(cfg.seed + 91_771)
    mats = []
    for _ in range(cfg.n_modes):
        raw = rng.randn(cfg.latent_dim, cfg.latent_dim)
        skew = raw - raw.T
        skew = skew / skew.norm().clamp_min(1e-8)
        mats.append(skew)
    return torch.stack(mats)


def _mode_endpoint(branch: torch.Tensor, generator: torch.Tensor, arc: float) -> torch.Tensor:
    tangent = generator @ branch
    tangent = tangent - (tangent * branch).sum() * branch
    tangent = tangent / tangent.norm().clamp_min(1e-8)
    return exponential_map(branch, tangent * arc)


def generate_completed_trajectories(
    cfg: ProtocolConfig,
    *,
    split: str,
) -> list[CompletedTrajectory]:
    """Generate train or held-out evaluation experiences.

    Train and evaluation contexts use disjoint RNG streams while sharing the
    same world transformations. This makes continuation laws learnable without
    repeating specific sphere points.
    """
    if split not in {"train", "eval"}:
        raise ValueError("split must be 'train' or 'eval'")
    count = cfg.train_trials if split == "train" else cfg.eval_trials
    stream_offset = 0 if split == "train" else 1_000_003
    rng = RNG(cfg.seed + stream_offset)
    generators = _skew_generators(cfg)
    out = []
    for _ in range(count):
        start = project_to_sphere(rng.randn(cfg.latent_dim))
        branch = project_to_sphere(rng.randn(cfg.latent_dim))
        # The flow settles for one observation at the branch. This makes the
        # event discoverable from experience itself: the ambiguous prefix ends
        # with zero motion, while a revealed prefix contains an observed
        # departure. No branch index or phase label is provided to the model.
        moving = slerp(
            start.unsqueeze(0).expand(cfg.history_steps, -1),
            branch.unsqueeze(0).expand(cfg.history_steps, -1),
            torch.linspace(0.0, 1.0, cfg.history_steps).unsqueeze(-1),
        )
        history = torch.cat([moving, branch.unsqueeze(0)], dim=0)
        endpoints = torch.stack(
            [_mode_endpoint(branch, generators[m], cfg.endpoint_arc) for m in range(cfg.n_modes)]
        )
        mode = int(rng.integers(0, cfg.n_modes))
        endpoint = endpoints[mode]
        probe = _mode_endpoint(branch, generators[mode], cfg.probe_arc)
        future = slerp(
            branch.unsqueeze(0).expand(cfg.future_steps + 1, -1),
            endpoint.unsqueeze(0).expand(cfg.future_steps + 1, -1),
            torch.linspace(0.0, 1.0, cfg.future_steps + 1).unsqueeze(-1),
        )
        out.append(
            CompletedTrajectory(
                ambiguous_prefix=history,
                revealed_prefix=torch.cat([history, probe.unsqueeze(0)], dim=0),
                trajectory=torch.cat([history, future[1:]], dim=0),
                endpoint=endpoint,
                mode_index=mode,
                all_mode_endpoints=endpoints,
            )
        )
    return out


def training_examples(trials: list[CompletedTrajectory]) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert completed past experience into prefix/endpoint training pairs."""
    contexts = []
    targets = []
    for trial in trials:
        contexts.append(prefix_context(trial.ambiguous_prefix))
        targets.append(trial.trajectory[-1])
        contexts.append(prefix_context(trial.revealed_prefix))
        targets.append(trial.trajectory[-1])
    return torch.stack(contexts), torch.stack(targets)


def prefix_context(prefix: torch.Tensor) -> torch.Tensor:
    """Build an observable context without event indices or generator state."""
    if prefix.ndim != 2 or prefix.shape[0] < 2:
        raise ValueError("prefix must have shape (time>=2, latent_dim)")
    current = prefix[-1]
    previous = prefix[-2]
    displacement = current - previous
    return torch.cat([current, previous, displacement], dim=-1)
