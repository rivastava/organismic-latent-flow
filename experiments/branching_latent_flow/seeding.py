"""Deterministic central seed handling.

Mirrors the discipline of the public ``olf.seeding`` module: one entry point
that seeds Python, NumPy, and PyTorch so that every experiment is reproducible
from an explicit integer seed.
"""

from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int | None) -> None:
    """Seed all stochastic libraries deterministically.

    Passing ``None`` is a no-op so callers can opt out explicitly.
    """
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class RNG:
    """A small wrapper that exposes reproducible numpy/torch draws under one seed.

    Used by the synthetic process so that generator randomness is fully
    separated from learner randomness and never leaks into the learner.
    """

    def __init__(self, seed: int):
        self.seed = int(seed)
        self.np = np.random.default_rng(self.seed)
        self._torch = torch.Generator()
        self._torch.manual_seed(self.seed)

    def randn(self, *shape):
        return torch.randn(*shape, generator=self._torch)

    def uniform(self, low: float, high: float, size=None):
        return self.np.uniform(low, high, size)

    def integers(self, low: int, high: int, size=None):
        return self.np.integers(low, high, size)
