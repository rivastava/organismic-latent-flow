"""Diagnostic checks for the ghost integration.

These checks are conservative: they never silently clip, skip, or convert
non-finite values to zero/null. A non-finite quantity is a hard error.
"""

import torch

from .config import PROHIBITED_LABEL_SUBSTRINGS


def finite_or_raise(name: str, t: torch.Tensor) -> torch.Tensor:
    if not torch.isfinite(t).all():
        raise ValueError(f"{name} is non-finite (NaN/Inf); refusing to proceed")
    return t


def assert_no_prohibited_labels(obj, path: str = "root") -> None:
    """Fail if any attribute name / dict key carries a prohibited substring.

    Ghosts must not carry relation, reward, success, task, world, role,
    identity, meaning, goal, scripted, privileged, or benchmark information.
    """
    if isinstance(obj, dict):
        items = obj.items()
    elif isinstance(obj, (list, tuple)):
        items = [(f"[{i}]", v) for i, v in enumerate(obj)]
    else:
        items = [(k, getattr(obj, k)) for k in dir(obj)
                 if not k.startswith("__") and not callable(getattr(obj, k))]
    for key, value in items:
        key_str = str(key).lower()
        for sub in PROHIBITED_LABEL_SUBSTRINGS:
            if sub in key_str:
                raise AssertionError(
                    f"prohibited label substring '{sub}' found in {path}.{key}"
                )
        if isinstance(value, (dict, list, tuple)) or hasattr(value, "__dict__"):
            # Recurse only into plain containers / dataclasses, not tensors.
            if not isinstance(value, torch.Tensor):
                assert_no_prohibited_labels(value, f"{path}.{key}")


def check_sphere_norm(points: torch.Tensor, eps: float = 1e-3) -> bool:
    """All rows of ``points`` (..., D) must lie on the unit sphere."""
    norms = points.reshape(-1, points.shape[-1]).norm(dim=-1)
    return bool((norms - 1.0).abs().max() <= eps)


def check_tangent_validity(anchor: torch.Tensor, tangent: torch.Tensor,
                           eps: float = 1e-3) -> bool:
    """``tangent`` must be orthogonal to ``anchor`` (tangent-space validity)."""
    dot = (anchor.reshape(-1, anchor.shape[-1]) * tangent.reshape(-1, tangent.shape[-1])).sum(dim=-1)
    return bool(dot.abs().max() <= eps)


def check_permutation_equivariant(op, population, perm: torch.Tensor) -> bool:
    """``op`` over the population must commute with permutation.

    ``op`` maps a GhostPopulation to a stacked tensor over the ghost axis.
    Equivariance: op(pop.permute(perm))[perm_inv] == op(pop).
    """
    base = op(population)
    permuted = op(population.permute(perm))
    perm = perm.long().reshape(-1)
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(perm.numel())
    if base.shape != permuted.shape:
        return False
    reordered = permuted[inv]
    return bool(torch.allclose(base, reordered, atol=1e-5, rtol=1e-5))


def grounding_monotone_nonincreasing(before: torch.Tensor, after: torch.Tensor) -> bool:
    """Grounding must never increase unless recoupling supplied evidence.

    Used as an assertion that an internal (non-recoupled) update does not
    raise grounding. ``before`` and ``after`` are per-ghost grounding tensors.
    """
    return bool((after <= before + 1e-6).all())
