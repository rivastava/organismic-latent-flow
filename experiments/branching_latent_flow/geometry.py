"""Spherical geometry primitives for the branching-latent-flow research track.

These mirror the conventions used in the public OLF ``geometry.py`` (unit-sphere
projection, tangent projection, exponential/log maps) but are implemented
independently inside this isolated experiment.

All vectors are row-stacked ``(..., d)`` tensors on the unit sphere ``S^{d-1}``.
"""

from __future__ import annotations

import torch

EPS = 1e-8


def project_to_sphere(h: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    """Project onto the unit sphere ``S^{d-1}``."""
    norm = torch.linalg.norm(h, dim=-1, keepdim=True)
    return h / (norm + eps)


def project_to_tangent(h: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """Project raw vector ``u`` onto the tangent space at ``h``.

    ``Pi_h u = u - (h . u) h``.
    """
    dot = torch.sum(h * u, dim=-1, keepdim=True)
    return u - dot * h


def exponential_map(h: torch.Tensor, v: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    """Exponential map of tangent vector ``v`` at ``h`` back onto the sphere."""
    norm_v = torch.linalg.norm(v, dim=-1, keepdim=True)
    mask = (norm_v > eps).float()
    cos_coeff = torch.cos(norm_v)
    sin_coeff = torch.sin(norm_v) / (norm_v + eps)
    mapped = cos_coeff * h + mask * sin_coeff * v
    return project_to_sphere(mapped)


def log_map_sphere(x: torch.Tensor, y: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    """Tangent vector at ``x`` that exp-maps to ``y``."""
    dot = (x * y).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
    angle = torch.acos(dot)
    y_proj = y - x * dot
    y_proj_norm = y_proj.norm(dim=-1, keepdim=True)
    scale = torch.where(
        y_proj_norm > eps,
        angle / y_proj_norm.clamp(min=eps),
        torch.zeros_like(y_proj_norm),
    )
    return project_to_tangent(x, y_proj * scale)


def slerp(
    x: torch.Tensor,
    y: torch.Tensor,
    t: torch.Tensor,
    eps: float = EPS,
) -> torch.Tensor:
    """Spherical linear interpolation from ``x`` toward ``y``.

    ``t`` broadcasts against the trailing dims and may be a scalar or a tensor
    of shape ``(..., 1)``. ``t=0`` returns ``x``; ``t=1`` returns ``y``.
    """
    omega = torch.acos((x * y).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0))
    sin_omega = torch.sin(omega).clamp_min(eps)
    a = torch.sin((1.0 - t) * omega) / sin_omega
    b = torch.sin(t * omega) / sin_omega
    out = a * x + b * y
    return project_to_sphere(out)


def angular_distance(x: torch.Tensor, y: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    """Great-circle angular distance (radians) between unit vectors."""
    dot = (x * y).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
    return torch.acos(dot)
