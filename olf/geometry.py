import torch

def project_to_sphere(h, eps=1e-8):
    """
    Projects the hidden state tensor h onto the unit sphere S^(d-1).
    Shape: (batch, dim)
    """
    norm = torch.linalg.norm(h, dim=-1, keepdim=True)
    return h / (norm + eps)

def project_to_tangent(h, u):
    """
    Projects raw flow vector u onto the tangent space of the unit sphere at state h.
    Formula: Pi_h u = u - (h^T u) h
    """
    # Dot product of h and u for each batch element
    dot_prod = torch.sum(h * u, dim=-1, keepdim=True)
    return u - dot_prod * h

def exponential_map(h, v, eps=1e-8):
    """
    Maps tangent vector v at point h back onto the sphere S^(d-1) exactly.
    Formula: exp_h(v) = cos(||v||)*h + sin(||v||)*(v/||v||)
    """
    norm_v = torch.linalg.norm(v, dim=-1, keepdim=True)
    
    # Avoid division by zero when norm_v is tiny
    mask = (norm_v > eps).float()
    
    # Cosine and Sine coefficients
    cos_coeff = torch.cos(norm_v)
    sin_coeff = torch.sin(norm_v) / (norm_v + eps)
    
    mapped = cos_coeff * h + mask * sin_coeff * v
    # Project to make sure numerical stability is maintained
    return project_to_sphere(mapped)


def log_map_sphere(x, y, eps=1e-8):
    """Logarithmic map on S^{d-1}: tangent direction at x pointing to y.

    Inverse of exponential_map(x, v) = y. Returns the tangent vector v
    at x such that exp_map(x, v) ≈ y. Used by MotorMemory to store
    transformation directions instead of Euclidean deltas (which vanish
    on the compact sphere).

    Args:
        x: (..., d) source point on the sphere.
        y: (..., d) target point on the sphere.
        eps: small constant for numerical stability.

    Returns:
        v: (..., d) tangent vector at x such that exp_map(x, v) ≈ y.
    """
    dot = (x * y).sum(dim=-1, keepdim=True).clamp(-1.0 + eps, 1.0 - eps)
    angle = torch.acos(dot)
    y_proj = y - x * dot
    y_proj_norm = y_proj.norm(dim=-1, keepdim=True).clamp(min=eps)
    v = y_proj / y_proj_norm * angle
    return v
