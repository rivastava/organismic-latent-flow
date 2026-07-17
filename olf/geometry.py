import torch

def project_to_sphere(h, eps=1e-8):
    """
    Projects the hidden state tensor h onto the unit sphere S^(d-1).
    Shape: (batch, dim)
    """
    norm = torch.linalg.norm(h, dim=-1, keepdim=True)
    return h / norm.clamp_min(eps)

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
    x = project_to_sphere(x)
    y = project_to_sphere(y)
    if bool(antipodal(x, y).any()):
        raise ValueError(
            "log map undefined for antipodal points (non-unique geodesic)"
        )
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


def antipodal(x, y, eps: float = 1e-4):
    """Return a per-point mask for numerically antipodal sphere points.

    Parallel transport along a geodesic is undefined when the source and
    target are exact opposites: every great circle through both points is a
    valid path, so the transported tangent is not unique.
    """
    dot = (x * y).sum(dim=-1)
    return dot <= -1.0 + eps


def parallel_transport_sphere(x, y, v, eps: float = 1e-8):
    """Parallel transport the tangent vector ``v`` (at ``x``) to the tangent
    space at ``y``, both on the unit sphere S^(d-1).

    Closed form along the geodesic from x to y:

        û  = (y - <x,y> x) / sin(theta)          # unit tangent at x toward y
        v_par   = <v, û> û                         # component along transport dir
        v_perp  = v - v_par                        # component in orthogonal complement
        û' = -sin(theta) x + cos(theta) û         # transported direction at y
        T(v) = <v, û> û'  +  v_perp                # v_perp is parallel-invariant

    The result is projected back onto the tangent at y for numerical safety.
    Raises ValueError if x and y are (numerically) antipodal.
    """
    x = project_to_sphere(x)
    y = project_to_sphere(y)
    if bool(antipodal(x, y).any()):
        raise ValueError(
            "parallel transport undefined for antipodal points (non-unique geodesic)"
        )
    dot = (x * y).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
    sin_theta = torch.sqrt((1.0 - dot * dot).clamp_min(0.0))
    cos_theta = dot
    safe_sin_theta = sin_theta.clamp_min(eps)
    u_hat = (y - cos_theta * x) / safe_sin_theta      # unit tangent at x toward y
    s = (v * u_hat).sum(dim=-1, keepdim=True)           # <v, û>
    v_par = s * u_hat
    v_perp = v - v_par                                  # invariant complement
    u_prime = -sin_theta * x + cos_theta * u_hat        # transported direction at y
    # Parallel transport: the along-geodesic component moves with the geodesic
    # direction; the orthogonal complement is invariant under the transport.
    general = s * u_prime + v_perp
    identity = project_to_tangent(y, v)
    transported = torch.where(sin_theta < eps, identity, general)
    return project_to_tangent(y, transported)


def angular_distance(a, b):
    """Great-circle (angular) distance between two points on the unit sphere."""
    a = project_to_sphere(a)
    b = project_to_sphere(b)
    cos = (a * b).sum(dim=-1).clamp(-1.0, 1.0)
    return torch.acos(cos)
