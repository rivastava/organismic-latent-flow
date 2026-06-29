"""olf/attractor.py

Constitutional attractor network per Constitution §6.

"A goal is a preferred region or attractor in latent dynamics."

This module provides:
  - LatentAttractor: a fixed point in latent space (S^{d-1}) that the
    flow tends to. Each attractor is a unit vector on the sphere.
  - AttractorField: a set of attractors with associated weights. The
    flow field biases the organism's h toward active attractors.
  - GoalUpdate: a rule for dissolving old attractors and creating new
    ones based on consequence. "The organism must be able to dissolve
    old attractors." (§6)

The attractor field is constitutional:
  - Goals are NOT symbolic instructions.
  - Goals are regions in latent space, not commands.
  - Attractors can be dissolved when they become harmful (§6: "A goal
    may become harmful in a different self-state. A once-useful action
    may become danger.").
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LatentAttractor(nn.Module):
    """A single latent attractor (a unit vector on S^{d-1}).

    The attractor is a learned unit vector a ∈ S^{d-1}. The flow field
    can be biased to point toward this attractor.
    """

    def __init__(self, latent_dim=32):
        super().__init__()
        self.latent_dim = latent_dim
        # Initialize as a random unit vector.
        v = torch.randn(latent_dim)
        self.a = nn.Parameter(v / v.norm())

    def get(self):
        """Returns the unit-vector attractor."""
        return F.normalize(self.a, p=2, dim=-1)


class AttractorField(nn.Module):
    """A set of latent attractors with weights and dissolution rules.

    The field holds K attractors. Each attractor has a weight w ∈ [0, 1]
    indicating its current salience. The flow field at h produces a
    bias toward weighted attractors:

        h_tendency = h + Σ_k w_k * (a_k - h) * dt

    Attractors can be dissolved (weight → 0) or created (add a new
    attractor at h) based on the organism's experience.
    """

    def __init__(self, latent_dim=32, max_attractors=8):
        super().__init__()
        self.latent_dim = latent_dim
        self.max_attractors = max_attractors
        # K attractors as parameters (we overprovision; only the first
        # `n_active` are used).
        attractors = []
        for _ in range(max_attractors):
            v = torch.randn(latent_dim)
            attractors.append(v / v.norm())
        self.attractors = nn.Parameter(torch.stack(attractors))
        # Per-attractor weight (logits, passed through sigmoid).
        self.weights = nn.Parameter(torch.zeros(max_attractors))
        # Per-attractor "harm" counter; if it exceeds a threshold, the
        # attractor is dissolved (weight → 0).
        self.register_buffer("harm_counters", torch.zeros(max_attractors))

    def get_active_attractors(self):
        """Returns (active_attractors, active_weights) where active is
        those with weight > 0.1.
        """
        weights = torch.sigmoid(self.weights)
        active_mask = weights > 0.1
        active_a = F.normalize(self.attractors, p=2, dim=-1)[active_mask]
        active_w = weights[active_mask]
        return active_a, active_w, active_mask

    def compute_tendency(self, h, dt=0.1):
        """Returns a latent vector that biases h toward the weighted
        attractors. The organism's flow can blend this with its
        current trajectory.

        Args:
            h: (batch, latent_dim) current latent state
            dt: float, "speed" of attractor pull

        Returns:
            tendency: (batch, latent_dim) the biased latent vector.
            bias: scalar, sum of weights (used for diagnostic logging).
        """
        if h.dim() == 1:
            h = h.unsqueeze(0)
        weights = torch.sigmoid(self.weights)  # (K,)
        all_a = F.normalize(self.attractors, p=2, dim=-1)  # (K, D)
        # Weighted centroid of all attractors.
        total_w = weights.sum()
        if total_w < 1e-6:
            return h, 0.0
        centroid = (weights.unsqueeze(-1) * all_a).sum(dim=0, keepdim=True)  # (1, D)
        centroid = F.normalize(centroid, p=2, dim=-1)
        # Tendency: pull h toward the weighted centroid.
        diff = centroid - h  # (batch, D)
        tendency = h + dt * diff
        tendency = F.normalize(tendency, p=2, dim=-1)
        # v0.3.1.2 diagnostic: per-call log of tendency magnitude.
        if getattr(self, "diag_log_target", None) is not None:
            self.diag_log_target.append({
                "n_active": int((torch.sigmoid(self.weights) > 0.1).sum().item()),
                "total_weight": float(total_w.item()),
                "tendency_norm": float(tendency.norm().item()),
            })
        return tendency, float(total_w.item())

    def dissolve(self, idx):
        """Constitution §6: dissolve an old attractor."""
        with torch.no_grad():
            self.weights[idx] = -10.0  # sigmoid(-10) ≈ 0

    def create_at(self, h, idx=None):
        """Create a new attractor at h.

        If idx is None, pick an inactive slot.
        """
        if h.dim() > 1:
            h = h.squeeze(0)
        h_n = F.normalize(h, p=2, dim=-1)
        if idx is None:
            weights = torch.sigmoid(self.weights)
            inactive = (weights < 0.1).nonzero(as_tuple=False).squeeze(-1)
            if inactive.numel() == 0:
                return None  # no room
            idx = int(inactive[0].item())
        with torch.no_grad():
            self.attractors[idx] = h_n
            self.weights[idx] = 0.0  # sigmoid(0) = 0.5
            self.harm_counters[idx] = 0.0
        return idx

    def record_harm(self, idx):
        """Increment harm counter for an attractor. If it exceeds a
        threshold, dissolve the attractor.
        """
        self.harm_counters[idx] += 1.0
        if self.harm_counters[idx] > 3:
            self.dissolve(idx)
