"""Bounded, permutation-equivariant set of role-free ghost trajectories.

The population starts EMPTY. Before any external evidence the organism has no
ghost trajectories, so no latent axis can influence action. A trajectory is
born only from an externally observed deformation (see lifecycle.maybe_birth).
This satisfies the constitutional rule that the first nonzero ghost tangent
must come from external evidence, never from an invented coordinate direction.
"""

import torch

from olf.geometry import angular_distance, project_to_sphere

from .trajectory import GhostTrajectory, transport_ghost


class GhostPopulation:
    """A finite set of temporary spherical trajectories.

    The set is order-agnostic: any operation over the population must commute
    with a permutation of the member ordering. Population-level tensors are
    stacked along dim 0 (the ghost axis) so a permutation of that axis is the
    only operation needed to re-order the set.
    """

    def __init__(self, latent_dim: int, capacity: int):
        if capacity < 0:
            raise ValueError("capacity must be >= 0")
        if latent_dim < 2:
            raise ValueError("latent_dim must be >= 2")
        self._latent_dim = latent_dim
        self.capacity = capacity
        self._ghosts: list[GhostTrajectory] = []
        # No neutral placeholder is created: an invented anchor/tangent would
        # bias a latent axis. The population is filled only by external birth.

    @classmethod
    def empty(cls, latent_dim: int, capacity: int) -> "GhostPopulation":
        """An empty population of the given capacity."""
        obj = cls.__new__(cls)
        obj._latent_dim = latent_dim
        obj.capacity = capacity
        obj._ghosts = []
        return obj

    def __len__(self) -> int:
        return len(self._ghosts)

    def __getitem__(self, i) -> GhostTrajectory:
        return self._ghosts[i]

    def to(self, device):
        self._ghosts = [g.to(device) for g in self._ghosts]
        return self

    @property
    def latent_dim(self) -> int:
        return self._latent_dim

    # ---- stacking (ghost axis = dim 0) -----------------------------------
    def stack(self, field: str) -> torch.Tensor:
        return torch.stack([getattr(g, field) for g in self._ghosts], dim=0)

    def anchors(self) -> torch.Tensor:
        return self.stack("anchor")

    def tangents(self) -> torch.Tensor:
        return self.stack("tangent")

    def scalars(self, field: str) -> torch.Tensor:
        return self.stack(field)

    # ---- mutation (kept minimal + diagnostic) ----------------------------
    def append(self, ghost: GhostTrajectory) -> bool:
        if len(self._ghosts) >= self.capacity:
            return False
        self._ghosts.append(ghost)
        return True

    def remove_at(self, idx: int) -> None:
        self._ghosts.pop(idx)

    def replace_at(self, idx: int, ghost: GhostTrajectory) -> None:
        self._ghosts[idx] = ghost

    def clear(self) -> None:
        self._ghosts.clear()

    # ---- permutation equivariance ----------------------------------------
    def permute(self, perm: torch.Tensor) -> "GhostPopulation":
        """Return a new population reordered by ``perm`` (long tensor)."""
        perm = perm.long().reshape(-1)
        if int(perm.numel()) != len(self._ghosts):
            raise ValueError("perm length must equal population size")
        out = GhostPopulation.__new__(GhostPopulation)
        out.capacity = self.capacity
        out._latent_dim = self._latent_dim
        out._ghosts = [self._ghosts[int(i)] for i in perm]
        return out

    def transport(self, real_prev, real_now, step: float) -> "GhostPopulation":
        out = GhostPopulation.__new__(GhostPopulation)
        out.capacity = self.capacity
        out._latent_dim = self._latent_dim
        out._ghosts = [
            transport_ghost(g, real_prev, real_now, step) for g in self._ghosts
        ]
        return out

    def tension(self, step: float) -> dict:
        """Measure disagreement among externally grounded future trajectories.

        The pairwise Fréchet energy is permutation-invariant and does not need a
        privileged thesis, antithesis, or ambient coordinate axis. Predictive
        uncertainty supplies the scale against which disagreement is measured.
        With fewer than two supported trajectories, tension is undefined rather
        than evidence of agreement.
        """
        if len(self._ghosts) < 2:
            return _undefined_tension(len(self._ghosts))

        credibility = self.scalars("credibility").clamp_min(0.0)
        grounding = self.scalars("grounding").clamp_min(0.0)
        raw_weights = credibility * grounding
        supported = raw_weights > 0.0
        support_count = int(supported.sum().item())
        if support_count < 2:
            return _undefined_tension(support_count)

        weights = raw_weights[supported]
        weight_sum = weights.sum()
        if float(weight_sum) <= torch.finfo(weights.dtype).eps:
            return _undefined_tension(support_count)
        weights = weights / weight_sum

        futures = torch.stack(
            [
                ghost.predicted_anchor(step)
                for ghost, keep in zip(self._ghosts, supported, strict=True)
                if bool(keep)
            ],
            dim=0,
        )
        pairwise = angular_distance(futures[:, None, :], futures[None, :, :])
        pair_weights = weights[:, None] * weights[None, :]
        energy = 0.5 * torch.sum(pair_weights * pairwise.square())

        uncertainty = self.scalars("uncertainty")[supported].clamp_min(0.0)
        uncertainty_energy = torch.sum(weights * uncertainty.square())
        denominator = energy + uncertainty_energy
        normalized = torch.where(
            denominator > torch.finfo(denominator.dtype).eps,
            energy / denominator,
            torch.zeros_like(denominator),
        )
        effective_support = 1.0 / torch.sum(weights.square()).clamp_min(
            torch.finfo(weights.dtype).eps
        )

        resultant = torch.sum(weights[:, None] * futures, dim=0)
        centroid = None
        if float(resultant.norm()) > torch.finfo(resultant.dtype).eps:
            centroid = project_to_sphere(resultant)

        return {
            "defined": True,
            "supported": support_count,
            "effective_support": float(effective_support),
            "energy": float(energy),
            "uncertainty_energy": float(uncertainty_energy),
            "normalized": float(normalized.clamp(0.0, 1.0)),
            "centroid": centroid,
        }


def _undefined_tension(supported: int) -> dict:
    return {
        "defined": False,
        "supported": int(supported),
        "effective_support": float(supported),
        "energy": None,
        "uncertainty_energy": None,
        "normalized": None,
        "centroid": None,
    }
