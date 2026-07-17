"""Role-free spherical ghost trajectory.

A ghost is a *temporary alternative trajectory* on the shared spherical latent
substrate. It carries only role-free, observable quantities:

  - anchor:      a point on the unit sphere S^(D-1)          shape (D,)
  - tangent:     transported tangent deformation at anchor    shape (D,)
  - credibility: scalar in [0, 1]  (predictive track record)
  - grounding:   scalar in [0, 1]  (external recoupling only; never rises
                 without positive external evidence)
  - uncertainty: scalar >= 0      (predictive spread)
  - persistence: scalar >= 0      (steps since birth / last support)
  - evidence_support:    scalar >= 0  (cumulative POSITIVE external support)
  - evidence_negative:   scalar >= 0  (cumulative NEGATIVE external support)
  - boundary_compat: scalar in [0, 1] (boundary risk track record)
  - horizon_expr: scalar in [0, 1] (strength of its future expression)

No semantic label, task, identity, world, relation, or reward field exists.

Action-conditioned evidence:
  Each ghost records the externally observed (released_action, observed_tangent)
  pairs. A locally fitted, ridge-regularized linear transfer map learns
      observed_tangent ~ M @ released_action
  in the tangent space at the current anchor. This is bounded, role-free,
  externally grounded, deterministic, and explicitly experimental scaffolding:
  before enough (action, tangent) evidence exists the ghost cannot form an
  action-conditioned prediction; before then its observed signature is used.
"""

import torch
from dataclasses import dataclass, field

from olf.geometry import (
    antipodal,
    exponential_map,
    log_map_sphere,
    parallel_transport_sphere,
    project_to_sphere,
    project_to_tangent,
)


@dataclass
class GhostTrajectory:
    anchor: torch.Tensor
    tangent: torch.Tensor
    credibility: torch.Tensor
    grounding: torch.Tensor
    uncertainty: torch.Tensor
    persistence: torch.Tensor
    evidence_support: torch.Tensor
    evidence_negative: torch.Tensor
    boundary_compat: torch.Tensor
    horizon_expr: torch.Tensor
    # Action-conditioned evidence: lists of detached (action, tangent) tensors.
    # Kept as plain Python lists (not autograd tensors) so the trajectory stays
    # a pure non-parametric memory record.
    transfer_actions: list = field(default_factory=list)
    transfer_anchors: list = field(default_factory=list)
    transfer_tangents: list = field(default_factory=list)

    def __post_init__(self):
        # Every *tensor* field must be finite; never convert NaN/Inf to zero/null.
        for name, t in self._as_dict().items():
            if isinstance(t, torch.Tensor) and not torch.isfinite(t).all():
                raise ValueError(f"ghost field {name} is non-finite")
        # Sphere + tangent invariants. The tangent must be orthogonal to the
        # anchor for BOTH signs of the dot product.
        if abs(float(self.anchor.norm() - 1.0)) > 1e-3:
            raise ValueError("ghost anchor must lie on the unit sphere")
        if abs(float(self.tangent.norm())) > 1e-3:
            dot = float((self.anchor * self.tangent).sum())
            if abs(dot) > 1e-3:
                raise ValueError(
                    f"ghost tangent must be orthogonal to anchor (dot={dot})"
                )
        evidence_lengths = {
            len(self.transfer_actions),
            len(self.transfer_anchors),
            len(self.transfer_tangents),
        }
        if len(evidence_lengths) != 1:
            raise ValueError("action evidence lists must have equal length")
        for source, action, tangent in zip(
            self.transfer_anchors,
            self.transfer_actions,
            self.transfer_tangents,
            strict=True,
        ):
            if not all(torch.isfinite(x).all() for x in (source, action, tangent)):
                raise ValueError("action evidence must be finite")
            if source.numel() != self.anchor.numel() or tangent.numel() != self.anchor.numel():
                raise ValueError("action evidence latent dimensions must match the ghost")
            if abs(float(source.reshape(-1).norm() - 1.0)) > 1e-3:
                raise ValueError("action evidence source must lie on the unit sphere")
            if abs(float((source.reshape(-1) * tangent.reshape(-1)).sum())) > 1e-3:
                raise ValueError("action evidence tangent must match its source anchor")

    def _as_dict(self):
        return {
            "anchor": self.anchor,
            "tangent": self.tangent,
            "credibility": self.credibility,
            "grounding": self.grounding,
            "uncertainty": self.uncertainty,
            "persistence": self.persistence,
            "evidence_support": self.evidence_support,
            "evidence_negative": self.evidence_negative,
            "boundary_compat": self.boundary_compat,
            "horizon_expr": self.horizon_expr,
            "transfer_actions": self.transfer_actions,
            "transfer_anchors": self.transfer_anchors,
            "transfer_tangents": self.transfer_tangents,
        }

    def to(self, device):
        data = self._as_dict()
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                data[key] = value.to(device)
            elif isinstance(value, list):
                data[key] = [item.to(device) for item in value]
        return GhostTrajectory(**data)

    def predicted_anchor(self, step: float,
                         action: torch.Tensor | None = None,
                         anchor: torch.Tensor | None = None) -> torch.Tensor:
        """Lift a predicted future point on S^(D-1).

        Without an action the signature deformation (fixed observed tangent) is
        used. With an action, the deformation is predicted by the evidence-learned
        action-conditioned transfer map, so the prediction depends on the
        candidate action actually being realized.
        """
        if anchor is None:
            anchor = self.anchor
        anchor = project_to_sphere(anchor.detach().reshape(-1))
        if action is None:
            return exponential_map(anchor, self.tangent * step)
        pred_tan = self.transfer_predict(action, anchor)
        return exponential_map(anchor, pred_tan * step)

    def transfer_predict(self, action: torch.Tensor,
                         anchor: torch.Tensor | None = None) -> torch.Tensor:
        """Action-conditioned deformation via a ridge-regularized linear map.

        Returns a tangent at ``anchor``. With fewer than two stored pairs the
        map is undefined and returns the zero tangent. Integration uses the
        observed signature continuation until this refinement is available.
        """
        if len(self.transfer_tangents) < 2:
            a0 = self.anchor if anchor is None else anchor
            return torch.zeros_like(project_to_sphere(a0.detach().reshape(-1)))
        query_anchor = project_to_sphere(
            (self.anchor if anchor is None else anchor).detach().reshape(-1)
        )
        A = torch.stack([a.detach().reshape(-1) for a in self.transfer_actions])
        T = torch.stack(
            [
                parallel_transport_sphere(
                    source.detach().reshape(-1),
                    query_anchor,
                    tangent.detach().reshape(-1),
                )
                for source, tangent in zip(
                    self.transfer_anchors,
                    self.transfer_tangents,
                    strict=True,
                )
            ]
        )
        # Ridge regression: M = T^T A (A^T A + lam I)^{-1}   (D, A)
        # Scale-derived numerical ridge. Trace(A^T A) is invariant to action-axis
        # permutations, so stabilization introduces no preferred action axis.
        output_dtype = A.dtype
        A = A.double()
        T = T.double()
        gram = A.t() @ A
        lam = (
            torch.finfo(A.dtype).eps ** 0.5
            * gram.trace().clamp_min(torch.finfo(A.dtype).eps)
            / A.shape[1]
        )
        ATA = gram + lam * torch.eye(
            A.shape[1], device=A.device, dtype=A.dtype
        )
        M = torch.linalg.solve(ATA, A.t() @ T).t()  # (D, A)
        action_v = action.detach().reshape(-1).to(device=A.device, dtype=A.dtype)
        pred = (M @ action_v).to(output_dtype)
        return project_to_tangent(query_anchor, pred)

    def add_action_evidence(self, source_anchor: torch.Tensor, action: torch.Tensor,
                            tangent: torch.Tensor) -> "GhostTrajectory":
        """Return a trajectory with one externally observed transformation."""
        data = self._as_dict()
        source = project_to_sphere(
            source_anchor.detach().reshape(-1)
        ).clone()
        data["transfer_actions"] = data["transfer_actions"] + [
            action.detach().reshape(-1).to(
                device=source.device,
                dtype=source.dtype,
            ).clone()
        ]
        data["transfer_anchors"] = data["transfer_anchors"] + [source]
        data["transfer_tangents"] = data["transfer_tangents"] + [
            project_to_tangent(
                source,
                tangent.detach().reshape(-1).to(
                    device=source.device, dtype=source.dtype
                ),
            ).clone()
        ]
        return GhostTrajectory(**data)

    def predicted_points(self, step: float):
        """Return (anchor, predicted_anchor) as the trajectory's situated points."""
        return self.anchor, self.predicted_anchor(step)

    def with_updates(self, **kwargs) -> "GhostTrajectory":
        data = self._as_dict()
        data.update(kwargs)
        return GhostTrajectory(**data)


def make_ghost(
    anchor: torch.Tensor,
    tangent: torch.Tensor,
    credibility: float = 1.0,
    grounding: float = 0.0,
    uncertainty: float = 1.0,
    persistence: float = 0.0,
    evidence_support: float = 0.0,
    evidence_negative: float = 0.0,
    boundary_compat: float = 1.0,
    horizon_expr: float = 1.0,
) -> GhostTrajectory:
    """Construct a ghost with scalar state initialized as detached tensors.

    A ghost born from an externally observed deformation starts with
    grounding = 0 and evidence_support = 0: it has not yet earned any
    external grounding. Its tangent is the observed (role-free) deformation,
    never an invented coordinate direction.
    """
    anchor = project_to_sphere(anchor.detach()).reshape(-1)
    tangent = project_to_tangent(anchor, tangent.detach().reshape(-1))
    return GhostTrajectory(
        anchor=anchor,
        tangent=tangent,
        credibility=anchor.new_tensor(float(credibility)),
        grounding=anchor.new_tensor(float(grounding)),
        uncertainty=anchor.new_tensor(float(uncertainty)),
        persistence=anchor.new_tensor(float(persistence)),
        evidence_support=anchor.new_tensor(float(evidence_support)),
        evidence_negative=anchor.new_tensor(float(evidence_negative)),
        boundary_compat=anchor.new_tensor(float(boundary_compat)),
        horizon_expr=anchor.new_tensor(float(horizon_expr)),
        transfer_actions=[],
        transfer_anchors=[],
        transfer_tangents=[],
    )


def transport_ghost(
    ghost: GhostTrajectory,
    real_prev: torch.Tensor,
    real_now: torch.Tensor,
    step: float,
) -> GhostTrajectory:
    """Persist a ghost across a real latent step.

    The real trajectory is authoritative. The ghost anchor is moved along the
    SAME geodesic displacement the real latent followed, by transporting that
    displacement (a tangent at real_prev) to the ghost's own anchor and applying
    it with the exponential map. The ghost's stored tangent is parallel-transported
    along the ghost's own old-anchor -> new-anchor geodesic. All additions occur
    in the tangent space of their stated anchor; point movement uses exponential
    maps.
    """
    real_prev = project_to_sphere(real_prev.detach().reshape(-1))
    real_now = project_to_sphere(real_now.detach().reshape(-1))
    # The geodesic displacement is undefined when real_prev and real_now are
    # (numerically) antipodal: every great circle through both is valid, so the
    # real displacement — and therefore the ghost's transported displacement —
    # is not unique.
    if antipodal(real_prev, real_now):
        raise ValueError(
            "ghost transport undefined across antipodal real points "
            "(non-unique geodesic displacement)"
        )
    # The exact geodesic displacement of the real trajectory, as a tangent at
    # real_prev. This is the real geodesic displacement, never the
    # Euclidean chord difference, which is not a tangent and is not equivariant.
    real_disp = log_map_sphere(real_prev, real_now)
    # Transport the real displacement to the ghost's anchor, then move the ghost
    # along that same geodesic displacement (exponential map).
    ghost_disp = parallel_transport_sphere(real_prev, ghost.anchor, real_disp)
    new_anchor = exponential_map(ghost.anchor, ghost_disp)
    # Parallel-transport the ghost's signature tangent along its own geodesic.
    new_tangent = parallel_transport_sphere(ghost.anchor, new_anchor, ghost.tangent)
    return ghost.with_updates(anchor=new_anchor, tangent=new_tangent)
