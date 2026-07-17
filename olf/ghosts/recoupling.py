"""External recoupling and the learned reachable-deformation space.

Recoupling is mandatory before the ghost population may influence another
action. It updates the real OLF flow first (handled by the caller), then
compares observed deformation with ghost predictions and only then updates
credibility/grounding and runs the lifecycle.

Reachability ownership rule: every prototype carries its SOURCE anchor,
the RELEASED ACTION that produced it, and the observed tangent deformation. A
candidate ghost is judged reachable only against prototypes whose action is
*compatible* with the candidate action (after transporting all tangents to the
current anchor). A span of unrelated past actions cannot make an incompatible
action claim the same reachable deformation. An empty buffer is reported as
unknown. Reachability is diagnostic and does not determine participation.
"""

from dataclasses import dataclass, field

import torch

from olf.geometry import parallel_transport_sphere, project_to_tangent

from .config import GhostConfig
from .evidence import (
    baseline_error,
    observed_deformation,
    predictive_error,
    update_after_recoupling,
)
from .lifecycle import evict, maybe_birth, merge_similar, LifecycleReport
from .population import GhostPopulation


class ReachabilityBuffer:
    """Low-rank, action-conditioned union of *observed* reachable deformations.

    Each prototype is a (source_anchor, released_action, tangent) triple. No
    arbitrary attractor is injected here; prototypes come solely from
    external recoupling.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.anchors: list[torch.Tensor] = []
        self.actions: list[torch.Tensor] = []
        self.prototypes: list[torch.Tensor] = []

    def add(self, anchor: torch.Tensor, action: torch.Tensor,
            tangent: torch.Tensor) -> None:
        if self.capacity <= 0:
            return
        self.anchors.append(anchor.detach().reshape(-1).clone())
        self.actions.append(
            action.detach().reshape(-1).to(
                device=anchor.device, dtype=anchor.dtype
            ).clone()
        )
        self.prototypes.append(
            tangent.detach().reshape(-1).to(
                device=anchor.device, dtype=anchor.dtype
            ).clone()
        )
        if len(self.prototypes) > self.capacity:
            self.anchors.pop(0)
            self.actions.pop(0)
            self.prototypes.pop(0)

    def empty(self) -> bool:
        return len(self.prototypes) == 0

    def _compatible_mask(self, action: torch.Tensor) -> torch.Tensor:
        """Boolean mask of prototypes whose action is compatible with ``action``.

        Compatibility is cosine similarity of the released actions; an action
        that is irrelevant (cosine below threshold) cannot lend its deformation
        to a different candidate action.
        """
        action = action.detach().reshape(-1)
        A = torch.stack(self.actions)  # (M, A)
        action = action.to(device=A.device, dtype=A.dtype)
        sim = torch.nn.functional.cosine_similarity(
            A, action.unsqueeze(0), dim=-1
        )
        # Directional compatibility is geometric: non-positive coupling cannot
        # support the queried action. Among positively coupled observations,
        # retain the strongest currently available evidence without a tuned
        # similarity cutoff.
        positive = sim > 0.0
        if not bool(positive.any()):
            return positive
        maximum = sim[positive].max()
        return positive & torch.isclose(sim, maximum, rtol=1e-5, atol=1e-7)

    def residual(self, tangent: torch.Tensor, action: torch.Tensor,
                 current_anchor: torch.Tensor) -> float | None:
        """Normalized orthogonal residual of ``tangent`` w.r.t. the action-compatible
        learned span.

        Prototypes are first parallel-transported to ``current_anchor`` so the
        span is built in a single tangent space, and only action-compatible
        prototypes contribute. Returns None when the buffer is empty or no
        action-compatible prototype exists (unknown reachability).
        """
        t = project_to_tangent(current_anchor.reshape(-1), tangent.detach().reshape(-1))
        if not self.prototypes:
            return None
        mask = self._compatible_mask(action)
        if not bool(mask.any()):
            return None  # incompatible / unknown action coverage
        current_anchor = current_anchor.detach().reshape(-1)
        transported = [
            parallel_transport_sphere(a.detach(), current_anchor, p.detach())
            for a, p in zip(self.anchors, self.prototypes, strict=True)
        ]
        P = torch.stack(transported, dim=0)  # (M, D)
        P = P[mask]
        PPt = P @ P.t()
        Pt = P @ t
        try:
            c = torch.linalg.solve(
                PPt
                + 1e-6
                * torch.eye(PPt.shape[0], device=PPt.device, dtype=PPt.dtype),
                Pt,
            )
        except RuntimeError:
            c = torch.linalg.lstsq(PPt, Pt).solution
        proj = P.t() @ c
        res = t - proj
        norm_res = float(torch.norm(res))
        norm_t = float(torch.norm(t))
        return norm_res / max(norm_t, 1e-8)

    def as_tensor(self) -> torch.Tensor:
        if not self.prototypes:
            return torch.empty(0)
        return torch.stack(self.prototypes, dim=0)


@dataclass
class RecoupleReport:
    per_ghost_error: list[float]
    grounding_updated: bool
    born: bool
    merged: bool
    evicted_count: int
    reachability_prototypes: int
    lifecycle_reasons: list[str] = field(default_factory=list)


def recouple(
    population: GhostPopulation,
    real_prev: torch.Tensor,
    observed_anchor: torch.Tensor,
    base_future_anchor: torch.Tensor,
    config: GhostConfig,
    buffer: ReachabilityBuffer,
    released_action: torch.Tensor | None = None,
) -> tuple:
    """Apply one mandatory recoupling and return (new_population, buffer, report)."""
    real_prev = project_to_sphere_input(real_prev)
    observed_anchor = project_to_sphere_input(observed_anchor)
    base_future_anchor = project_to_sphere_input(base_future_anchor)

    # 1. Record the *observed* reachable deformation (external evidence only),
    #    tagged with its source anchor and the released action that produced it.
    obs_def = observed_deformation(real_prev, observed_anchor)
    if released_action is not None:
        buffer.add(real_prev, released_action, obs_def)

    # 2. Contrastive evidence update for each ghost (vs the passive baseline).
    b_err = baseline_error(base_future_anchor, observed_anchor)
    updated = []
    per_ghost_error = []
    for g in population._ghosts:
        err = float(predictive_error(g, observed_anchor, config.transport_step))
        per_ghost_error.append(err)
        updated.append(
            update_after_recoupling(g, observed_anchor, config.transport_step, b_err)
        )

    pop_after = GhostPopulation.empty(config.latent_dim, config.effective_capacity)
    pop_after._ghosts = updated

    # 3. Lifecycle from the observation only (online-statistic rules + diagnostics).
    report = LifecycleReport()
    before = len(pop_after)
    pop_after = maybe_birth(pop_after, real_prev, observed_anchor, config, report)
    born = len(pop_after) != before

    before_m = len(pop_after)
    pop_after = merge_similar(pop_after, config, report)
    merged = len(pop_after) != before_m

    before_e = len(pop_after)
    pop_after = evict(pop_after, config, report)
    evicted = before_e - len(pop_after)

    # 4. Transformation evidence: attach (released_action, observed
    #    tangent) to the ghost whose signature prediction best matched the
    #    observed deformation. This is the ONLY place a ghost learns the
    #    action->deformation relation; it is purely external recoupling.
    if released_action is not None and len(pop_after._ghosts) > 0:
        errs = [
            float(predictive_error(g, observed_anchor, config.transport_step))
            for g in pop_after._ghosts
        ]
        best = int(torch.tensor(errs).argmin())
        g = pop_after._ghosts[best]
        pop_after._ghosts[best] = g.add_action_evidence(
            real_prev, released_action, obs_def
        )

    rec_report = RecoupleReport(
        per_ghost_error=per_ghost_error,
        grounding_updated=True,
        born=born,
        merged=merged,
        evicted_count=evicted,
        reachability_prototypes=len(buffer.prototypes),
        lifecycle_reasons=report.reasons,
    )
    return pop_after, buffer, rec_report


def project_to_sphere_input(t):
    from olf.geometry import project_to_sphere
    return project_to_sphere(t.detach().reshape(-1))


def measure_reachability(tangent: torch.Tensor, action: torch.Tensor,
                         current_anchor: torch.Tensor,
                         buffer: ReachabilityBuffer, threshold: float) -> tuple:
    """Return (reachable: bool, residual: float | None).

    Empty buffer or no action-compatible prototype => UNKNOWN => not reachable.
    Otherwise reachable when the normalized residual is within threshold.
    """
    res = buffer.residual(tangent, action, current_anchor)
    if res is None:
        return False, None
    reachable = res <= threshold
    return reachable, res
