"""Observable predictive evidence for ghost trajectories.

All evidence is derived from *external* observation after recoupling. Internal
rollout / rehearsal can never increase grounding or evidence support.

Grounding is CONTRASTIVE: a ghost earns grounding only when it predicts the
recoupled observation better than the organism's own passive / current-flow
baseline (the FLC base future latent).
Predicting worse than the baseline never increases grounding and reduces
credibility. A maximally wrong ghost therefore gains no grounding.
"""

import torch

from olf.geometry import (
    angular_distance,
    log_map_sphere,
    project_to_sphere,
    project_to_tangent,
)

from .diagnostics import finite_or_raise
from .trajectory import GhostTrajectory


def predicted_anchor(
    ghost: GhostTrajectory,
    step: float,
    action: torch.Tensor | None = None,
    anchor: torch.Tensor | None = None,
) -> torch.Tensor:
    return ghost.predicted_anchor(step, action=action, anchor=anchor)


def predictive_error(ghost: GhostTrajectory, observed_anchor: torch.Tensor,
                     step: float, action: torch.Tensor | None = None,
                     anchor: torch.Tensor | None = None) -> torch.Tensor:
    """Angular distance between the ghost's predicted anchor and the observed one.

    Returns a scalar tensor. Larger = the ghost was more wrong. No hidden label.
    """
    pred = predicted_anchor(ghost, step, action=action, anchor=anchor)
    err = angular_distance(pred, observed_anchor.reshape(-1))
    return finite_or_raise("predictive_error", err)


def baseline_error(base_future_anchor: torch.Tensor,
                   observed_anchor: torch.Tensor) -> torch.Tensor:
    """Angular error of the organism's passive / current-flow baseline."""
    err = angular_distance(base_future_anchor.reshape(-1), observed_anchor.reshape(-1))
    return finite_or_raise("baseline_error", err)


def observed_deformation(real_prev: torch.Tensor, observed_anchor: torch.Tensor
                         ) -> torch.Tensor:
    """Log-map deformation of the real trajectory (tangent at real_prev)."""
    return finite_or_raise(
        "observed_deformation",
        project_to_tangent(
            project_to_sphere(real_prev.reshape(-1)),
            log_map_sphere(project_to_sphere(real_prev.reshape(-1)),
                           project_to_sphere(observed_anchor.reshape(-1))),
        ),
    )


def update_after_recoupling(
    ghost: GhostTrajectory,
    observed_anchor: torch.Tensor,
    step: float,
    baseline_err: torch.Tensor,
    learning_rate: float = 0.1,
    prediction_error: torch.Tensor | None = None,
) -> GhostTrajectory:
    """Update role-free state from an *external* observation only.

    Contrastive evidence against the organism's passive baseline:
      advantage = baseline_err - err
      positive  = relu(advantage)       (ghost predicts better than baseline)
      negative  = relu(-advantage)      (baseline predicts better than ghost)

    Grounding rises ONLY with positive evidence. Credibility is the fraction
    of accumulated contrastive evidence that is positive. Internal rollout
    can never call this; grounding/evidence are manufactured only here.
    """
    err = (
        predictive_error(ghost, observed_anchor, step)
        if prediction_error is None
        else prediction_error
    )
    advantage = baseline_err - err
    positive = torch.relu(advantage)
    negative = torch.relu(-advantage)

    new_ground = torch.clamp(ghost.grounding + learning_rate * positive, 0.0, 1.0)
    new_unc = torch.clamp(
        ghost.uncertainty * (1.0 - learning_rate) + err * learning_rate, 0.0, 1e6
    )
    new_pos = ghost.evidence_support + positive
    new_neg = ghost.evidence_negative + negative
    evidence_total = new_pos + new_neg
    new_cred = torch.where(
        evidence_total > torch.finfo(evidence_total.dtype).eps,
        new_pos / evidence_total,
        ghost.credibility,
    )
    new_persist = ghost.persistence + 1.0
    return ghost.with_updates(
        credibility=new_cred,
        grounding=new_ground,
        uncertainty=new_unc,
        evidence_support=new_pos,
        evidence_negative=new_neg,
        persistence=new_persist,
    )


def internal_rehearsal_update(ghost: GhostTrajectory) -> GhostTrajectory:
    """Advance persistence WITHOUT touching grounding or evidence support.

    Internal rollout is allowed to age a ghost (so dormant ones can be evicted)
    but must NEVER manufacture evidence. Grounding, evidence_support, and
    evidence_negative are held fixed; this is asserted by the caller/tests.
    """
    return ghost.with_updates(persistence=ghost.persistence + 1.0)
