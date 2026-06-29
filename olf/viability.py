"""olf/viability.py

Constitutional viability-constrained policy.

Per Constitution §6: "Goals Are Attractors, Not Commands."
Per Constitution §7: "Terminal Boundaries Are Not Rewards."

A goal is a preferred region in latent dynamics. The terminal boundary
is a viability constraint, not a negative reward. Together, they imply
a policy trained to MAINTAIN viability, not maximize scalar reward.

This module provides:
  - ViabilityPredictor: a small head on the latent state h that predicts
    whether the organism will remain viable in the near future.
  - ViabilityConstrainedPolicy: a policy that selects actions to keep
    the predicted viability high. This is constitutional — it does not
    use external reward, only the organism's own boundary prediction.

The policy objective:
  maximize E[ViabilityPredictor(h_{t+1})] under the learned dynamics
  subject to: a_t in safe set defined by VetoBoundary

This replaces the "policy = maximize external reward" framing with
"policy = stay alive" — which is what a real organism does.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ViabilityPredictor(nn.Module):
    """Predicts the organism's near-future viability from its current
    latent state h.

    Viability v_t ∈ [0, 1] is the probability that the organism will
    remain viable (not terminal) for the next N steps.

    Training target: v_t = 1.0 if episode ended in success, 0.0 if
    death/starvation, 0.5 if timeout. The predictor learns to forecast
    the outcome from the current state.
    """

    def __init__(self, latent_dim=32, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        # Optimistic initialization: viability starts at 0.8.
        with torch.no_grad():
            self.net[-1].bias.fill_(1.0)

    def forward(self, h):
        """Returns v_t in [0, 1]. Input: (batch, latent_dim)."""
        if h.dim() == 1:
            h = h.unsqueeze(0)
        return torch.sigmoid(self.net(h).squeeze(-1))


class ViabilityConstrainedPolicy(nn.Module):
    """Constitutional policy: select actions that keep viability high.

    This is a head on (h, sigma) that outputs a 3-dim action. It is
    trained to maximize ViabilityPredictor(h_{t+1}) rather than
    external reward. Per Constitution §6, the goal is an attractor in
    latent dynamics — keeping h in a viable region.

    Crucially, this DOES NOT use external reward. The training signal
    is the organism's own viability prediction. The boundary is the
    hard viability gate (VetoBoundary) which is a separate mechanism
    per §7.
    """

    def __init__(self, latent_dim=32, hidden_dim=64, num_entities=2,
                 viability_predictor=None, action_dim=3):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_entities = num_entities
        self.action_dim = action_dim

        in_dim = latent_dim + hidden_dim * num_entities
        self.policy_head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )

        # If a viability predictor is given, use it for self-supervised
        # training. Otherwise instantiate a new one.
        self.viability_predictor = viability_predictor or ViabilityPredictor(
            latent_dim=latent_dim
        )

    def forward(self, h, sigma_t):
        """Returns the action proposal.

        Args:
            h: latent state (1, latent_dim)
            sigma_t: situated binding (1, num_entities, hidden_dim)

        Returns:
            action: (1, action_dim) in [-1, 1]
        """
        if h.dim() == 1:
            h = h.unsqueeze(0)
        if sigma_t.dim() == 3:
            flat = sigma_t.reshape(sigma_t.size(0), -1)
        else:
            flat = sigma_t
        inp = torch.cat([h, flat], dim=-1)
        return self.policy_head(inp)


class ViabilityTrainingLoss(nn.Module):
    """Loss for the viability-constrained policy.

    L = -E[log π(a_t | s_t) * adv_t]
        + α * BCE(v_pred(h_t), v_target_t)
        + β * ||a_t||^2  (regularize action magnitude)

    Where adv_t = v_pred(h_{t+1}) - v_pred(h_t)  (viability improvement).
    This is a self-supervised advantage that does NOT use external reward.
    """

    def __init__(self, alpha=0.1, beta=0.001):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, log_prob, viability_pred_next, viability_pred_now,
                viability_target, action_taken):
        """Compute the constitutional policy loss.

        Args:
            log_prob: (T,) log probability of taken action at each step
            viability_pred_next: (T,) predicted viability at h_{t+1}
            viability_pred_now: (T,) predicted viability at h_t
            viability_target: (T,) target viability (1=success, 0=death, 0.5=timeout)
            action_taken: (T, action_dim) the actions that were taken

        Returns:
            scalar loss
        """
        # Self-supervised advantage: predicted improvement in viability.
        adv = viability_pred_next - viability_pred_now  # (T,)
        policy_loss = -(log_prob * adv.detach()).mean()

        # Viability prediction loss (BCE).
        viability_loss = F.binary_cross_entropy(
            viability_pred_next.clamp(1e-6, 1 - 1e-6),
            viability_target,
        )

        # Action magnitude regularization.
        action_reg = (action_taken ** 2).mean()

        return policy_loss + self.alpha * viability_loss + self.beta * action_reg
