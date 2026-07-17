"""olf/salience.py

Prospective Salience Gate.

Per the RTCM design: "Retrospective salience marks
an event important after its outcome is known. Prospective salience estimates
salience: the estimated future causal value of the event at the moment it
occurs."

This module computes a scalar salience s_t = P(event_t becomes causally
useful later | event_t, context_t, goal_t) for each event, and decides
which events to promote into long-term memory.

Inputs to the salience:
    - event vector (h_t)
    - novelty (distance from recent trace)
    - uncertainty (predictor's confidence)
    - predicted value (semantics.value head)
    - state change magnitude (h_{t+1} − h_t)
    - surprise (prediction error at t)

Output:
    - salience in [0, 1] per event
    - policy: write if salience > threshold, else skip

The gate is constitutional because it does not add a new "hard" module
— it is a small learned head that shapes the WRITE policy, not the
read or act policy. Per memo: "write more aggressively when future
causal value is high, not merely when the current event is loud or
recent."
"""

import torch
import torch.nn as nn


class ProspectiveSalienceGate(nn.Module):
    """Estimates the prospective causal value of an event and decides
    whether to write it to long-term memory.

    Salience inputs (concatenated into a feature vector):
        - h_t (latent_dim)
        - novelty = ||h_t - trace_mean|| (1)
        - uncertainty = predictor's reported uncertainty (1)
        - predicted_value = semantics.value head (1)
        - state_change_magnitude = ||h_{t+1} - h_t|| (1)
        - surprise = 1 - predictor confidence (1)
    """

    def __init__(self, latent_dim=32, hidden_dim=32):
        super().__init__()
        self.latent_dim = latent_dim
        # Input: latent_dim (h_t) + 5 scalar features
        in_dim = latent_dim + 5
        self.salience_net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        # Initialize to output ~0.5 so the gate starts permissive
        with torch.no_grad():
            self.salience_net[-1].bias.fill_(0.0)

    def compute_salience(
        self,
        h_t,
        novelty=0.0,
        uncertainty=0.0,
        predicted_value=0.0,
        state_change_magnitude=0.0,
        surprise=0.0,
    ):
        """Returns salience in [0, 1]."""
        if h_t.dim() == 1:
            h_t = h_t.unsqueeze(0)
        feats = torch.tensor(
            [[novelty, uncertainty, predicted_value, state_change_magnitude, surprise]],
            dtype=h_t.dtype,
            device=h_t.device,
        )
        inp = torch.cat([h_t, feats], dim=-1)
        logit = self.salience_net(inp).squeeze(-1)
        return torch.sigmoid(logit)

    def should_write(self, salience, threshold=0.3):
        """Hard policy: write if salience > threshold. Soft version:
        use salience as a write probability.

        The threshold can decay over training (curriculum: write more
        early when memory is sparse, write less later when full).
        """
        return salience > threshold
