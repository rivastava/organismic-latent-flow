"""Latent transfer fields for future-latent control."""

import torch
import torch.nn as nn

from olf.geometry import log_map_sphere, project_to_tangent


class InverseTransferField(nn.Module):
    """Learnable inverse transfer from future latent error to present correction.

    FLC uses the same RTCM-shaped idea as causal recall:

        future_effect ~= R_delta present_correction
        present_correction ~= R_delta.T future_effect

    This module keeps that operation explicit. The correction lives in the
    tangent space at the current latent state.
    """

    def __init__(self, latent_dim=32):
        super().__init__()
        self.latent_dim = latent_dim
        self.R_delta = nn.Linear(latent_dim, latent_dim, bias=False)
        with torch.no_grad():
            self.R_delta.weight.copy_(torch.eye(latent_dim))

    def inverse_correction(self, current_latent, future_latent):
        """Map a future latent target into a present tangent correction."""
        future_error = log_map_sphere(current_latent, future_latent)
        correction = torch.matmul(future_error, self.R_delta.weight)
        return project_to_tangent(current_latent, correction)

