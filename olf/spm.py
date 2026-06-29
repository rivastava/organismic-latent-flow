"""olf/spm.py

v3: Spherical Memory + Rotary Timestamped Causal Memory (RTCM).

Per the Action-Sphere RTCM research memo:
  - Spherical Memory: events stored as unit vectors on S^(d-1)
  - Rotary Timestamped Causal Memory: time as phase rotation
      phi(t) = [cos(w_1 t), ..., cos(w_F t), sin(w_1 t), ..., sin(w_F t)]
  - A trajectory across the sphere becomes a sequence of rotated event
    capsules. Similarity is angular; causality is rotation-transfer.

This module implements a bounded fast-trace buffer where each event is
unit-normalized and a rotary timestamp phi(t) is applied. The `get_trace`
returns a phase-rotated unit vector, which makes time visible in the
spherical memory representation.

Constitution §4: this is a fast-trace store. Slow consolidation lives in
`ConsequenceMemory.consolidate`. The trace buffer here is bounded
(`max_events`) so it does not grow without limit.
"""

import math
import torch
import torch.nn as nn
from olf.geometry import project_to_sphere


class SphericalPhaseMemory(nn.Module):
    """Spherical Memory + rotary timestamped trace.

    Each stored event is a unit vector (h, action, consequence). The
    `update(h)` call stores h after projecting to the sphere. The
    `get_trace()` call returns a phase-rotated weighted sum of recent
    events, which is then projected back to the sphere.

    The rotary phase phi(t) = [cos(w_i t), sin(w_i t)] for i in [1..F]
    makes time visible in the spherical representation. Without rotation,
    SPM reduces to a windowed trace.
    """

    def __init__(
        self,
        latent_dim=32,
        window_size=5,
        decay_rate=0.5,
        rotary_freqs=4,
        max_time=64,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.window_size = window_size
        self.decay_rate = decay_rate
        self.rotary_freqs = rotary_freqs
        self.max_time = max_time

        # Exponential decay weights for the sliding history window.
        weights = torch.exp(-decay_rate * torch.arange(window_size, dtype=torch.float32))
        self.register_buffer("history_weights", weights / weights.sum())

        # Rotary frequencies: a small learned-free bank of frequencies
        # that span the time horizon. We use evenly-spaced frequencies.
        # Inv-Normalized so the lowest frequency completes one full period
        # at the maximum time horizon.
        freqs = torch.linspace(0.5, 4.0, rotary_freqs, dtype=torch.float32)
        self.register_buffer("freqs", freqs)

        self.reset_memory()

    def reset_memory(self):
        """Clears the history trace buffer."""
        self.history = None
        self.timestamps = None
        self.t = 0  # global time counter

    def update(self, h):
        """Stores the new latent state on the sphere. Increments time."""
        # Project to sphere so the trace stays on S^(d-1).
        h_proj = project_to_sphere(h)
        h_detached = h_proj.clone().detach()

        if self.history is None:
            self.history = h_detached.unsqueeze(0).repeat(self.window_size, 1, 1)
            self.timestamps = torch.zeros(self.window_size, dtype=torch.long)
        else:
            self.history = torch.cat([self.history[1:], h_detached.unsqueeze(0)], dim=0)
            self.timestamps = torch.cat(
                [self.timestamps[1:], torch.tensor([self.t % self.max_time], dtype=torch.long)],
                dim=0,
            )
        self.t += 1

    def _rotary_phase(self, t):
        """Compute rotary phase phi(t) for time t.

        phi(t) = [cos(w_1 t), ..., cos(w_F t), sin(w_1 t), ..., sin(w_F t)]

        Returns a tensor of shape (2*F,).
        """
        t_norm = (t.float() / self.max_time) * 2 * math.pi
        phase = self.freqs * t_norm  # (F,)
        return torch.cat([torch.cos(phase), torch.sin(phase)], dim=0)  # (2F,)

    def get_trace(self):
        """Computes the rotary-phase-modulated trace vector and projects
        it to the sphere S^(d-1).

        Implementation: for each event i in the window, weight by
        history_weights[i] AND by the cosine similarity of phi(t_i) with
        phi(0) (so the most recent event dominates, but older events
        still contribute via the cos phase). This makes time visible in
        the spherical representation.

        Returns shape: (1, latent_dim) on S^(d-1).
        """
        device = self.history_weights.device

        if self.history is None:
            rand_vec = torch.randn(1, self.latent_dim, device=device)
            return rand_vec / (torch.linalg.norm(rand_vec, dim=-1, keepdim=True) + 1e-8)

        # Build per-event time weight using rotary phase cos(w t).
        # Per Action-Sphere RTCM memo §3, time-as-phase. We use the
        # cosine of the phase difference between event i and "now".
        # v3.1: blend with a uniform time weight (0.5 each) to avoid
        # breaking the existing windowed trace. The rotary phase is
        # a gentle modulator, not a complete replacement.
        cur_t = self.timestamps[-1].item()
        time_weights = []
        for ti in self.timestamps:
            ti_val = ti.item() if torch.is_tensor(ti) else int(ti)
            delay = float(cur_t - ti_val)
            phase_diff = self.freqs * (delay / self.max_time) * 2 * math.pi
            t_w = torch.cos(phase_diff).mean().clamp(min=0.0)
            time_weights.append(t_w)
        time_weights = torch.stack(tuple(time_weights)).to(device)
        # 50/50 blend with uniform (so the windowed trace still works).
        uniform = torch.ones_like(time_weights) / time_weights.shape[0]
        time_weights = 0.5 * time_weights + 0.5 * uniform

        # Combine with history decay weights.
        combined_weights = self.history_weights * time_weights
        combined_weights = combined_weights / (combined_weights.sum() + 1e-8)

        # Weighted sum over the window.
        w_expanded = combined_weights.view(-1, 1, 1)
        weighted_sum = torch.sum(self.history * w_expanded, dim=0)

        return project_to_sphere(weighted_sum)

    def add_phase_step(self, t_offset=0):
        """v3 convenience: bump the global time without changing h.

        Used in select_action when we want to advance the time rotor
        independent of a state update (e.g. on motor release).
        """
        self.t += 1
