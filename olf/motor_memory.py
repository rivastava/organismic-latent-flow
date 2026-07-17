"""olf/motor_memory.py

Motor transformation memory: stores (before_h, action, after_h, success, reward)
with explicit delta_h = after_h - before_h so the InventionGenerator can
compose transformations, not just replay old actions.

Constitutional role:
    action meaning = predicted transformation of future flow

The organism must learn that actions are world-transformations, not output
classes. This module records the (state, action, next_state) triple, with a
success/reward label so the InventionGenerator can suppress failed
transformations and prefer successful ones.
"""

import torch
import torch.nn as nn
from olf.geometry import log_map_sphere


class MotorMemory(nn.Module):
    """Bounded memory of (before_h, action, after_h, success, reward) tuples
    with explicit delta_h = after_h − before_h.

    Used by InventionGenerator under impasse to compose candidate actions
    from previously-observed successful (or failed) transformations.
    """

    def __init__(self, latent_dim=32, action_dim=3, max_traces=200):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.max_traces = max_traces

        self.register_buffer("trace_before", torch.zeros(max_traces, latent_dim))
        self.register_buffer("trace_actions", torch.zeros(max_traces, action_dim))
        self.register_buffer("trace_after", torch.zeros(max_traces, latent_dim))
        self.register_buffer("trace_delta", torch.zeros(max_traces, latent_dim))
        self.register_buffer("trace_success", torch.zeros(max_traces))
        self.register_buffer("trace_reward", torch.zeros(max_traces))
        self.register_buffer("trace_active", torch.zeros(max_traces))
        self.write_idx = 0

    def add_transformation(self, before_h, action, after_h, success, reward):
        """Record a transformation (one per step or one per action composite).

        Args:
            before_h: (latent_dim,) or (1, latent_dim) — latent state before action.
            action:    (action_dim,) or (1, action_dim) — action taken.
            after_h:  (latent_dim,) or (1, latent_dim) — latent state after.
            success:  float in {0.0, 1.0} — did the agent survive / succeed?
            reward:   float — episode-end reward (or per-step reward).
        """
        device = self.trace_before.device

        b = before_h.detach().to(device).reshape(-1)
        a = torch.as_tensor(action, dtype=torch.float32, device=device).reshape(-1)
        af = after_h.detach().to(device).reshape(-1)

        if b.shape[0] != self.latent_dim:
            pad = torch.zeros(self.latent_dim, device=device)
            pad[: min(self.latent_dim, b.shape[0])] = b[: min(self.latent_dim, b.shape[0])]
            b = pad
        if a.shape[0] != self.action_dim:
            pad = torch.zeros(self.action_dim, device=device)
            pad[: min(self.action_dim, a.shape[0])] = a[: min(self.action_dim, a.shape[0])]
            a = pad
        if af.shape[0] != self.latent_dim:
            pad = torch.zeros(self.latent_dim, device=device)
            pad[: min(self.latent_dim, af.shape[0])] = af[: min(self.latent_dim, af.shape[0])]
            af = pad

        idx = self.write_idx
        self.trace_before[idx] = b
        self.trace_actions[idx] = a
        self.trace_after[idx] = af
        # tangent-space velocity via log_map instead of Euclidean delta.
        # Euclidean delta = af - b vanishes on the sphere (compact, tiny increments).
        # log_map gives the actual geodesic direction and distance on S^{d-1}.
        self.trace_delta[idx] = log_map_sphere(b.unsqueeze(0), af.unsqueeze(0)).squeeze(0)
        self.trace_success[idx] = float(success)
        self.trace_reward[idx] = float(reward)
        self.trace_active[idx] = 1.0
        self.write_idx = (self.write_idx + 1) % self.max_traces

    def query_similar_action(self, h_query, k=3, return_delta=True):
        """Find past actions whose before_h is similar to h_query and whose
        outcome was successful.

        Returns (actions, deltas, scores) tensors (k entries each), or None
        if the buffer is empty.

        amendments:
            - score blends similarity + reward (not success * 2 - 0.5 alone,
              which was not strong enough)
            - failed transformations (success < 0.5) are masked with -1.0 so
              the organism does not repeatedly invent from failed tries.

        appends a per-call telemetry record to self.query_log
        (a list) for the diagnostic report. Telemetry is collected only
        when the calling organism has diag_mode = True; the call site
        sets self.diag_log_target before calling.
        """
        self._query_count = getattr(self, "_query_count", 0) + 1
        active = self.trace_active > 0.5
        if not active.any():
            return None

        before = self.trace_before[active]
        actions = self.trace_actions[active]
        deltas = self.trace_delta[active]
        success = self.trace_success[active]
        reward = self.trace_reward[active]

        h = h_query.detach().reshape(1, -1).to(before.device)
        if h.shape[-1] != self.latent_dim:
            pad = torch.zeros(1, self.latent_dim, device=before.device)
            pad[..., : min(self.latent_dim, h.shape[-1])] = h[
                ..., : min(self.latent_dim, h.shape[-1])
            ]
            h = pad

        h_n = h / (h.norm() + 1e-8)
        b_n = before / (before.norm(dim=-1, keepdim=True) + 1e-8)
        sim = (b_n * h_n).sum(dim=-1)

        # amendment: reward-blended score; failed transformations are
        # suppressed by subtracting 1.0.
        score = sim + 0.5 * reward
        # masked_fill needs a 0-dim value tensor; pass -1.0 directly.
        score = torch.where(success < 0.5, score - 1.0, score)

        k = min(k, len(score))
        vals, idx = torch.topk(score, k)
        # diagnostic telemetry. Per-call log if the calling
        # organism has set diag_log_target = self. Logging only.
        if getattr(self, "diag_log_target", None) is not None:
            self.diag_log_target.append({
                "n_active": int(self.trace_active.sum().item()),
                "top_sim": float(vals[0].item()) if len(vals) > 0 else 0.0,
                "top_score": float(vals[0].item()) if len(vals) > 0 else 0.0,
                "top_success": float(success[idx[0]].item()) if len(idx) > 0 else 0.0,
                "action_norm_top": float(actions[idx[0]].detach().norm().item()) if len(idx) > 0 else 0.0,
                "delta_norm_top": float(deltas[idx[0]].detach().norm().item()) if len(idx) > 0 else 0.0,
            })
        if return_delta:
            return actions[idx], deltas[idx], vals
        return actions[idx], vals

    def size(self) -> int:
        return int(self.trace_active.sum().item())


class _EmptyMotorMemory(nn.Module):
    """Drop-in stub for the no_motor_memory ablation."""

    def __init__(self, latent_dim=32, action_dim=3):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.register_buffer("trace_active", torch.zeros(1))

    def add_transformation(self, *args, **kwargs):
        return None

    def query_similar_action(self, *args, **kwargs):
        return None

    def size(self) -> int:
        return 0
