"""olf/consequence_memory.py

Bounded fast trace memory + slow flow-field consolidation.

"Slow learning may deform the flow field, but fast trace
memory must still exist. Do not remove geometric memory too early."

Two memory layers co-exist here:
  1. **Fast trace memory** — a circular buffer of (σ, a, s_t, s_{t+1}, c)
     tuples with Gaussian-kernel nearest-neighbor retrieval. Used by
     InventionGenerator for hallucinated rollout validation.
  2. **Slow consolidation path** — `consolidate(flow_net, lr)` samples from
     the trace buffer and performs a small gradient step on the flow field
     so the LTC proposal better predicts observed (s_t, a_t) → s_{t+1}
     transitions. The lr is intentionally tiny (≪ main learning rate) so
     that fast trace memory remains the dominant short-term store.
"""

import torch
import torch.nn as nn


class ConsequenceMemory(nn.Module):
    """Bounded fast trace memory + slow flow consolidation hook."""

    def __init__(self, trace_dim=64, action_dim=3, latent_dim=32, max_traces=1000):
        super().__init__()
        self.trace_dim = trace_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.max_traces = max_traces

        # Fast trace buffers
        self.register_buffer("trace_sigmas", torch.zeros(max_traces, trace_dim))
        self.register_buffer("trace_actions", torch.zeros(max_traces, action_dim))
        self.register_buffer("trace_s_before", torch.zeros(max_traces, latent_dim))
        self.register_buffer("trace_s_after", torch.zeros(max_traces, latent_dim))
        self.register_buffer("trace_consequences", torch.zeros(max_traces, 4))
        self.register_buffer("trace_active", torch.zeros(max_traces))

        self.write_idx = 0

    def add_trace(self, sigma, action, s_before, consequence, s_after=None):
        """Stores a fast trace transition tuple.

        Args:
            sigma:        situated binding (1, trace_dim)
            action:       action taken (action_dim,)
            s_before:     latent state BEFORE the action (1, latent_dim)
            consequence:  [reward, was_lethal, hunger_delta, fatigue_delta] (4,)
            s_after:      latent state AFTER the action (1, latent_dim), optional.
                          When provided, enables slow consolidation.
        """
        device = self.trace_sigmas.device

        s_val = sigma.clone().detach().to(device).reshape(1, -1)
        if s_val.shape[-1] != self.trace_dim:
            # truncate / pad to fit
            flat = torch.zeros(1, self.trace_dim, device=device)
            flat[..., : min(self.trace_dim, s_val.shape[-1])] = s_val[
                ..., : min(self.trace_dim, s_val.shape[-1])
            ]
            s_val = flat
        a_val = (
            torch.as_tensor(action, dtype=torch.float32, device=device)
            .reshape(-1)
        )
        if a_val.shape[0] != self.action_dim:
            pad = torch.zeros(self.action_dim, device=device)
            pad[: min(self.action_dim, a_val.shape[0])] = a_val[
                : min(self.action_dim, a_val.shape[0])
            ]
            a_val = pad
        sb_val = s_before.clone().detach().to(device).reshape(1, -1)
        if sb_val.shape[-1] != self.latent_dim:
            sb_val = sb_val[..., : self.latent_dim]
        c_val = (
            torch.as_tensor(consequence, dtype=torch.float32, device=device)
            .reshape(-1)
        )
        if c_val.shape[0] != 4:
            pad = torch.zeros(4, device=device)
            pad[: min(4, c_val.shape[0])] = c_val[: min(4, c_val.shape[0])]
            c_val = pad
        if s_after is not None:
            sa_val = s_after.clone().detach().to(device).reshape(1, -1)
            if sa_val.shape[-1] != self.latent_dim:
                sa_val = sa_val[..., : self.latent_dim]
        else:
            sa_val = torch.zeros(1, self.latent_dim, device=device)

        idx = self.write_idx
        self.trace_sigmas[idx] = s_val.squeeze(0)
        self.trace_actions[idx] = a_val
        self.trace_s_before[idx] = sb_val.squeeze(0)
        self.trace_s_after[idx] = sa_val.squeeze(0)
        self.trace_consequences[idx] = c_val
        self.trace_active[idx] = 1.0

        self.write_idx = (self.write_idx + 1) % self.max_traces

    def retrieve_trace(self, sigma_query, action_query, k=5, sigma_kernel=0.5):
        """Smooth retrieval over nearby trajectories via Gaussian similarity."""
        device = self.trace_sigmas.device
        active_mask = self.trace_active > 0.5

        if not active_mask.any():
            return (
                torch.zeros(1, self.latent_dim, device=device),
                torch.zeros(1, 4, device=device),
                0.0,
            )

        sigmas = self.trace_sigmas[active_mask]
        actions = self.trace_actions[active_mask]
        s_after = self.trace_s_after[active_mask]
        consequences = self.trace_consequences[active_mask]

        sq = sigma_query.clone().detach().to(device).reshape(1, -1)
        if sq.shape[-1] != sigmas.shape[-1]:
            pad = torch.zeros(1, sigmas.shape[-1], device=device)
            pad[..., : min(sigmas.shape[-1], sq.shape[-1])] = sq[
                ..., : min(sigmas.shape[-1], sq.shape[-1])
            ]
            sq = pad
        aq = action_query.clone().detach().to(device).reshape(1, -1)
        if aq.shape[-1] != actions.shape[-1]:
            pad = torch.zeros(1, actions.shape[-1], device=device)
            pad[..., : min(actions.shape[-1], aq.shape[-1])] = aq[
                ..., : min(actions.shape[-1], aq.shape[-1])
            ]
            aq = pad

        dist_sigma = torch.sum((sigmas - sq) ** 2, dim=-1)
        dist_action = torch.sum((actions - aq) ** 2, dim=-1)
        total_dist = dist_sigma + dist_action
        similarity = torch.exp(-total_dist / (2.0 * sigma_kernel * sigma_kernel))

        vals, indices = torch.topk(similarity, k=min(k, len(similarity)))
        sum_sim = torch.sum(vals)
        if sum_sim > 1e-6:
            weights = vals / sum_sim
            w_exp = weights.view(-1, 1)
            pred_next_h = torch.sum(s_after[indices] * w_exp, dim=0, keepdim=True)
            pred_cons = torch.sum(consequences[indices] * w_exp, dim=0, keepdim=True)
            confidence = sum_sim.item() / k
        else:
            pred_next_h = torch.zeros(1, self.latent_dim, device=device)
            pred_cons = torch.zeros(1, 4, device=device)
            confidence = 0.0

        return pred_next_h, pred_cons, confidence

    def consolidate(self, flow_net, lr=1e-4, n_samples=16, min_buffer=32):
        """Slow plasticity step that deforms the flow field parameters.

        Samples (s_t, a_t, s_{t+1}) tuples from the trace buffer and runs a
        small gradient descent step on the flow net's parameters so that
        its predicted next-state delta (s_{t+1} − s_t) matches the
        observed one. This is the slow consolidation path.

        Returns the loss tensor or None if the buffer is too small.
        """
        device = self.trace_sigmas.device
        active = self.trace_active > 0.5
        n_active = int(active.sum().item())
        if n_active < min_buffer:
            return None

        n_samples = min(n_samples, n_active)
        idx = torch.randint(0, n_active, (n_samples,), device=device)
        active_indices = torch.nonzero(active, as_tuple=False).squeeze(-1)
        chosen = active_indices[idx]

        s_before = self.trace_s_before[chosen]
        s_after = self.trace_s_after[chosen]

        # Fast trace memory excludes observations and uses a
        # "neutral" zero-observation stand-in sized to whatever the flow
        # net's input dimension expects. The flow net still receives s_t as
        # the prior state, so the gradient is on the s_t → s_{t+1} mapping.
        flow_input_dim = None
        for module in flow_net.modules():
            if isinstance(module, nn.Linear):
                # Pick the LARGEST in_features (the input layer) to recover
                # the original input size.
                if flow_input_dim is None or module.in_features > flow_input_dim:
                    flow_input_dim = module.in_features
        if flow_input_dim is None:
            return None

        latent_dim = s_before.shape[-1]
        obs_dim = flow_input_dim - latent_dim
        if obs_dim < 0:
            return None
        zero_obs = torch.zeros(s_before.shape[0], obs_dim, device=device)
        x_flow = torch.cat([zero_obs, s_before], dim=-1)

        # Predict s_{t+1} given s_t via flow net. Loss is MSE to actual s_{t+1}.
        h_pred = flow_net(x_flow, s_before)
        target = s_after
        loss = torch.nn.functional.mse_loss(h_pred, target)

        if not loss.requires_grad:
            return loss.detach()

        params = [p for p in flow_net.parameters() if p.requires_grad]
        grads = torch.autograd.grad(loss, params, allow_unused=True)
        with torch.no_grad():
            for p, g in zip(params, grads, strict=False):
                if g is not None:
                    p.add_(-lr * g)
        return loss.detach()

    def size(self) -> int:
        return int(self.trace_active.sum().item())
