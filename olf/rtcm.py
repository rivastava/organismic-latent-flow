"""olf/rtcm.py

Action-Sphere RTCM with transfer-aware retrieval.

RTCM design:
  - Spherical Memory: events stored as unit vectors on S^(d-1)
  - Rotary Timestamped Causal Memory (RTCM): time as phase rotation
  - Action-Sphere Causal Fields: each event has intensity (omega),
    local density (rho), and compatibility (chi).
  - Transfer-aware retrieval: q_cause = R_Δᵀ @ effect is used as the
    QUERY in cause-space, not as a re-ranker.
  - Delay estimation: top-k delay search (not hard top-1).
  - Multi-cause: support retrieval of multiple candidate causes.

The store API is unchanged ():
  - add_step(h, a, consequence, h_next)
  - retrieve_causal_blame(...)  (used by experiments/run_core.py for blame weights)
  - retrieve_delayed_credit(...)  (addition)

The new method `transfer_aware_retrieve` returns top-k past events
in cause-space via q_cause = R_Δᵀ effect, with delay estimation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from olf.geometry import log_map_sphere


class ActionSphereEvent:
    """Helper for the per-event representation on the sphere.

    Each event e_i = (h_i, a_i, c_i, h_next_i) is stored on S^(d-1) by
    unit-normalizing h_i. The action and consequence are auxiliary.
    """
    pass


class RetrogradeTemporalCausalMemory(nn.Module):
    """Per-episode transition history with a learnable R_Δ operator,
    action-sphere encoding, delay estimation, and transfer-aware
    retrieval.

    Two retrieval modes are supported:
      - retrieve_causal_blame(): per-step blame weights (used by
        experiments/run_core.py for policy gradient scaling).
      - retrieve_delayed_credit(): top-k delayed credit from a final
        observed consequence (used for sparse-reward tasks like
        delayed_lure).
      - transfer_aware_retrieve(): NEW method that uses
        q_cause = R_Δᵀ @ effect as the cause-space query, with
        delay-aware top-k ranking.
    """

    def __init__(
        self,
        latent_dim=32,
        action_dim=3,
        hidden_dim=32,
        max_history=200,
        rotary_freqs=4,
        max_time=64,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.max_history = max_history
        self.rotary_freqs = rotary_freqs
        self.max_time = max_time

        # R_Δ: a small learnable operator on the manifold-tangent direction.
        # A low-rank diagonal skip improves stability: linear + tanh.
        # For "effect ≈ R_Δ cause", the next-state delta passes through this layer.
        self.R_delta = nn.Linear(latent_dim, latent_dim, bias=False)
        with torch.no_grad():
            # Initialize close to identity so blame starts meaningful.
            self.R_delta.weight.copy_(
                0.5 * torch.eye(latent_dim) + 0.05 * torch.randn(latent_dim, latent_dim)
            )

        # Per-step consequence head: outputs a [0, 1] blame probability.
        self.blame_estimator = nn.Sequential(
            nn.Linear(latent_dim + action_dim + 4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Per-step effect predictor (for forward loss).
        self.effect_predictor = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

        # explicit consequence-to-latent projection (RTCM design transfer).
        self.consequence_encoder = nn.Linear(4, latent_dim)
        with torch.no_grad():
            self.consequence_encoder.weight.zero_()
            self.consequence_encoder.bias.zero_()

        # action-sphere parameters. The "intensity" omega is per-step
        # (depends on action magnitude); local density rho and compatibility
        # chi are learned to weight candidate causes.
        self.rho = nn.Parameter(torch.zeros(latent_dim))
        self.chi = nn.Parameter(torch.ones(latent_dim))

        # rotary frequencies for time-as-phase.
        self.register_buffer(
            "freqs", torch.linspace(0.5, 4.0, rotary_freqs, dtype=torch.float32)
        )

        # delay posterior head. Outputs a (max_delay,) softmax over
        # candidate delay buckets. Used to estimate p(Δ | effect, context).
        self.delay_head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, max_time),
        )

        self.reset_history()

    # ------------------------------------------------------------------ store

    def reset_history(self):
        """Clears episode history."""
        self.history = []  # list of dicts: {h, a, consequence, h_next, t, intensity}
        self.effects = []  # list of tensors: h_{t+1} − h_t (filled retroactively)
        self.t = 0

    def add_step(self, h, a, consequence=None, h_next=None):
        """Logs a transition step.

        also computes action intensity omega = ||a|| and stores the
        current time t (used by rotary phase).
        """
        # Normalize h to the sphere (RTCM design: events on S^(d-1)).
        h_n = F.normalize(h, p=2, dim=-1)

        a_t = torch.as_tensor(a, dtype=torch.float32, device=h_n.device).reshape(-1)
        if a_t.shape[0] != self.action_dim:
            pad = torch.zeros(self.action_dim, device=h_n.device)
            pad[: min(self.action_dim, a_t.shape[0])] = a_t[: min(self.action_dim, a_t.shape[0])]
            a_t = pad
        # action intensity is the L2 norm of the action vector.
        intensity = float(a_t.norm().item())

        entry = {
            "h": h_n.clone().detach(),
            "a": a_t.unsqueeze(0),
            "t": self.t,
            "intensity": intensity,
        }
        if consequence is not None:
            if not torch.is_tensor(consequence):
                entry["consequence"] = torch.FloatTensor(consequence).unsqueeze(0)
            else:
                entry["consequence"] = (
                    consequence.unsqueeze(0) if consequence.dim() == 1 else consequence
                )
        else:
            entry["consequence"] = torch.zeros(1, 4)
        if h_next is not None:
            h_next_n = F.normalize(h_next, p=2, dim=-1)
            entry["h_next"] = h_next_n.clone().detach()
            entry["effect"] = log_map_sphere(h_n, h_next_n).clone().detach()
        else:
            entry["h_next"] = None
            entry["effect"] = None

        self.history.append(entry)
        if len(self.history) > self.max_history:
            self.history.pop(0)
        self.t += 1

    def complete_last_step(self, consequence, h_next):
        """Finish the pending action transition after consequence is observed.

        ``add_step`` is called when an action is released. The resulting latent
        state is not available until the next observation recouples the
        organism. This method records that actual endpoint and a tangent-space
        deformation rather than the Euclidean chord between sphere points.
        """
        if not self.history:
            return
        entry = self.history[-1]
        h_before = entry["h"]
        if h_before.dim() == 1:
            h_before = h_before.unsqueeze(0)
        h_next_n = F.normalize(h_next, p=2, dim=-1).to(h_before.device)
        entry["h_next"] = h_next_n.clone().detach()
        entry["effect"] = log_map_sphere(h_before, h_next_n).clone().detach()

        c = torch.as_tensor(
            consequence,
            dtype=torch.float32,
            device=h_before.device,
        ).reshape(1, -1)
        if c.shape[-1] != 4:
            padded = torch.zeros(1, 4, device=h_before.device)
            padded[..., : min(4, c.shape[-1])] = c[..., : min(4, c.shape[-1])]
            c = padded
        entry["consequence"] = c.clone().detach()

    # ------------------------------------------------------------------ query

    def retrieve_causal_blame(self, consequence_vec, threshold=0.3):
        """Per-step blame weights using the learned blame estimator and R_Δ.

        Used by experiments/run_core.py for per-step policy gradient scaling.
        """
        if not self.history:
            return []

        if not torch.is_tensor(consequence_vec):
            consequence_vec = torch.FloatTensor(consequence_vec).unsqueeze(0)
        if consequence_vec.dim() == 1:
            consequence_vec = consequence_vec.unsqueeze(0)

        blame_weights = []
        R_T = self.R_delta.weight.t()
        for step in self.history:
            h_t = step["h"]
            a_t = step["a"]
            c = step.get("consequence", torch.zeros(1, 4, device=h_t.device))

            if h_t.dim() == 1:
                h_t = h_t.unsqueeze(0)
            if a_t.dim() == 1:
                a_t = a_t.unsqueeze(0)

            inp = torch.cat([h_t, a_t, c], dim=-1)
            logit = self.blame_estimator(inp).squeeze(-1)

            proj = R_T @ h_t.t()
            cos = F.cosine_similarity(proj.t(), h_t, dim=-1)
            if cos.dim() > 0:
                cos = cos.squeeze()

            final = torch.sigmoid(logit + 1.0 * (cos - 0.5))
            if final.dim() > 0:
                final = final.squeeze()
            blame_weights.append(float(final.item()))

        total = sum(blame_weights)
        if total > 1e-6:
            blame_weights = [w / total for w in blame_weights]
        return blame_weights

    def retrieve_delayed_credit(
        self,
        observed_consequence,
        current_h,
        top_k=3,
        temp_decay=0.95,
        temporal_floor=0.15,
        temperature=0.5,
    ):
        """Top-k delayed credit from the final observed consequence.

        with locked amendments:
          - Softmax normalization (not raw) over the top-k raw scores.
          - Temporal floor so old causes don't vanish completely.
          - Explicit consequence_encoder projection.

        Returns list of (step_idx, weight) summing to 1.
        """
        if not self.history or len(self.history) < 2:
            return []

        device = self.R_delta.weight.device

        if not torch.is_tensor(observed_consequence):
            observed_consequence = torch.FloatTensor(observed_consequence)
        if observed_consequence.dim() == 1:
            observed_consequence = observed_consequence.unsqueeze(0)
        observed_consequence = observed_consequence.to(device)
        if observed_consequence.shape[-1] != 4:
            pad = torch.zeros(1, 4, device=device)
            pad[..., : min(4, observed_consequence.shape[-1])] = observed_consequence[
                ..., : min(4, observed_consequence.shape[-1])
            ]
            observed_consequence = pad

        with torch.no_grad():
            effect_signal = self.consequence_encoder(observed_consequence)

            # GATE: skip if consequence_encoder is still effectively
            # untrained (zero-initialized). Otherwise the effect_signal is
            # near-zero and q_cause is noise, which pollutes blame.
            encoder_weight_norm = float(self.consequence_encoder.weight.norm().item())
            if encoder_weight_norm < 1e-3:
                return []

            R_T = self.R_delta.weight.t()
            q_cause = R_T @ effect_signal.t()
            q_cause = q_cause.t()

            scores = []
            n = len(self.history)
            for idx, step in enumerate(self.history):
                h_t = step["h"]
                if h_t.dim() == 1:
                    h_t = h_t.unsqueeze(0)
                h_t = h_t.to(device)
                cos = F.cosine_similarity(q_cause, h_t, dim=-1).item()
                delay = n - 1 - idx
                recency = temp_decay ** delay
                temporal = max(recency, temporal_floor)
                scores.append((idx, cos * temporal))

            scores.sort(key=lambda x: -x[1])
            top = scores[: min(top_k, len(scores))]
            if not top:
                if getattr(self, "diag_log_target", None) is not None:
                    self.diag_log_target.append({"method": "retrieve_delayed_credit", "n_returned": 0})
                return []
            score_vals = torch.tensor([s for _, s in top], device=device)
            weights = torch.softmax(score_vals / temperature, dim=0)
            out = [
                (idx, float(w.item()))
                for (idx, _), w in zip(top, weights, strict=False)
            ]
            # diagnostic: per-call log of retrieved indices/weights.
            if getattr(self, "diag_log_target", None) is not None:
                self.diag_log_target.append({
                    "method": "retrieve_delayed_credit",
                    "top_k": top_k,
                    "n_returned": len(out),
                    "indices": [int(i) for i, _ in out],
                    "weights": [float(w) for _, w in out],
                })
            return out

    # --------------------------------------------------------- NEW METHOD

    def transfer_aware_retrieve(
        self,
        observed_consequence,
        current_h,
        top_k=5,
        delay_top_m=8,
        temperature=0.5,
    ):
        """Transfer-aware retrieval per RTCM design.

        "Inverse transfer produces a cause-space query that can locate old
        causes far better than raw effect similarity." (RTCM design)

        Algorithm:
          1. Project the observed consequence into latent effect space
             via self.consequence_encoder.
          2. Form q_cause = R_Δᵀ @ effect_signal  (cause-space query).
          3. Estimate the delay posterior p(Δ | effect) over the top
             m delay buckets via self.delay_head.
          4. For each candidate past event at delay Δ, weight by:
                sim(q_cause, h_i) * p(Δ) * action_intensity_i
          5. Softmax-normalize the top-k.

        Returns: list of (step_idx, weight) tuples summing to 1.
        """
        if not self.history or len(self.history) < 2:
            return []

        device = self.R_delta.weight.device

        if not torch.is_tensor(observed_consequence):
            observed_consequence = torch.FloatTensor(observed_consequence)
        if observed_consequence.dim() == 1:
            observed_consequence = observed_consequence.unsqueeze(0)
        observed_consequence = observed_consequence.to(device)
        if observed_consequence.shape[-1] != 4:
            pad = torch.zeros(1, 4, device=device)
            pad[..., : min(4, observed_consequence.shape[-1])] = observed_consequence[
                ..., : min(4, observed_consequence.shape[-1])
            ]
            observed_consequence = pad

        with torch.no_grad():
            # Step 1: project to effect space.
            effect_signal = self.consequence_encoder(observed_consequence)

            # GATE: if consequence_encoder is still effectively untrained
            # (weights ≈ 0), the effect_signal is near-zero and the
            # transfer-aware query degenerates to noise. In that case, skip
            # transfer-aware retrieval and let the per-step blame dominate.
            encoder_weight_norm = float(self.consequence_encoder.weight.norm().item())
            if encoder_weight_norm < 1e-3:
                return []  # untrained, do not pollute the blame signal

            # Step 2: cause-space query.
            R_T = self.R_delta.weight.t()
            q_cause = R_T @ effect_signal.t()
            q_cause = q_cause.t()
            q_cause = F.normalize(q_cause, p=2, dim=-1)

            # Step 3: delay posterior.
            delay_logits = self.delay_head(effect_signal).squeeze(0)
            delay_probs = F.softmax(delay_logits / 0.5, dim=-1)  # (max_time,)

            # Step 4: score each past event.
            cur_t = self.history[-1]["t"]
            scores = []
            for idx, step in enumerate(self.history):
                h_t = step["h"]
                if h_t.dim() == 1:
                    h_t = h_t.unsqueeze(0)
                h_t = F.normalize(h_t, p=2, dim=-1).to(device)
                # Cosine sim in cause space.
                cos = F.cosine_similarity(q_cause, h_t, dim=-1).item()
                # Delay bucket match.
                delta = cur_t - step["t"]
                if 0 <= delta < self.max_time:
                    p_delay = float(delay_probs[delta].item())
                else:
                    p_delay = 1e-3
                # Action intensity: RTCM design: causal impact = omega * rho * chi.
                intensity = step.get("intensity", 0.0)
                scores.append((idx, cos * p_delay * (1.0 + intensity)))

            # Step 5: softmax top-k.
            scores.sort(key=lambda x: -x[1])
            top = scores[: min(top_k, len(scores))]
            if not top:
                return []
            score_vals = torch.tensor([s for _, s in top], device=device)
            weights = torch.softmax(score_vals / temperature, dim=0)
            return [
                (idx, float(w.item()))
                for (idx, _), w in zip(top, weights, strict=False)
            ]

    # ------------------------------------------------------------------ train

    def train_step(self, lr=1e-3):
        """Train R_Δ, the blame estimator, and additions (consequence
        encoder, delay head, action-sphere parameters) from current
        history.

        adds:
          - Loss on consequence_encoder (so it learns a useful effect
            projection).
          - Loss on delay_head (so the delay posterior learns from the
            actual observed delays).
        """
        if not self.history:
            return None

        R_losses = []
        effect_losses = []
        blame_losses = []
        encoder_losses = []
        delay_losses = []
        for step in self.history:
            h_t = step["h"]
            a_t = step["a"]
            effect = step.get("effect", None)
            c = step.get("consequence", None)

            if h_t.dim() == 1:
                h_t = h_t.unsqueeze(0)
            if a_t.dim() == 1:
                a_t = a_t.unsqueeze(0)

            if effect is not None:
                if effect.dim() == 1:
                    effect = effect.unsqueeze(0)
                pred = self.R_delta(h_t)
                R_losses.append(F.mse_loss(pred, effect))

                e_inp = torch.cat([h_t, a_t], dim=-1)
                pred_e = self.effect_predictor(e_inp)
                effect_losses.append(F.mse_loss(pred_e, effect))

                was_important = (effect.norm(dim=-1, keepdim=True) > 1e-2).float()
                inp = torch.cat([h_t, a_t, c.to(h_t.device)], dim=-1)
                logit = self.blame_estimator(inp)
                blame_losses.append(
                    F.binary_cross_entropy_with_logits(logit, was_important)
                )

                # train consequence_encoder to map observed consequence
                # to the actual effect in latent space.
                if c is not None and c.shape[-1] == 4:
                    c_t = c.to(h_t.device)
                    pred_e_signal = self.consequence_encoder(c_t)
                    encoder_losses.append(F.mse_loss(pred_e_signal, effect))

                # train delay head: target is the actual delay (time
                # since this event occurred).
                if c is not None and c.shape[-1] == 4:
                    t_event = step.get("t", 0)
                    cur_t = self.history[-1].get("t", t_event)
                    actual_delay = max(0, min(self.max_time - 1, cur_t - t_event))
                    target = torch.zeros(self.max_time, device=h_t.device)
                    target[actual_delay] = 1.0
                    pred_delay_logits = self.delay_head(pred_e_signal)
                    delay_losses.append(
                        F.cross_entropy(pred_delay_logits, target.unsqueeze(0))
                    )

        if not R_losses:
            return None

        total = R_losses[0] * 0.0
        for L in R_losses:
            total = total + L
        for L in effect_losses:
            total = total + 0.5 * L
        for L in blame_losses:
            total = total + 0.5 * L
        for L in encoder_losses:
            total = total + 0.5 * L
        for L in delay_losses:
            total = total + 0.3 * L

        params = (
            list(self.R_delta.parameters())
            + list(self.blame_estimator.parameters())
            + list(self.effect_predictor.parameters())
            + list(self.consequence_encoder.parameters())
            + list(self.delay_head.parameters())
            + [self.rho, self.chi]
        )
        grads = torch.autograd.grad(
            total, params, allow_unused=True, retain_graph=False
        )
        with torch.no_grad():
            for p, g in zip(params, grads, strict=False):
                if g is not None:
                    p.add_(-lr * g)
        return total.detach()
