"""olf/hierarchical_rtcm.py

Hierarchical RTCM per Action-Sphere RTCM research memo §6.4.

"In favorable conditions with strong anchors, transfer-aware hierarchical
retrieval recovered causes up to 65,536 steps back."

The hierarchical structure is a 2-level RTCM:
  - Level 1 (coarse): one summary vector per "chunk" of N steps.
  - Level 2 (fine): the existing per-step RTCM (R_Δ at the step level).

Retrieval proceeds in two stages:
  1. Score all coarse-level chunks by their similarity to q_cause.
  2. Within the top-K coarse chunks, search fine-level events.

This is the "hierarchical spherical memory search" in the memo's
combined architecture (Figure 1).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HierarchicalRTCM(nn.Module):
    """A 2-level hierarchical RTCM.

    Level 1: coarse chunks of N steps. Each chunk has a summary
        vector (the mean of the step vectors in the chunk, projected
        to S^{d-1}).
    Level 2: fine per-step events (R_Δ-style).

    Retrieval:
        given an effect_signal, compute q_cause = R_Δᵀ @ effect_signal.
        score coarse chunks by cosine sim to q_cause.
        within top-K coarse chunks, score fine events similarly.
        return the top-N fine events as candidate causes.
    """

    def __init__(self, latent_dim=32, chunk_size=16, top_chunks=2,
                 top_fine_per_chunk=3, action_dim=3, hidden_dim=32):
        super().__init__()
        self.latent_dim = latent_dim
        self.chunk_size = chunk_size
        self.top_chunks = top_chunks
        self.top_fine_per_chunk = top_fine_per_chunk

        # The fine-level RTCM (R_Δ + consequence_encoder).
        from olf.rtcm import RetrogradeTemporalCausalMemory
        self.fine_rtcm = RetrogradeTemporalCausalMemory(
            latent_dim=latent_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            max_history=chunk_size * top_chunks * 4,  # enough for several chunks
        )

    def add_step(self, h, a, consequence=None, h_next=None):
        """Add a step to the fine RTCM."""
        self.fine_rtcm.add_step(h, a, consequence, h_next)

    def reset_history(self):
        self.fine_rtcm.reset_history()

    def retrieve_top_k(self, observed_consequence, top_k=5, temperature=0.5):
        """Hierarchical retrieval per memo §6.4.

        1. Compute q_cause = R_Δᵀ @ effect_signal (cause-space query).
        2. Split fine history into coarse chunks of size `chunk_size`.
        3. Score each chunk by cosine sim of its mean to q_cause.
        4. Keep top `top_chunks` chunks.
        5. Within those, score each fine event and keep top
           `top_fine_per_chunk`.
        6. Softmax-normalize the final top-k.
        """
        if not self.fine_rtcm.history:
            return []
        history = self.fine_rtcm.history
        n = len(history)
        if n < 2:
            return []

        device = self.fine_rtcm.R_delta.weight.device

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
            # Check if consequence_encoder is trained enough.
            encoder_weight_norm = float(self.fine_rtcm.consequence_encoder.weight.norm().item())
            if encoder_weight_norm < 1e-3:
                return []

            effect_signal = self.fine_rtcm.consequence_encoder(observed_consequence)
            R_T = self.fine_rtcm.R_delta.weight.t()
            q_cause = (R_T @ effect_signal.t()).t()
            q_cause = F.normalize(q_cause, p=2, dim=-1)

            # Split into chunks.
            n_chunks = (n + self.chunk_size - 1) // self.chunk_size
            chunk_scores = []
            for ci in range(n_chunks):
                start = ci * self.chunk_size
                end = min((ci + 1) * self.chunk_size, n)
                chunk_h = torch.stack(
                    [F.normalize(history[i]["h"].reshape(-1), p=2, dim=-1) for i in range(start, end)]
                ).to(device)
                chunk_mean = chunk_h.mean(dim=0, keepdim=True)
                chunk_mean = F.normalize(chunk_mean, p=2, dim=-1)
                sim = F.cosine_similarity(q_cause, chunk_mean, dim=-1).item()
                chunk_scores.append((ci, sim, start, end))

            # Top-K chunks.
            chunk_scores.sort(key=lambda x: -x[1])
            top_chunks = chunk_scores[: self.top_chunks]

            # Within top chunks, score fine events.
            fine_scores = []
            for _ci, csim, start, end in top_chunks:
                for i in range(start, end):
                    h_t = F.normalize(history[i]["h"].reshape(-1), p=2, dim=-1).to(device).unsqueeze(0)
                    sim = F.cosine_similarity(q_cause, h_t, dim=-1).item()
                    # Weight by chunk sim too (so chunks the query likes more dominate).
                    fine_scores.append((i, sim * (1.0 + csim)))

            # Top-K fine events.
            fine_scores.sort(key=lambda x: -x[1])
            top = fine_scores[: min(top_k, len(fine_scores))]
            if not top:
                return []
            score_vals = torch.tensor([s for _, s in top], device=device)
            weights = torch.softmax(score_vals / temperature, dim=0)
            return [
                (idx, float(w.item()))
                for (idx, _), w in zip(top, weights, strict=False)
            ]
