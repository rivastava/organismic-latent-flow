"""olf/constellation.py

Constellation Memory per RTCM design.

"Constellation Memory attempted primary + support-cause retrieval;
learned salience was still too weak."

A constellation is a SET of related causes that together explain an
observed effect. When one cause is insufficient (e.g., a delayed
consequence with multiple contributing events), the organism should
retrieve the whole constellation, not just the single most-similar
event.

The implementation:
  - Cluster the fine-level events by their effect-projection similarity.
  - For each cluster (constellation), compute a centroid.
  - At retrieval time, find the top-2 most similar constellations.
  - Within each, return the top-K events as primary + support causes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConstellationMemory(nn.Module):
    """Constellation (multi-cause) memory.

    Clusters fine events into constellations and retrieves multi-cause
    explanations. Used in addition to single-cause retrieval from RTCM.
    """

    def __init__(self, latent_dim=32, n_constellations=4, events_per_constellation=5,
                 hidden_dim=32):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_constellations = n_constellations
        self.events_per_constellation = events_per_constellation

        # Learnable constellation centroids (initialized as random unit vectors).
        centroids = torch.randn(n_constellations, latent_dim)
        centroids = centroids / centroids.norm(dim=-1, keepdim=True)
        self.centroids = nn.Parameter(centroids)

    def compute_membership(self, h_t):
        """Returns membership weights for h_t in each constellation."""
        h_n = F.normalize(h_t, p=2, dim=-1)
        c_n = F.normalize(self.centroids, p=2, dim=-1)
        # Cosine similarity → softmax (so each event gets a distribution).
        sims = (h_n.unsqueeze(-2) * c_n.unsqueeze(0)).sum(dim=-1)
        return F.softmax(sims * 5.0, dim=-1)

    def retrieve_constellation(self, observed_consequence, current_h,
                              primary_k=2, support_k=3, temperature=0.5,
                              history=None, consequence_encoder=None,
                              R_delta=None, top_k_per_constellation=None):
        """Retrieve a multi-cause explanation.

        Returns a list of (constellation_idx, primary_events, support_events)
        tuples, sorted by relevance.
        """
        if history is None or len(history) < 2:
            return []
        if consequence_encoder is None or R_delta is None:
            return []

        device = R_delta.weight.device
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
            encoder_weight_norm = float(consequence_encoder.weight.norm().item())
            if encoder_weight_norm < 1e-3:
                return []

            effect_signal = consequence_encoder(observed_consequence)
            R_T = R_delta.weight.t()
            q_cause = (R_T @ effect_signal.t()).t()
            q_cause = F.normalize(q_cause, p=2, dim=-1)

            # Score each event by sim to q_cause, weighted by constellation
            # membership.
            event_scores = []
            for i, step in enumerate(history):
                h_t = F.normalize(step["h"].reshape(-1), p=2, dim=-1).to(device).unsqueeze(0)
                sim = F.cosine_similarity(q_cause, h_t, dim=-1).item()
                mem = self.compute_membership(h_t).squeeze(0)
                # Score is sim weighted by the highest constellation membership.
                max_mem = float(mem.max().item())
                best_c = int(mem.argmax().item())
                event_scores.append((i, sim * (1.0 + max_mem), best_c))

            # Group by constellation.
            from collections import defaultdict
            by_constellation = defaultdict(list)
            for idx, score, cidx in event_scores:
                by_constellation[cidx].append((idx, score))

            # Top 2 constellations by their best member.
            constellation_best = []
            for cidx, evs in by_constellation.items():
                evs.sort(key=lambda x: -x[1])
                constellation_best.append((cidx, evs[0][1], evs))
            constellation_best.sort(key=lambda x: -x[1])
            top = constellation_best[:2]  # primary + support

            return [
                {
                    "constellation": cidx,
                    "primary": evs[: primary_k],
                    "support": evs[primary_k : primary_k + support_k],
                }
                for cidx, _, evs in top
            ]
