import torch
import torch.nn as nn

class InventionGenerator(nn.Module):
    """
    InventionGenerator generates novel actions (u) under impasse.
    It performs hallucinated rollouts by querying the ConsequenceMemory trace buffer
    to validate candidate action effects, preventing compulsive novelty.
    """
    def __init__(self, latent_dim=32, action_dim=3, hidden_dim=64):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        
        # Generation network: creates base action candidates
        # Input: [h_t, situated_embeddings_mean]
        self.proposal_net = nn.Sequential(
            nn.Linear(latent_dim + hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim),
            nn.Tanh()
        )
        
    def forward(self, h, situated_embeds, consequence_memory, motor_memory=None, num_candidates=5):
        """Generates and validates invention actions using consequence memory rollouts.

        v3: motor_memory seeds candidates from previously-observed
        successful transformations.

        v0.3.2: delta scoring. Retrieved delta_h is used to BONUS candidates
        whose action-space direction/magnitude aligns with successful past
        transformations. No latent→action dimension mixing — deltas are used
        only for scoring, not for direct candidate modification.
        """
        # Mean situated embedding to represent task state
        avg_embed = situated_embeds.mean(dim=1)

        # Generate base action proposal
        inputs = torch.cat([h, avg_embed], dim=-1)
        base_proposal = self.proposal_net(inputs)

        # If consequence memory is empty, return base proposal with some explore noise
        if consequence_memory.trace_active.sum() == 0:
            noise = torch.randn_like(base_proposal) * 0.1
            return torch.clamp(base_proposal + noise, -1.0, 1.0)

        candidates = []

        # v3: seed candidates from motor_memory if available and non-empty.
        delta_info = None  # v0.3.2: store delta statistics for scoring
        if motor_memory is not None and motor_memory.size() > 0:
            mm_result = motor_memory.query_similar_action(h, k=3, return_delta=True)
            if mm_result is not None:
                mm_actions, mm_deltas, mm_scores = mm_result
                if mm_actions is not None and len(mm_actions) > 0:
                    # Bias base proposal toward top motor memory action.
                    top_action = mm_actions[0]
                    base_proposal = 0.5 * base_proposal + 0.5 * top_action.reshape_as(base_proposal)
                    # Add motor-memory-seeded candidates.
                    for i in range(min(2, len(mm_actions))):
                        noise = torch.randn_like(base_proposal) * 0.15
                        cand = torch.clamp(mm_actions[i].reshape_as(base_proposal) + noise, -1.0, 1.0)
                        candidates.append(cand)
                    # v0.3.2: extract delta statistics for scoring bonus.
                    # Use the mean delta direction and magnitude as the
                    # "successful transformation signature."
                    successful_deltas = mm_deltas[mm_scores > 0]  # only positive-scored
                    if len(successful_deltas) > 0:
                        delta_mean = successful_deltas.mean(dim=0)  # (latent_dim,)
                        delta_norm = float(delta_mean.norm().item())
                        delta_info = {"mean": delta_mean, "norm": delta_norm}

        # Create a set of candidate action mutations from the (possibly
        # motor-memory-seeded) base proposal.
        for _i in range(num_candidates):
            noise = torch.randn_like(base_proposal) * 0.25
            cand = torch.clamp(base_proposal + noise, -1.0, 1.0)
            candidates.append(cand)

        best_cand = base_proposal
        best_score = -999.0

        # Validate candidates through consequence memory lookup (hallucination)
        for cand in candidates:
            pred_h, pred_cons, confidence = consequence_memory.retrieve_trace(avg_embed, cand)

            # Consequence vec: [reward, danger_risk, hunger_delta, fatigue_delta]
            pred_reward = pred_cons[0, 0].item()
            pred_danger = pred_cons[0, 1].item()

            # Score: reward minus danger risk, scaled by similarity confidence
            score = (pred_reward - 1.5 * pred_danger) * (1.0 + confidence)

            # v0.3.2.4: delta scoring bonus. Use tangent-space delta direction
            # to score candidates by their predicted transformation alignment.
            # The retrieved delta predicts what transformation the stored action
            # produced. Candidates that would produce similar transformations
            # (in latent space) get higher scores.
            if delta_info is not None and delta_info["norm"] > 1e-4:
                delta_mean = delta_info["mean"]  # tangent-space velocity (latent_dim,)
                # Score alignment: how well does this candidate's implied
                # transformation align with successful past transformations?
                # We use the cosine similarity between the delta direction
                # and the candidate's action-space direction (projected through
                # a simple heuristic: action norm correlates with latent delta norm).
                cand_norm = float(cand.norm().item())
                delta_norm = delta_info["norm"]
                # Norm ratio: candidates with similar magnitude to successful
                # transformations get a bonus
                norm_ratio = min(cand_norm, delta_norm) / (max(cand_norm, delta_norm) + 1e-8)
                # Direction alignment: use the delta's angular consistency
                # (higher norm → more decisive transformation → stronger signal)
                alignment_bonus = 1.0 + 0.3 * norm_ratio * min(1.0, delta_norm / (delta_norm + 0.1))
                score = score * alignment_bonus

            if score > best_score:
                best_score = score
                best_cand = cand

        return best_cand
