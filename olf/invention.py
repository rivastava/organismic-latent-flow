import torch
import torch.nn as nn


class InventionGenerator(nn.Module):
    """Generate abstract alternatives when the organism reaches an impasse."""

    def __init__(self, latent_dim=32, action_dim=3, hidden_dim=64):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.proposal_net = nn.Sequential(
            nn.Linear(latent_dim + hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim),
            nn.Tanh(),
        )

    def forward(
        self,
        h,
        situated_embeds,
        consequence_memory,
        motor_memory=None,
        num_candidates=5,
    ):
        """Generate alternatives and select a non-dominated viable action.

        Candidate birth is symmetric around the current proposal and may reuse
        actions observed in nearby latent states. Selection uses predicted
        lethal risk, body-state deformation, and retrieval uncertainty. It does
        not read stored reward, success, task identity, or privileged state.
        """
        avg_embed = situated_embeds.mean(dim=1)
        inputs = torch.cat([h, avg_embed], dim=-1)
        base_proposal = self.proposal_net(inputs)
        if not bool((consequence_memory.trace_active > 0.5).any()):
            return base_proposal

        candidates = [base_proposal, -base_proposal]
        if motor_memory is not None and motor_memory.size() > 0:
            observed = motor_memory.query_nearby_actions(
                h, k=max(1, num_candidates // 2)
            )
            if observed is not None:
                for action in observed:
                    action = action.reshape_as(base_proposal)
                    candidates.extend((action, -action))
        candidates = candidates[:max(1, num_candidates)]

        objectives = []
        confidences = []
        for candidate in candidates:
            _, predicted, confidence = consequence_memory.retrieve_trace(
                avg_embed, candidate
            )
            objectives.append(
                (
                    float(predicted[0, 1].clamp_min(0.0)),
                    float(predicted[0, 2]),
                    float(predicted[0, 3]),
                    1.0 - float(confidence),
                )
            )
            confidences.append(float(confidence))

        front = _non_dominated_indices(objectives)
        selected = max(
            front,
            key=lambda index: (
                confidences[index],
                -float(torch.norm(candidates[index] - base_proposal).detach()),
                -index,
            ),
        )
        return torch.clamp(candidates[selected], -1.0, 1.0)


def _non_dominated_indices(objectives):
    front = []
    for index, current in enumerate(objectives):
        dominated = False
        for other_index, other in enumerate(objectives):
            if index == other_index:
                continue
            no_worse = all(
                left <= right
                for left, right in zip(other, current, strict=True)
            )
            strictly_better = any(
                left < right
                for left, right in zip(other, current, strict=True)
            )
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            front.append(index)
    return front
