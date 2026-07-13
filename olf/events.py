"""Observation-transition events used for consequence grounding."""

import torch


def entity_feature_event_mask(before, after, threshold=1e-5):
    """Detect per-entity feature transitions without using positions.

    Slots provide within-episode correspondence only. Appearance,
    disappearance, or a content change is an event; relative motion by itself
    is deliberately excluded.
    """
    if before.shape != after.shape:
        raise ValueError(
            f"feature tensors must have identical shape, got {before.shape} and {after.shape}"
        )
    if before.ndim != 3:
        raise ValueError("feature tensors must have shape (batch, entities, features)")
    return torch.linalg.vector_norm(after - before, dim=-1) > threshold
