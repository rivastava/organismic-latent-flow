import torch

from olf.attractor import AttractorField


def test_attractor_slots_start_inactive_and_exert_no_random_pull():
    field = AttractorField(latent_dim=8, max_attractors=3)
    h = torch.zeros(1, 8)
    h[0, 0] = 1.0

    active, weights, _ = field.get_active_attractors()
    tendency, total_weight = field.compute_tendency(h)

    assert active.shape == (0, 8)
    assert weights.numel() == 0
    assert total_weight < 0.001
    assert torch.equal(tendency, h)


def test_create_at_activates_an_experience_grounded_slot():
    field = AttractorField(latent_dim=8, max_attractors=3)
    target = torch.zeros(1, 8)
    target[0, 2] = 1.0

    idx = field.create_at(target)
    active, _, _ = field.get_active_attractors()

    assert idx == 0
    assert active.shape == (1, 8)
    assert torch.allclose(active[0], target[0])


def test_nearby_attractor_observations_merge_without_consuming_capacity():
    field = AttractorField(latent_dim=8, max_attractors=3)
    first = torch.zeros(1, 8)
    first[0, 0] = 1.0
    nearby = first.clone()
    nearby[0, 1] = 0.05

    first_idx = field.create_at(first)
    second_idx = field.create_at(nearby)
    active, _, _ = field.get_active_attractors()

    assert first_idx == second_idx == 0
    assert active.shape[0] == 1
