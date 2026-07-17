import pytest
import torch

from olf.geometry import (
    angular_distance,
    antipodal,
    exponential_map,
    log_map_sphere,
    parallel_transport_sphere,
    project_to_sphere,
    project_to_tangent,
)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_parallel_transport_handles_mixed_coincident_batch(dtype):
    x = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=dtype)
    y = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=dtype)
    v = torch.tensor([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]], dtype=dtype)

    transported = parallel_transport_sphere(x, y, v)

    assert torch.isfinite(transported).all()
    assert torch.allclose(transported[0], v[0], atol=1e-6, rtol=1e-6)
    assert torch.allclose((transported * y).sum(dim=-1), torch.zeros(2, dtype=dtype), atol=1e-6)
    assert torch.allclose(transported.norm(dim=-1), v.norm(dim=-1), atol=1e-6, rtol=1e-6)


def test_mixed_antipodal_batch_is_detected_and_rejected():
    x = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    y = torch.tensor([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    v = torch.tensor([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]])

    assert antipodal(x, y).tolist() == [True, False]
    with pytest.raises(ValueError, match="antipodal"):
        parallel_transport_sphere(x, y, v)
    with pytest.raises(ValueError, match="antipodal"):
        log_map_sphere(x, y)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_exp_log_round_trip_and_tangency(dtype):
    generator = torch.Generator().manual_seed(17)
    x = project_to_sphere(torch.randn(256, 12, generator=generator, dtype=dtype))
    raw = torch.randn(256, 12, generator=generator, dtype=dtype)
    tangent = project_to_tangent(x, raw)
    tangent = tangent / tangent.norm(dim=-1, keepdim=True).clamp_min(1e-12) * 0.7
    y = exponential_map(x, tangent)

    recovered = log_map_sphere(x, y)
    reconstructed = exponential_map(x, recovered)

    tolerance = 2e-5 if dtype == torch.float32 else 2e-7
    assert torch.allclose(reconstructed, y, atol=tolerance, rtol=tolerance)
    assert torch.allclose((x * recovered).sum(dim=-1), torch.zeros(256, dtype=dtype), atol=tolerance)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_parallel_transport_preserves_inner_products(dtype):
    generator = torch.Generator().manual_seed(23)
    x = project_to_sphere(torch.randn(256, 10, generator=generator, dtype=dtype))
    direction = project_to_tangent(x, torch.randn(256, 10, generator=generator, dtype=dtype))
    direction = direction / direction.norm(dim=-1, keepdim=True).clamp_min(1e-12) * 0.8
    y = exponential_map(x, direction)
    u = project_to_tangent(x, torch.randn(256, 10, generator=generator, dtype=dtype))
    v = project_to_tangent(x, torch.randn(256, 10, generator=generator, dtype=dtype))

    transported_u = parallel_transport_sphere(x, y, u)
    transported_v = parallel_transport_sphere(x, y, v)

    tolerance = 2e-5 if dtype == torch.float32 else 2e-7
    assert torch.allclose((u * v).sum(dim=-1), (transported_u * transported_v).sum(dim=-1), atol=tolerance, rtol=tolerance)
    assert torch.allclose((transported_u * y).sum(dim=-1), torch.zeros(256, dtype=dtype), atol=tolerance)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_angular_distance_preserves_exact_endpoints(dtype):
    x = torch.zeros(2, 4, dtype=dtype)
    x[:, 0] = 1.0

    assert torch.equal(angular_distance(x, x), torch.zeros(2, dtype=dtype))
    assert torch.equal(angular_distance(x, -x), torch.full((2,), torch.pi, dtype=dtype))
