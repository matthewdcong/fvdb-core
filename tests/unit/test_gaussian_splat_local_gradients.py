# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

"""Compare Gaussian-owned DGX gradient shards against the CUDA implementation."""

import pytest
import torch

import fvdb.functional as F

pytest.importorskip("torch_dgx", reason="torch-dgx is required for managed gradient outputs")


@pytest.fixture(autouse=True)
def require_dgx():
    if not torch.cuda.is_available() or not torch.dgx.is_available():
        pytest.skip("CUDA and DGX are required")


def _camera_matrices():
    matrices = torch.eye(4).repeat(2, 1, 1)
    matrices[1, :3, 3] = torch.tensor([0.2, -0.1, 0.3])
    return matrices


def _assert_outputs_match(actual, expected):
    for result, reference in zip(actual, expected, strict=True):
        if reference is None:
            assert result is None
        else:
            assert result is not None
            torch.testing.assert_close(result, reference, atol=1e-3, rtol=2e-4)


@pytest.mark.parametrize("num_gaussians", [1, 1025])
@pytest.mark.parametrize("stored_degree,degree", [(0, 0), (4, 0), (4, 1), (4, 4)])
@pytest.mark.parametrize("means_grad,camera_grad", [(False, False), (True, False), (False, True), (True, True)])
def test_sh_backward_local_shards(num_gaussians, stored_degree, degree, means_grad, camera_grad):
    # One Gaussian exercises empty GPU shards; 1025 exercises uneven shard boundaries.
    generator = torch.Generator().manual_seed(218)
    means = torch.randn(num_gaussians, 3, generator=generator)
    means[:, 2] += 4.0
    coefficients = torch.randn(num_gaussians, (stored_degree + 1) ** 2 - 1, 3, generator=generator)
    color_grad = torch.randn(2, num_gaussians, 3, generator=generator)
    radii = torch.ones(2, num_gaussians, 2, dtype=torch.int32)
    radii[:, 6::7] = 0
    radii[1, 1::5, 0] = 0
    matrices = _camera_matrices()

    def run(device):
        inputs = [tensor.to(device) for tensor in (means, matrices, coefficients, color_grad, radii)]
        ids = torch.empty(0, dtype=torch.int32, device=device)
        copies = []
        # Release each result so later calls can reuse managed and local allocations.
        for _ in range(3):
            result = F.evaluate_spherical_harmonics_bwd(
                degree,
                2,
                num_gaussians,
                inputs[0],
                inputs[1],
                ids,
                ids,
                inputs[2],
                inputs[3],
                inputs[4],
                means_grad,
                camera_grad,
            )
            copies.append([None if tensor is None else tensor.cpu() for tensor in result])
            del result
        return copies

    expected = run("cuda:0")
    actual = run("dgx")
    for result, reference in zip(actual, expected, strict=True):
        _assert_outputs_match(result, reference)
        if stored_degree > degree:
            unused = result[1][:, (degree + 1) ** 2 - 1 :]
            torch.testing.assert_close(unused, torch.zeros_like(unused), atol=0.0, rtol=0.0)
        torch.testing.assert_close(result[0][6::7], torch.zeros_like(result[0][6::7]), atol=0.0, rtol=0.0)


@pytest.mark.parametrize("num_gaussians", [1, 1025])
@pytest.mark.parametrize("ortho", [False, True])
@pytest.mark.parametrize("camera_grad", [False, True])
def test_projection_backward_local_shards(num_gaussians, ortho, camera_grad):
    generator = torch.Generator().manual_seed(219)
    means = torch.rand(num_gaussians, 3, generator=generator) - 0.5
    means[:, 2] += 4.0
    quats = torch.randn(num_gaussians, 4, generator=generator)
    quats = torch.nn.functional.normalize(quats, dim=-1)
    log_scales = torch.full((num_gaussians, 3), -2.0)
    matrices = _camera_matrices()
    intrinsics = torch.tensor([[40.0, 0.0, 64.0], [0.0, 38.0, 48.0], [0.0, 0.0, 1.0]]).repeat(2, 1, 1)
    geometry = (means, quats, log_scales, matrices, intrinsics)
    cuda_geometry = [tensor.cuda() for tensor in geometry]
    radii, _, _, conics, compensations = F.project_gaussians_analytic_fwd(
        *cuda_geometry, 128, 96, 0.3, 0.01, 1e10, 0.0, True, ortho
    )
    radii = radii.cpu()
    radii[:, 6::7] = 0
    radii[1, 1::5, 0] = 0
    conics, compensations = conics.cpu(), compensations.cpu()
    means2d_grad = torch.randn(2, num_gaussians, 2, generator=generator)
    conics_grad = torch.randn(2, num_gaussians, 3, generator=generator)
    depths_grad = torch.randn(2, num_gaussians, generator=generator)
    compensations_grad = torch.randn(2, num_gaussians, generator=generator)

    def run(device):
        inputs = [tensor.to(device) for tensor in geometry]
        projected = [tensor.to(device) for tensor in (compensations, radii, conics)]
        gradients = [tensor.to(device) for tensor in (means2d_grad, depths_grad, conics_grad, compensations_grad)]
        # These state arrays retain global Gaussian indexing, unlike the local gradient buffers.
        norms = torch.linspace(0.1, 0.2, num_gaussians).to(device)
        max_radii = torch.full((num_gaussians,), 2, dtype=torch.int32, device=device)
        counts = torch.full((num_gaussians,), 3, dtype=torch.int32, device=device)
        copies = []
        for _ in range(3):
            result = F.project_gaussians_analytic_bwd(
                *inputs,
                projected[0],
                128,
                96,
                0.3,
                projected[1],
                projected[2],
                gradients[0],
                gradients[1] if camera_grad else None,
                gradients[2],
                gradients[3],
                camera_grad,
                ortho,
                norms,
                max_radii,
                counts,
            )
            copies.append([None if tensor is None else tensor.cpu() for tensor in result])
            del result
        return copies, [norms.cpu(), max_radii.cpu(), counts.cpu()]

    expected, expected_state = run("cuda:0")
    actual, actual_state = run("dgx")
    for result, reference in zip(actual, expected, strict=True):
        _assert_outputs_match(result, reference)
        for gradient in (result[0], result[2], result[3]):
            torch.testing.assert_close(gradient[6::7], torch.zeros_like(gradient[6::7]), atol=0.0, rtol=0.0)
    _assert_outputs_match(actual_state, expected_state)
