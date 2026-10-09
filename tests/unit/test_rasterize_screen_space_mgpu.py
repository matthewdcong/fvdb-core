# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
"""Numerical regression tests for cooperative, camera-by-camera rasterization."""

from contextlib import ExitStack

import pytest
import torch

pytest.importorskip("torch_dgx")

import fvdb.functional as F

pytestmark = pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires at least two CUDA GPUs")


def _scene(camera_count, layout="unpacked", gaussian_count=17, width=35, height=19, empty_intersections=False):
    generator = torch.Generator().manual_seed(137)
    tile_size = 8
    tiles_h, tiles_w = (height + tile_size - 1) // tile_size, (width + tile_size - 1) // tile_size
    record_cameras = 1 if layout == "shared_packed" else camera_count
    means = torch.rand(record_cameras, gaussian_count, 2, generator=generator) * torch.tensor([width, height])
    conics = torch.tensor([0.025, 0.002, 0.03]).expand(record_cameras, gaussian_count, 3).clone()
    features = torch.rand(record_cameras, gaussian_count, 3, generator=generator)
    opacities = torch.rand(record_cameras, gaussian_count, generator=generator) * 0.2 + 0.05
    gaussians = [means, conics, features, opacities]
    if layout != "unpacked":
        gaussians = [value.flatten(0, 1) for value in gaussians]

    offsets, ids = [], []
    for camera in range(camera_count):
        base = 0 if layout == "shared_packed" else camera * gaussian_count
        for tile in range(tiles_h * tiles_w):
            offsets.append(len(ids))
            if not empty_intersections:
                # Repeated records across tiles exercise accumulation on multiple GPUs.
                ids.extend(base + g for g in range(gaussian_count) if (tile + g) % 5 != 4)
    masks = torch.ones(camera_count, tiles_h, tiles_w, dtype=torch.bool)
    masks[0, -1, -1] = False
    if camera_count > 1:
        masks[1] = False  # A camera with no contribution must still publish zero gradients.
    return {
        **dict(zip(("means2d", "conics", "features", "opacities"), gaussians)),
        "image_width": width,
        "image_height": height,
        "image_origin_w": 0,
        "image_origin_h": 0,
        "tile_size": tile_size,
        "tile_offsets": torch.tensor(offsets, dtype=torch.int64).reshape(camera_count, tiles_h, tiles_w),
        "tile_gaussian_ids": torch.tensor(ids, dtype=torch.int32),
        "backgrounds": torch.rand(camera_count, 3, generator=generator),
        "masks": masks,
        "d_loss_d_rendered_features": torch.randn(camera_count, height, width, 3, generator=generator),
        "d_loss_d_rendered_alphas": torch.randn(camera_count, height, width, 1, generator=generator),
    }


def _run(scene, device, abs_grad, strided=False):
    inputs = {name: value.to(device) if isinstance(value, torch.Tensor) else value for name, value in scene.items()}
    if strided:
        for name in ("means2d", "conics", "features", "opacities"):
            value = inputs[name]
            storage = torch.empty((*value.shape, 2), dtype=value.dtype, device=device)
            storage[..., 0].copy_(value)
            inputs[name] = storage[..., 0]
            assert not inputs[name].is_contiguous()
    grad_features = inputs.pop("d_loss_d_rendered_features")
    grad_alphas = inputs.pop("d_loss_d_rendered_alphas")
    rendered, alphas, last_ids = F.rasterize_screen_space_gaussians_fwd(**inputs)
    gradients = F.rasterize_screen_space_gaussians_bwd(
        **inputs,
        rendered_alphas=alphas,
        last_ids=last_ids,
        d_loss_d_rendered_features=grad_features,
        d_loss_d_rendered_alphas=grad_alphas,
        abs_grad=abs_grad,
    )
    return tuple(value.cpu() if value is not None else None for value in (rendered, alphas, last_ids, *gradients))


def _assert_parity(scene, abs_grad=True, strided=False):
    expected = _run(scene, "cuda:0", abs_grad, strided)
    actual = _run(scene, "dgx", abs_grad, strided)
    for reference, result in zip(expected, actual):
        if reference is None:
            assert result is None
        else:
            # Tile atomics and reduce-scatter accumulate in different orders.
            torch.testing.assert_close(result, reference, atol=2e-5, rtol=2e-4)
    if scene["masks"] is not None and scene["means2d"].ndim == 3 and scene["means2d"].shape[0] > 1:
        for gradient in actual[3:]:
            if gradient is not None:
                assert torch.count_nonzero(gradient[1]) == 0


@pytest.mark.parametrize("camera_count", [1, 2, 4])
@pytest.mark.parametrize("layout", ["unpacked", "packed", "shared_packed"])
@pytest.mark.parametrize("abs_grad", [False, True])
def test_camera_cooperation_matches_cuda(camera_count, layout, abs_grad):
    # Seventeen records deliberately misalign scalar and whole-Gaussian reduction shards.
    _assert_parity(_scene(camera_count, layout), abs_grad)


@pytest.mark.parametrize("layout", ["unpacked", "packed"])
def test_camera_cooperation_with_strided_gaussians(layout):
    _assert_parity(_scene(4, layout), strided=True)


@pytest.mark.parametrize(
    "gaussian_count,width,height,empty_intersections",
    [(1, 7, 5, False), (0, 35, 19, False), (17, 35, 19, True)],
)
def test_camera_cooperation_with_empty_work(gaussian_count, width, height, empty_intersections):
    scene = _scene(
        4, gaussian_count=gaussian_count, width=width, height=height, empty_intersections=empty_intersections
    )
    # Keep the single tile active so its one record contributes while other GPUs have no work.
    scene["masks"] = None
    scene["backgrounds"] = None
    _assert_parity(scene)


def test_camera_cooperation_on_nondefault_streams():
    with ExitStack() as stack:
        for device in range(torch.cuda.device_count()):
            stack.enter_context(torch.cuda.stream(torch.cuda.Stream(device=device)))
        _assert_parity(_scene(4))
