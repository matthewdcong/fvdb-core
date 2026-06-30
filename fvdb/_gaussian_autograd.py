# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""Python torch.autograd.Function wrappers for Gaussian splatting dispatch functions."""

from __future__ import annotations

from typing import Any

import torch

from . import _fvdb_cpp as _C


# ---------------------------------------------------------------------------
#  Projection (analytic, jagged)
# ---------------------------------------------------------------------------


class _ProjectGaussiansJaggedFn(torch.autograd.Function):
    """Python autograd wrapper for the jagged Gaussian projection dispatch."""

    @staticmethod
    def forward(
        ctx,
        g_sizes: torch.Tensor,
        means: torch.Tensor,
        quats: torch.Tensor,
        scales: torch.Tensor,
        c_sizes: torch.Tensor,
        world_to_cam: torch.Tensor,
        projection_matrices: torch.Tensor,
        image_width: int,
        image_height: int,
        eps2d: float,
        near: float,
        far: float,
        min_radius_2d: float,
        ortho: bool,
    ):
        result = _C.project_gaussians_analytic_jagged_fwd(
            g_sizes,
            means,
            quats,
            scales,
            c_sizes,
            world_to_cam,
            projection_matrices,
            image_width,
            image_height,
            eps2d,
            near,
            far,
            min_radius_2d,
            ortho,
        )
        radii, means2d, depths, conics, compensations = (
            result[0],
            result[1],
            result[2],
            result[3],
            result[4],
        )

        ctx.save_for_backward(
            g_sizes,
            means,
            quats,
            scales,
            c_sizes,
            world_to_cam,
            projection_matrices,
            radii,
            conics,
        )
        ctx.image_width = image_width
        ctx.image_height = image_height
        ctx.eps2d = eps2d
        ctx.ortho = ortho

        return radii, means2d, depths, conics, compensations

    @staticmethod
    def backward(ctx: Any, *grad_outputs: torch.Tensor | None) -> tuple[torch.Tensor | None, ...]:
        grad_means2d = grad_outputs[1]
        grad_depths = grad_outputs[2]
        grad_conics = grad_outputs[3]
        # grad_outputs[4] is grad_compensations -- the jagged backward dispatch
        # does not consume it, so we ignore it here.
        if grad_means2d is not None:
            grad_means2d = grad_means2d.contiguous()
        if grad_depths is not None:
            grad_depths = grad_depths.contiguous()
        if grad_conics is not None:
            grad_conics = grad_conics.contiguous()

        g_sizes, means, quats, scales, c_sizes = ctx.saved_tensors[:5]
        world_to_cam, projection_matrices, radii, conics = ctx.saved_tensors[5:]

        assert grad_means2d is not None
        assert grad_depths is not None
        assert grad_conics is not None
        d_means, _, d_quats, d_scales, d_w2c = _C.project_gaussians_analytic_jagged_bwd(
            g_sizes,
            means,
            quats,
            scales,
            c_sizes,
            world_to_cam,
            projection_matrices,
            ctx.image_width,
            ctx.image_height,
            ctx.eps2d,
            radii,
            conics,
            grad_means2d,
            grad_depths,
            grad_conics,
            ctx.needs_input_grad[5],
            ctx.ortho,
        )

        return (
            None,
            d_means,
            d_quats,
            d_scales,
            None,
            d_w2c,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
