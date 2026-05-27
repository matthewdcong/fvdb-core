# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""Python torch.autograd.Function wrappers for Gaussian splatting dispatch functions.

This module replaces the C++ autograd layer (detail/autograd/) with equivalent
Python autograd wrappers that call the raw dispatch forward/backward functions
exposed through pybind11. This is the first step in the migration from C++ to
Python autograd.
"""

from __future__ import annotations

from typing import Any, cast

import gsplat  # noqa: F401  registers torch.ops.gsplat
import torch

from . import _fvdb_cpp as _C
from ._fvdb_cpp import JaggedTensor as JaggedTensorCpp
from .jagged_tensor import JaggedTensor

_FVDB_SH_BIAS = 0.5  # 3DGS color convention: SH output is added on top of mid-gray

# CameraModelType enum values registered by gsplat — pinhole=0, ortho=1. We
# only route the pinhole and ortho paths through gsplat; OpenCV / fisheye
# models stay on fvdb's unscented-transform path.
_GSPLAT_CAMERA_MODEL_PINHOLE = 0
_GSPLAT_CAMERA_MODEL_ORTHO = 1


def _sh_bridge_inputs(
    num_cameras: int,
    num_gaussians: int,
    view_dirs: torch.Tensor,
    sh0_or_dummy: torch.Tensor,
    shN_coeffs: torch.Tensor,
    radii: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Reshape fvdb's separated-coeffs / per-camera-dirs API into gsplat's
    combined-coeffs API.

    Returns (dirs[C,N,3], coeffs[N,K,3], masks[C,N] bool or None).
    """
    C, N = num_cameras, num_gaussians
    coeffs = torch.cat([sh0_or_dummy, shN_coeffs], dim=1)  # [N, K, D]

    if view_dirs.numel() == 0:
        dirs = coeffs.new_zeros(C, N, 3)
    else:
        dirs = view_dirs.contiguous()

    if radii.numel() > 0:
        masks = (radii > 0).all(dim=-1).contiguous()
    else:
        masks = None
    return dirs, coeffs, masks


# ---------------------------------------------------------------------------
#  Projection (analytic)
# ---------------------------------------------------------------------------


class _ProjectGaussiansFn(torch.autograd.Function):
    """Python autograd wrapper for the analytic Gaussian projection forward/backward dispatch."""

    @staticmethod
    def forward(
        ctx,
        means: torch.Tensor,
        quats: torch.Tensor,
        log_scales: torch.Tensor,
        world_to_cam: torch.Tensor,
        projection_matrices: torch.Tensor,
        image_width: int,
        image_height: int,
        eps2d: float,
        near: float,
        far: float,
        min_radius_2d: float,
        calc_compensations: bool,
        ortho: bool,
        accum_grad_norms: torch.Tensor | None = None,
        accum_step_counts: torch.Tensor | None = None,
        accum_max_radii: torch.Tensor | None = None,
    ):
        scales = log_scales.exp()
        camera_model_type = _GSPLAT_CAMERA_MODEL_ORTHO if ortho else _GSPLAT_CAMERA_MODEL_PINHOLE
        radii, means2d, depths, conics, compensations = torch.ops.gsplat.projection_ewa_3dgs_fused(
            means,
            None,  # covars — quats/scales path
            quats,
            scales,
            None,  # opacities — only used for AccuTile, not needed here
            world_to_cam,
            projection_matrices,
            image_width,
            image_height,
            eps2d,
            near,
            far,
            min_radius_2d,
            calc_compensations,
            camera_model_type,
        )
        if not calc_compensations:
            compensations = None

        to_save = [means, quats, log_scales, world_to_cam, projection_matrices, radii, conics]
        if compensations is not None:
            to_save.append(compensations)
        ctx.save_for_backward(*to_save)

        ctx.image_width = image_width
        ctx.image_height = image_height
        ctx.eps2d = eps2d
        ctx.calc_compensations = calc_compensations
        ctx.camera_model_type = camera_model_type
        ctx.accum_grad_norms = accum_grad_norms
        ctx.accum_step_counts = accum_step_counts
        ctx.accum_max_radii = accum_max_radii

        if compensations is not None:
            return radii, means2d, depths, conics, compensations
        return radii, means2d, depths, conics

    @staticmethod
    def backward(ctx: Any, *grad_outputs: torch.Tensor | None) -> tuple[torch.Tensor | None, ...]:
        grad_means2d = grad_outputs[1]
        grad_depths = grad_outputs[2]
        grad_conics = grad_outputs[3]
        maybe_grad_comp = grad_outputs[4:]
        if grad_means2d is not None:
            grad_means2d = grad_means2d.contiguous()
        if grad_depths is not None:
            grad_depths = grad_depths.contiguous()
        if grad_conics is not None:
            grad_conics = grad_conics.contiguous()

        grad_compensations: torch.Tensor | None = None
        if ctx.calc_compensations and maybe_grad_comp:
            gc = maybe_grad_comp[0]
            if gc is not None:
                grad_compensations = gc.contiguous()

        saved = ctx.saved_tensors
        means = saved[0]
        quats = saved[1]
        log_scales = saved[2]
        world_to_cam = saved[3]
        projection_matrices = saved[4]
        radii = saved[5]
        conics = saved[6]
        compensations = saved[7] if ctx.calc_compensations else None

        assert grad_means2d is not None
        assert grad_depths is not None
        assert grad_conics is not None
        scales = log_scales.exp()
        viewmats_requires_grad = ctx.needs_input_grad[3]
        d_means, _v_covars, d_quats, d_scales_raw, d_w2c = torch.ops.gsplat.projection_ewa_3dgs_fused_bwd(
            means,
            None,  # covars
            quats,
            scales,
            world_to_cam,
            projection_matrices,
            ctx.image_width,
            ctx.image_height,
            ctx.eps2d,
            ctx.camera_model_type,
            radii,
            conics,
            compensations,
            grad_means2d,
            grad_depths,
            grad_conics,
            grad_compensations,
            viewmats_requires_grad,
        )
        # gsplat emits dL/d(scale); fvdb's API returns dL/d(log_scale).
        # Chain rule: dL/d(log s) = dL/d(s) * s.
        d_log_scales = d_scales_raw * scales

        # Accumulator side-effects — fvdb's bwd kernel folded these in. With
        # gsplat we recompute them in Python from grad_means2d + per-axis radii.
        if ctx.accum_grad_norms is not None and ctx.accum_step_counts is not None:
            visible = (radii > 0).all(dim=-1)  # [C, N]
            C_total = visible.size(0)
            sx = grad_means2d[..., 0] * (float(ctx.image_width) * 0.5 * C_total)
            sy = grad_means2d[..., 1] * (float(ctx.image_height) * 0.5 * C_total)
            norm = torch.sqrt(sx * sx + sy * sy)
            masked_norm = torch.where(visible, norm, norm.new_zeros(()))
            ctx.accum_grad_norms.add_(masked_norm.sum(dim=0).to(ctx.accum_grad_norms.dtype))
            ctx.accum_step_counts.add_(visible.sum(dim=0).to(ctx.accum_step_counts.dtype))
            if ctx.accum_max_radii is not None:
                max_r = radii.amax(dim=-1).amax(dim=0).to(ctx.accum_max_radii.dtype)
                ctx.accum_max_radii.copy_(torch.maximum(ctx.accum_max_radii, max_r))

        return (
            d_means,
            d_quats,
            d_log_scales,
            d_w2c,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


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


# ---------------------------------------------------------------------------
#  Spherical harmonics evaluation
# ---------------------------------------------------------------------------


class _EvaluateGaussianSHFn(torch.autograd.Function):
    """Python autograd wrapper for the SH evaluation forward/backward dispatch."""

    @staticmethod
    def forward(
        ctx,
        sh_degree_to_use: int,
        num_cameras: int,
        view_dirs: torch.Tensor,  # [C, N, 3] or empty
        sh0_coeffs: torch.Tensor,  # [N, 1, D]
        shN_coeffs: torch.Tensor,  # [N, K-1, D] or empty
        radii: torch.Tensor,  # [C, N, 2]
    ) -> torch.Tensor:
        N = sh0_coeffs.size(0)
        dirs, coeffs, masks = _sh_bridge_inputs(num_cameras, N, view_dirs, sh0_coeffs, shN_coeffs, radii)
        raw = torch.ops.gsplat.spherical_harmonics(sh_degree_to_use, dirs, coeffs, masks)
        # gsplat returns the unbiased SH evaluation and leaves masked rows
        # uninitialized; fvdb's convention is +0.5 on valid rows, 0 on masked.
        if masks is not None:
            render_quantities = torch.where(masks.unsqueeze(-1), raw + _FVDB_SH_BIAS, raw.new_zeros(()))
        else:
            render_quantities = raw + _FVDB_SH_BIAS

        ctx.save_for_backward(view_dirs, shN_coeffs, radii)
        ctx.sh_degree_to_use = sh_degree_to_use
        ctx.num_cameras = num_cameras
        ctx.num_gaussians = sh0_coeffs.size(0)

        return render_quantities

    @staticmethod
    def backward(ctx: Any, *grad_outputs: torch.Tensor | None) -> tuple[torch.Tensor | None, ...]:
        d_loss_d_colors = grad_outputs[0]
        if d_loss_d_colors is None:
            return (None, None, None, None, None, None)
        d_loss_d_colors = d_loss_d_colors.contiguous()

        view_dirs, shN_coeffs, radii = ctx.saved_tensors

        compute_d_loss_d_view_dirs = view_dirs.numel() > 0 and view_dirs.requires_grad

        C, N = ctx.num_cameras, ctx.num_gaussians
        D = d_loss_d_colors.size(2)
        K = shN_coeffs.size(1) + 1

        # The SH backward depends on shN only for v_dirs; the DC term has no
        # x/y/z dependence. We feed a zero placeholder for sh0 so the combined
        # coeffs tensor has the [N, K, 3] layout gsplat expects.
        sh0_dummy = shN_coeffs.new_zeros(N, 1, D)
        dirs, coeffs, masks = _sh_bridge_inputs(C, N, view_dirs, sh0_dummy, shN_coeffs, radii)
        gsplat_compute_v_dirs = compute_d_loss_d_view_dirs and K > 1
        v_coeffs, v_dirs = torch.ops.gsplat.spherical_harmonics_bwd(
            ctx.sh_degree_to_use,
            dirs,
            coeffs,
            masks,
            d_loss_d_colors,
            gsplat_compute_v_dirs,
        )

        # v_coeffs is [N, K, D] — gsplat already accumulated across cameras.
        d_sh0 = v_coeffs[:, 0:1, :].contiguous()
        if K > 1:
            d_shN: torch.Tensor | None = v_coeffs[:, 1:, :].contiguous()
            d_view_dirs: torch.Tensor | None = v_dirs if gsplat_compute_v_dirs else None
        else:
            d_shN = None
            d_view_dirs = None

        return (None, None, d_view_dirs, d_sh0, d_shN, None)


# ---------------------------------------------------------------------------
#  Dense screen-space rasterization
# ---------------------------------------------------------------------------


class _RasterizeScreenSpaceGaussiansFn(torch.autograd.Function):
    """Python autograd wrapper for the dense Gaussian rasterization forward/backward dispatch."""

    @staticmethod
    def forward(
        ctx,
        means2d: torch.Tensor,
        conics: torch.Tensor,
        colors: torch.Tensor,
        opacities: torch.Tensor,
        image_width: int,
        image_height: int,
        image_origin_w: int,
        image_origin_h: int,
        tile_size: int,
        tile_offsets: torch.Tensor,
        tile_gaussian_ids: torch.Tensor,
        absgrad: bool,
        backgrounds: torch.Tensor | None,
        masks: torch.Tensor | None,
    ):
        rendered_colors, rendered_alphas, last_ids = torch.ops.gsplat.rasterize_to_pixels_3dgs(
            means2d,
            conics,
            colors,
            opacities,
            backgrounds,
            masks,
            image_width,
            image_height,
            tile_size,
            tile_offsets,
            tile_gaussian_ids,
        )

        # gsplat writes ``backgrounds`` into masked-tile pixels in the colors
        # buffer but leaves ``rendered_alphas`` uninitialized — fvdb's
        # contract is alpha==0 on masked tiles. Expand the tile mask to per
        # pixel and zero those rows.
        if masks is not None:
            C_, tH, tW = masks.shape
            pixel_mask = (
                masks[:, :, None, :, None]
                .expand(C_, tH, tile_size, tW, tile_size)
                .reshape(C_, tH * tile_size, tW * tile_size)[:, :image_height, :image_width]
            )
            rendered_alphas = torch.where(pixel_mask.unsqueeze(-1), rendered_alphas, rendered_alphas.new_zeros(()))

        to_save = [means2d, conics, colors, opacities, tile_offsets, tile_gaussian_ids, rendered_alphas, last_ids]
        if backgrounds is not None:
            to_save.append(backgrounds)
            ctx.has_backgrounds = True
        else:
            ctx.has_backgrounds = False
        if masks is not None:
            to_save.append(masks)
            ctx.has_masks = True
        else:
            ctx.has_masks = False
        ctx.save_for_backward(*to_save)

        ctx.image_width = image_width
        ctx.image_height = image_height
        ctx.image_origin_w = image_origin_w
        ctx.image_origin_h = image_origin_h
        ctx.tile_size = tile_size
        ctx.absgrad = absgrad

        return rendered_colors, rendered_alphas

    @staticmethod
    def backward(ctx: Any, *grad_outputs: torch.Tensor | None) -> tuple[torch.Tensor | None, ...]:
        d_loss_d_rendered_colors = grad_outputs[0]
        d_loss_d_rendered_alphas = grad_outputs[1]
        if d_loss_d_rendered_colors is not None:
            d_loss_d_rendered_colors = d_loss_d_rendered_colors.contiguous()
        if d_loss_d_rendered_alphas is not None:
            d_loss_d_rendered_alphas = d_loss_d_rendered_alphas.contiguous()

        saved = ctx.saved_tensors
        means2d, conics, colors, opacities = saved[0], saved[1], saved[2], saved[3]
        tile_offsets, tile_gaussian_ids = saved[4], saved[5]
        rendered_alphas, last_ids = saved[6], saved[7]

        backgrounds: torch.Tensor | None = None
        masks: torch.Tensor | None = None
        opt_idx = 8
        if ctx.has_backgrounds:
            backgrounds = saved[opt_idx]
            opt_idx += 1
        if ctx.has_masks:
            masks = saved[opt_idx]

        assert d_loss_d_rendered_colors is not None
        assert d_loss_d_rendered_alphas is not None
        # gsplat's bwd emits an extra "absgrad" gradient slot for means2d.
        _v_means2d_abs, d_means2d, d_conics, d_colors, d_opacities = torch.ops.gsplat.rasterize_to_pixels_3dgs_bwd(
            means2d,
            conics,
            colors,
            opacities,
            backgrounds,
            masks,
            ctx.image_width,
            ctx.image_height,
            ctx.tile_size,
            tile_offsets,
            tile_gaussian_ids,
            rendered_alphas,
            last_ids,
            d_loss_d_rendered_colors,
            d_loss_d_rendered_alphas,
            ctx.absgrad,
        )

        return (
            d_means2d,
            d_conics,
            d_colors,
            d_opacities,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


# ---------------------------------------------------------------------------
#  Sparse screen-space rasterization
# ---------------------------------------------------------------------------


class _RasterizeScreenSpaceGaussiansSparseFn(torch.autograd.Function):
    """Python autograd wrapper for the sparse Gaussian rasterization forward/backward dispatch."""

    @staticmethod
    def forward(
        ctx,
        means2d: torch.Tensor,
        conics: torch.Tensor,
        features: torch.Tensor,
        opacities: torch.Tensor,
        pixels_to_render: JaggedTensor,
        image_width: int,
        image_height: int,
        image_origin_w: int,
        image_origin_h: int,
        tile_size: int,
        tile_offsets: torch.Tensor,
        tile_gaussian_ids: torch.Tensor,
        active_tiles: torch.Tensor,
        tile_pixel_mask: torch.Tensor,
        tile_pixel_cumsum: torch.Tensor,
        pixel_map: torch.Tensor,
        absgrad: bool,
        backgrounds: torch.Tensor | None,
        masks: torch.Tensor | None,
    ):
        result = _C.rasterize_screen_space_gaussians_sparse_fwd(
            pixels_to_render._impl,
            means2d,
            conics,
            features,
            opacities,
            image_width,
            image_height,
            image_origin_w,
            image_origin_h,
            tile_size,
            tile_offsets,
            tile_gaussian_ids,
            active_tiles,
            tile_pixel_mask,
            tile_pixel_cumsum,
            pixel_map,
            -1,
            backgrounds,
            masks,
        )
        rendered_colors_jt = JaggedTensor(impl=result[0])
        rendered_alphas_jt = JaggedTensor(impl=result[1])
        last_ids_jt = JaggedTensor(impl=result[2])

        joffsets = pixels_to_render.joffsets
        jidx = pixels_to_render.jidx
        jlidx = pixels_to_render.jlidx

        to_save = [
            means2d,
            conics,
            features,
            opacities,
            tile_offsets,
            tile_gaussian_ids,
            pixels_to_render.jdata,
            rendered_colors_jt.jdata,
            rendered_alphas_jt.jdata,
            last_ids_jt.jdata,
            joffsets,
            jidx,
            jlidx,
            active_tiles,
            tile_pixel_mask,
            tile_pixel_cumsum,
            pixel_map,
        ]
        if backgrounds is not None:
            to_save.append(backgrounds)
            ctx.has_backgrounds = True
        else:
            ctx.has_backgrounds = False
        if masks is not None:
            to_save.append(masks)
            ctx.has_masks = True
        else:
            ctx.has_masks = False
        ctx.save_for_backward(*to_save)

        ctx.image_width = image_width
        ctx.image_height = image_height
        ctx.image_origin_w = image_origin_w
        ctx.image_origin_h = image_origin_h
        ctx.tile_size = tile_size
        ctx.absgrad = absgrad
        ctx.num_outer_lists = len(pixels_to_render)

        return rendered_colors_jt.jdata, rendered_alphas_jt.jdata

    @staticmethod
    def backward(ctx: Any, *grad_outputs: torch.Tensor | None) -> tuple[torch.Tensor | None, ...]:
        d_loss_d_rendered_features_jdata = grad_outputs[0]
        d_loss_d_rendered_alphas_jdata = grad_outputs[1]
        if d_loss_d_rendered_features_jdata is not None:
            d_loss_d_rendered_features_jdata = d_loss_d_rendered_features_jdata.contiguous()
        if d_loss_d_rendered_alphas_jdata is not None:
            d_loss_d_rendered_alphas_jdata = d_loss_d_rendered_alphas_jdata.contiguous()

        saved = ctx.saved_tensors
        means2d, conics, features, opacities = saved[0], saved[1], saved[2], saved[3]
        tile_offsets, tile_gaussian_ids = saved[4], saved[5]
        pixels_jdata = saved[6]
        rendered_alphas_jdata = saved[8]
        last_ids_jdata = saved[9]
        joffsets, jidx, jlidx = saved[10], saved[11], saved[12]
        active_tiles = saved[13]
        tile_pixel_mask, tile_pixel_cumsum, pixel_map = saved[14], saved[15], saved[16]

        backgrounds: torch.Tensor | None = None
        masks: torch.Tensor | None = None
        opt_idx = 17
        if ctx.has_backgrounds:
            backgrounds = saved[opt_idx]
            opt_idx += 1
        if ctx.has_masks:
            masks = saved[opt_idx]

        pixels_jt = JaggedTensor(impl=_C.JaggedTensor.from_data_offsets_and_list_ids(pixels_jdata, joffsets, jlidx))
        rendered_alphas_jt = pixels_jt.jagged_like(rendered_alphas_jdata)
        last_ids_jt = pixels_jt.jagged_like(last_ids_jdata)
        assert d_loss_d_rendered_features_jdata is not None
        assert d_loss_d_rendered_alphas_jdata is not None
        d_loss_d_rendered_features_jt = pixels_jt.jagged_like(d_loss_d_rendered_features_jdata)
        d_loss_d_rendered_alphas_jt = pixels_jt.jagged_like(d_loss_d_rendered_alphas_jdata)

        result = _C.rasterize_screen_space_gaussians_sparse_bwd(
            pixels_jt._impl,
            means2d,
            conics,
            features,
            opacities,
            ctx.image_width,
            ctx.image_height,
            ctx.image_origin_w,
            ctx.image_origin_h,
            ctx.tile_size,
            tile_offsets,
            tile_gaussian_ids,
            rendered_alphas_jt._impl,
            last_ids_jt._impl,
            d_loss_d_rendered_features_jt._impl,
            d_loss_d_rendered_alphas_jt._impl,
            active_tiles,
            tile_pixel_mask,
            tile_pixel_cumsum,
            pixel_map,
            ctx.absgrad,
            -1,
            backgrounds,
            masks,
        )
        d_means2d = result[1]
        d_conics = result[2]
        d_colors = result[3]
        d_opacities = result[4]

        return (
            d_means2d,
            d_conics,
            d_colors,
            d_opacities,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


# ---------------------------------------------------------------------------
#  World-space rasterization
# ---------------------------------------------------------------------------


class _RasterizeWorldSpaceGaussiansFn(torch.autograd.Function):
    """Python autograd wrapper for world-space Gaussian rasterization forward/backward dispatch."""

    @staticmethod
    def forward(
        ctx,
        means: torch.Tensor,
        quats: torch.Tensor,
        log_scales: torch.Tensor,
        features: torch.Tensor,
        opacities: torch.Tensor,
        world_to_cam_start: torch.Tensor,
        world_to_cam_end: torch.Tensor,
        projection_matrices: torch.Tensor,
        distortion_coeffs: torch.Tensor,
        rolling_shutter_type: int,
        camera_model: int,
        image_width: int,
        image_height: int,
        image_origin_w: int,
        image_origin_h: int,
        tile_size: int,
        tile_offsets: torch.Tensor,
        tile_gaussian_ids: torch.Tensor,
        backgrounds: torch.Tensor | None,
        masks: torch.Tensor | None,
    ):
        result = _C.rasterize_world_space_gaussians_fwd(
            means,
            quats,
            log_scales,
            features,
            opacities,
            world_to_cam_start,
            world_to_cam_end,
            projection_matrices,
            distortion_coeffs,
            _C.RollingShutterType(rolling_shutter_type),
            _C.CameraModel(camera_model),
            image_width,
            image_height,
            image_origin_w,
            image_origin_h,
            tile_size,
            tile_offsets,
            tile_gaussian_ids,
            backgrounds,
            masks,
        )
        rendered_features = result[0]
        rendered_alphas = result[1]
        last_ids = result[2]

        to_save = [
            means,
            quats,
            log_scales,
            features,
            opacities,
            world_to_cam_start,
            world_to_cam_end,
            projection_matrices,
            distortion_coeffs,
            tile_offsets,
            tile_gaussian_ids,
            rendered_alphas,
            last_ids,
        ]
        if backgrounds is not None:
            to_save.append(backgrounds)
            ctx.has_backgrounds = True
        else:
            ctx.has_backgrounds = False
        if masks is not None:
            to_save.append(masks)
            ctx.has_masks = True
        else:
            ctx.has_masks = False
        ctx.save_for_backward(*to_save)

        ctx.image_width = image_width
        ctx.image_height = image_height
        ctx.image_origin_w = image_origin_w
        ctx.image_origin_h = image_origin_h
        ctx.tile_size = tile_size
        ctx.rolling_shutter_type = rolling_shutter_type
        ctx.camera_model = camera_model

        return rendered_features, rendered_alphas

    @staticmethod
    def backward(ctx: Any, *grad_outputs: torch.Tensor | None) -> tuple[torch.Tensor | None, ...]:
        d_loss_d_rendered_features = grad_outputs[0]
        d_loss_d_rendered_alphas = grad_outputs[1]
        if d_loss_d_rendered_features is not None:
            d_loss_d_rendered_features = d_loss_d_rendered_features.contiguous()
        if d_loss_d_rendered_alphas is not None:
            d_loss_d_rendered_alphas = d_loss_d_rendered_alphas.contiguous()

        saved = ctx.saved_tensors
        means, quats, log_scales = saved[0], saved[1], saved[2]
        features, opacities = saved[3], saved[4]
        world_to_cam_start, world_to_cam_end = saved[5], saved[6]
        projection_matrices, distortion_coeffs = saved[7], saved[8]
        tile_offsets, tile_gaussian_ids = saved[9], saved[10]
        rendered_alphas, last_ids = saved[11], saved[12]

        backgrounds: torch.Tensor | None = None
        masks: torch.Tensor | None = None
        opt_idx = 13
        if ctx.has_backgrounds:
            backgrounds = saved[opt_idx]
            opt_idx += 1
        if ctx.has_masks:
            masks = saved[opt_idx]

        assert d_loss_d_rendered_features is not None
        assert d_loss_d_rendered_alphas is not None
        result = _C.rasterize_world_space_gaussians_bwd(
            means,
            quats,
            log_scales,
            features,
            opacities,
            world_to_cam_start,
            world_to_cam_end,
            projection_matrices,
            distortion_coeffs,
            _C.RollingShutterType(ctx.rolling_shutter_type),
            _C.CameraModel(ctx.camera_model),
            ctx.image_width,
            ctx.image_height,
            ctx.image_origin_w,
            ctx.image_origin_h,
            ctx.tile_size,
            tile_offsets,
            tile_gaussian_ids,
            rendered_alphas,
            last_ids,
            d_loss_d_rendered_features,
            d_loss_d_rendered_alphas,
            backgrounds,
            masks,
        )
        d_means = result[0]
        d_quats = result[1]
        d_log_scales = result[2]
        d_features = result[3]
        d_opacities = result[4]

        return (
            d_means,
            d_quats,
            d_log_scales,
            d_features,
            d_opacities,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
