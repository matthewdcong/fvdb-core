# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
from __future__ import annotations

import ctypes
import importlib.util as _importlib_util
import pathlib

import torch

if torch.cuda.is_available():
    torch.cuda.init()


def _parse_device_string(device_or_device_string: str | torch.device) -> torch.device:
    """
    Parses a device string and returns a torch.device object. For CUDA devices
    without an explicit index, uses the current CUDA device. If the input is a torch.device
    object, it is returned unmodified.

     Args:
         device_string (str | torch.device):
             A device string (e.g., "cpu", "cuda", "cuda:0") or a torch.device object.
             If a string is provided, it should be a valid device identifier.

     Returns:
         torch.device: The parsed device object with proper device index set if a string is passed
         in otherwise returns the input torch.device object.
    """
    if isinstance(device_or_device_string, torch.device):
        return device_or_device_string
    if not isinstance(device_or_device_string, str):
        raise TypeError(f"Expected a string or torch.device, but got {type(device_or_device_string)}")
    device = torch.device(device_or_device_string)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    return device


# Load NanoVDB Editor shared libraries so symbols are globally available before importing the pybind module.
# This helps the dynamic linker resolve dependencies like libpnanovdb*.so when loading fvdb's extensions.
_spec = _importlib_util.find_spec("nanovdb_editor")
if _spec is not None and _spec.origin is not None:
    try:
        _libdir = pathlib.Path(_spec.origin).parent / "lib"
        for _so in sorted(_libdir.glob("libpnanovdb*.so")):
            try:
                ctypes.CDLL(str(_so), mode=ctypes.RTLD_GLOBAL)
            except OSError:
                print(f"Failed to load {_so} from {_libdir}")
                pass
    except Exception:
        print("Failed to load nanovdb_editor from", _libdir)
        pass

# isort: off
from ._fvdb_cpp import (
    NanoVDBGridMetadata,
    config,
    morton,
    hilbert,
)
from ._volume_render import volume_render

# Import JaggedTensor from jagged_tensor.py
from .jagged_tensor import JaggedTensor, jcat
from .grid import Grid
from .grid_batch import GridBatch, gcat
from .grid import Grid
from .attention import scaled_dot_product_attention


from .convolution_plan import ConvolutionPlan
from .gaussian_splatting import GaussianSplat3d, ProjectedGaussianSplats, _evaluate_gaussian_sh
from .enums import CameraModel, ProjectionMethod, RollingShutterType, ShOrderingMode, SmoothingMode


def evaluate_spherical_harmonics(
    sh_degree: int,
    num_cameras: int,
    sh0: torch.Tensor,
    radii: torch.Tensor,
    shN: torch.Tensor | None = None,
    view_directions: torch.Tensor | None = None,
) -> torch.Tensor:
    """Evaluate spherical harmonics to compute view-dependent features/colors.

    Args:
        sh_degree: Degree of spherical harmonics to use (0-3 typically).
        num_cameras: Number of camera views (C).
        sh0: DC term coefficients with shape [N, 1, D].
        radii: Per-axis projected radii with shape [C, N, 2] (int32). A point
               is masked out unless both axes are positive.
        shN: Higher-order SH coefficients with shape [N, K-1, D] where
             K = (sh_degree+1)^2. Required when sh_degree > 0.
        view_directions: Unnormalized view directions with shape [C, N, 3].
                         Required when sh_degree > 0.

    Returns:
        Tensor of shape [C, N, D] containing the evaluated features/colors.
    """
    if sh0.ndim != 3:
        raise ValueError(f"sh0 must be 3-dimensional [N, 1, D], got {sh0.ndim}D")
    if sh0.shape[1] != 1:
        raise ValueError(f"sh0.shape[1] must be 1, got {sh0.shape[1]}")
    if radii.ndim != 3 or radii.shape[-1] != 2:
        raise ValueError(f"radii must be shape [C, N, 2], got {tuple(radii.shape)}")
    if sh0.shape[0] != radii.shape[1]:
        raise ValueError(f"sh0.shape[0] ({sh0.shape[0]}) must match radii.shape[1] ({radii.shape[1]})")
    if sh_degree > 0 and view_directions is None:
        raise ValueError("view_directions is required when sh_degree > 0")
    if view_directions is None:
        view_directions = radii.new_empty(0)
    if shN is None:
        N, _, D = sh0.shape
        shN = sh0.new_empty(N, 0, D)
    return _evaluate_gaussian_sh(sh_degree, num_cameras, view_directions, sh0, shN, radii)


# Import torch-compatible functions that work with both Tensor and JaggedTensor
from .torch_jagged import (
    # Unary operations
    relu,
    relu_,
    sigmoid,
    tanh,
    exp,
    log,
    sqrt,
    floor,
    ceil,
    round,
    nan_to_num,
    clamp,
    # Binary operations
    add,
    sub,
    mul,
    true_divide,
    floor_divide,
    remainder,
    pow,
    maximum,
    minimum,
    # Comparisons
    eq,
    ne,
    lt,
    le,
    gt,
    ge,
    where,
    # Reductions
    sum,
    mean,
    amax,
    amin,
    argmax,
    argmin,
    all,
    any,
    norm,
    var,
    std,
)

# The following import needs to come after all classes and functions are defined
# in order to avoid a circular dependency error.
# Make these available without an explicit submodule import
from . import functional, nn, utils, version, viz
from .version import __version__

__version_info__ = tuple(int(x) for x in __version__.split(".")[:3])

__all__ = [
    # Core classes
    "Grid",
    "GridBatch",
    "JaggedTensor",
    "GaussianSplat3d",
    "ProjectedGaussianSplats",
    "CameraModel",
    "ProjectionMethod",
    "RollingShutterType",
    "ShOrderingMode",
    "SmoothingMode",
    "ConvolutionPlan",
    "NanoVDBGridMetadata",
    # Concatenation of jagged tensors or grid/grid batches
    "jcat",
    "gcat",
    # Morton/Hilbert operations
    "morton",
    "hilbert",
    # Specialized operations
    "scaled_dot_product_attention",
    "volume_render",
    "evaluate_spherical_harmonics",
    # Torch-compatible functions (work with both Tensor and JaggedTensor)
    "relu",
    "relu_",
    "sigmoid",
    "tanh",
    "exp",
    "log",
    "sqrt",
    "floor",
    "ceil",
    "round",
    "nan_to_num",
    "clamp",
    "add",
    "sub",
    "mul",
    "true_divide",
    "floor_divide",
    "remainder",
    "pow",
    "maximum",
    "minimum",
    "eq",
    "ne",
    "lt",
    "le",
    "gt",
    "ge",
    "where",
    "sum",
    "mean",
    "amax",
    "amin",
    "argmax",
    "argmin",
    "all",
    "any",
    "norm",
    "var",
    "std",
    # Config
    "config",
    # Submodules
    "functional",
    "version",
    "viz",
    "nn",
    "utils",
]
