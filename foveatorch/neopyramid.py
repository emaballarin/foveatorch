#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#  Copyright (c) 2018-2023 The Kornia Developers (original implementation,
#                see: https://github.com/kornia/kornia/blob/master/kornia/geometry/transform/pyramid.py)
#            (c) 2024 Emanuele Ballarin <emanuele@ballarin.cc>
#
#  Originally released under the terms of the Apache License, Version 2.0
#  (see: https://github.com/kornia/kornia/blob/master/LICENSE)
#
#  Released under the terms of the MIT License
#  (see: https://url.ballarin.cc/mitlicense)
#
# ------------------------------------------------------------------------------
from __future__ import annotations

import torch as th
import torch.nn.functional as F
from kornia.core import Device
from kornia.core import Dtype
from kornia.core import Tensor
from kornia.core.check import KORNIA_CHECK
from kornia.core.check import KORNIA_CHECK_SHAPE
from kornia.filters import filter2d
from kornia.filters import get_gaussian_kernel2d

__all__ = [
    "pyrdown",
    "pyrup",
    "pyramid_build_down",
    "pyramid_scale_up",
    "retina_pyramid",
    "retina_pyramid_blur",
]


def pyrdown(
    xinput: Tensor,
    kernel: Tensor,
    border_type: str = "reflect",
    align_corners: bool = False,
    factor: float = 2.0,
) -> Tensor:
    r"""Blur a tensor and downsample it.

    Args:
        xinput: tensor to be downsampled.
        kernel: kernel to be used for blurring the input.
        border_type: padding mode to be applied before convolving.
          Expected modes are: ``'constant'``, ``'reflect'``,
          ``'replicate'`` or ``'circular'``.
        align_corners: interpolation flag.
        factor: downsampling factor.

    Return:
        downsampled tensor.
    """
    KORNIA_CHECK_SHAPE(xinput, ["B", "C", "H", "W"])

    # Blur, then Downsample
    out: Tensor = F.interpolate(
        input=filter2d(xinput, kernel, border_type),
        scale_factor=1.0 / factor,
        mode="bilinear",
        align_corners=align_corners,
    )
    return out


def pyrup(
    xinput: Tensor,
    kernel: Tensor,
    border_type: str = "reflect",
    align_corners: bool = False,
    factor: float = 2.0,
    force_size: tuple[int, int] | None = None,
) -> Tensor:
    r"""Upsample a tensor and then blur it.

    Args:
        xinput: tensor to be upsampled.
        kernel: kernel to be used for blurring the input.
        border_type: padding mode to be applied before convolving.
          Expected modes are: ``'constant'``, ``'reflect'``, ``'replicate'`` or ``'circular'``.
        align_corners: interpolation flag.
        factor: upsampling factor.
        force_size: size of the output image, HxW; optional.

    Return:
        upsampled tensor.
    """
    KORNIA_CHECK_SHAPE(xinput, ["B", "C", "H", "W"])

    # Upsample
    if force_size is None:
        x_up: Tensor = F.interpolate(
            input=xinput,
            scale_factor=factor,
            mode="bilinear",
            align_corners=align_corners,
        )
    else:
        x_up: Tensor = F.interpolate(
            input=xinput,
            size=force_size,
            mode="bilinear",
            align_corners=align_corners,
        )

    # then Blur
    x_blur: Tensor = filter2d(x_up, kernel, border_type)
    return x_blur


def pyramid_build_down(
    xinput: Tensor,
    kernel: Tensor,
    max_level: int,
    border_type: str = "reflect",
    align_corners: bool = False,
    factor: float = 2.0,
) -> list[Tensor]:
    r"""Construct the (downward) Gaussian pyramid for a tensor image.

    The function constructs an iterable of images and builds the (downward)
    Gaussian pyramid by recursively applying pyrDown to the previously built
    pyramid layers.

    Args:
        xinput : tensor to be used to construct the pyramid.
        kernel: kernel to be used for blurring the input.
        max_level: 0-based index of the last (the smallest) pyramid layer.
          It must be non-negative.
        border_type: padding mode to be applied before convolving.
          Expected modes are: ``'constant'``, ``'reflect'``,
          ``'replicate'`` or ``'circular'``.
        align_corners: interpolation flag.
        factor: downsampling factor.

    Shape:
        - Input: :math:`(B, C, H, W)`
        - Output :math:`[(B, C, H, W), (B, C, H/factor, W/factor), ...]`
    """
    KORNIA_CHECK_SHAPE(xinput, ["B", "C", "H", "W"])
    KORNIA_CHECK(
        isinstance(max_level, int) or max_level < 0,
        f"Invalid iterations, it must be a positive integer. Got: {max_level}",
    )

    # Trivial pyramid with the original image only
    pyramid: list[Tensor] = [xinput]

    # Iteration and Downsampling
    _: int
    for _ in range(max_level - 1):
        pyramid.append(pyrdown(pyramid[-1], kernel, border_type, align_corners, factor))

    return pyramid


def pyramid_scale_up(
    xinput: Tensor,
    kernel: Tensor,
    iterations: int,
    border_type: str = "reflect",
    align_corners: bool = False,
    factor: float = 2.0,
    force_size: tuple[int, int] | None = None,
) -> Tensor:
    r"""Sequentially upscale a tensor image, through the (upward) Gaussian pyramid.

    The function recursively applies pyrUp to the previously built pyramid
    layers.

    Args:
        xinput : input tensor image.
        kernel: kernel to be used for blurring the input.
        iterations: 0-based index of the last (the largest) pyramid layer.
          It must be non-negative.
        border_type: padding mode to be applied before convolving.
          The expected modes are: ``'constant'``, ``'reflect'``,
          ``'replicate'`` or ``'circular'``.
        align_corners: interpolation flag.
        factor: upsampling factor.
        force_size: size of the output image, HxW; optional.

    Shape:
        - Input: :math:`(B, C, H, W)`
        - Output :math:`[(B, C, H, W), (B, C, factor * H, factor * W), ...]`
    """
    KORNIA_CHECK_SHAPE(xinput, ["B", "C", "H", "W"])
    KORNIA_CHECK(
        isinstance(iterations, int) or iterations < 0,
        f"Invalid iterations, it must be a positive integer. Got: {iterations}",
    )

    # Initialisation with the original image
    xoutput: Tensor = xinput

    # Iteration and Upsampling
    itidx: int
    for itidx in range(iterations):
        xoutput: Tensor = pyrup(
            xoutput,
            kernel,
            border_type,
            align_corners,
            factor,
            force_size=force_size if itidx == iterations - 1 else None,
        )

    return xoutput


def retina_pyramid(
    xinput: Tensor,
    kernel_size: int = 5,
    sigma: float = 1.0,
    nlayers: int = 6,
    border_type: str = "reflect",
    align_corners: bool = False,
    factor: float = 2.0,
    device: Device | None = None,
    dtype: Dtype | None = None,
) -> list[Tensor]:
    r"""Build the retinal transform (pyramid) from a tensor image.

    Args:
        xinput : input tensor image.
        kernel_size: size of the kernel to be used for blurring the input. It must be positive.
        sigma: standard deviation of the Gaussian kernel to be used for
          blurring the input.
        nlayers: number of layers of the pyramid to be built. It must be non-negative.
        border_type: padding mode to be applied before convolving.
          Expected modes are: ``'constant'``, ``'reflect'``,
          ``'replicate'`` or ``'circular'``.
        align_corners: interpolation flag.
        factor: upsampling factor.
        device: Device desired to compute; optional.
        dtype: Dtype desired for compute; optional.

    Shape:
        - Input: :math:`(B, C, H, W)`
        - Output :math:`[(B, C, H, W), ...]`
    """
    KORNIA_CHECK_SHAPE(xinput, ["B", "C", "H", "W"])
    KORNIA_CHECK(
        isinstance(nlayers, int) or nlayers < 0,
        f"Invalid iterations, it must be a positive integer. Got: {nlayers}",
    )
    KORNIA_CHECK(
        isinstance(kernel_size, int) or kernel_size <= 0,
        f"Invalid kernel size, it must be a positive integer. Got: {kernel_size}",
    )

    # Select device
    if device is None:
        device = "cuda" if th.cuda.is_available() else "cpu"
    else:
        if device == "cuda" and not th.cuda.is_available():
            raise RuntimeError("CUDA is not available.")

    # Define the kernel
    kernel: Tensor = get_gaussian_kernel2d(
        (kernel_size, kernel_size), (sigma, sigma), dtype=dtype, device=device
    )

    # Build the downward pyramid
    pyramid: list[Tensor] = pyramid_build_down(
        xinput,
        kernel,
        max_level=nlayers,
        border_type=border_type,
        align_corners=align_corners,
        factor=factor,
    )

    # Upscale the layers
    layer_idx: int
    for layer_idx in range(1, nlayers):
        pyramid[layer_idx] = pyramid_scale_up(
            pyramid[layer_idx],
            kernel,
            iterations=layer_idx,
            border_type=border_type,
            align_corners=align_corners,
            factor=factor,
            force_size=xinput.shape[-2:],
        )

    return pyramid


def retina_pyramid_blur(
    xinput: Tensor,
    pyramid_level: int,
    kernel_size: int = 5,
    sigma: float = 1,
    nlayers: int = 6,
    border_type: str = "reflect",
    align_corners: bool = False,
    factor: float = 2.0,
    device: Device | None = None,
    dtype: Dtype | None = None,
) -> Tensor:
    return retina_pyramid(
        xinput=xinput,
        kernel_size=kernel_size,
        sigma=sigma,
        nlayers=nlayers,
        border_type=border_type,
        align_corners=align_corners,
        factor=factor,
        device=device,
        dtype=dtype,
    )[pyramid_level]
