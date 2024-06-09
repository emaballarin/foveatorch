#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#  Copyright (c) 2018-2023 Zhibo Yang (CV2/NumPy original,
#                see: https://github.com/ouyangzhibo/Image_Foveation_Python/blob/master/retina_transform.py)
#            (c) 2024 Emanuele Ballarin <emanuele@ballarin.cc>
#
#  Originally released under the terms of the MIT License
#  (see: https://github.com/ouyangzhibo/Image_Foveation_Python/blob/master/LICENSE)
#
#  Released under the terms of the MIT License
#  (see: https://url.ballarin.cc/mitlicense)
#
# ------------------------------------------------------------------------------
from __future__ import annotations

import torch as th
from kornia.core import Device
from kornia.core import Tensor
from kornia.core.check import KORNIA_CHECK
from kornia.core.check import KORNIA_CHECK_SHAPE
from torch import logical_and

from .neopyramid import retina_pyramid
from .neopyramid import retina_pyramid_blur

# ------------------------------------------------------------------------------

__all__ = ["foveate_atomic", "foveate", "Foveate2D", "FovealPyramidBlur"]


def foveate_atomic(
    xinput: Tensor,
    fixation_pts: Tensor | None = None,
    nlayers: int = 6,
    p: float = 7.5,
    alpha: float = 2.5,
    kernel_size: int = 5,
    sigma: float = 0.248,
    k: float = 1.0,
    eps: float = 1e-5,
    device: Device | None = None,
) -> Tensor:
    r"""Foveate an image via the retinal transform.

    Args:
        xinput (torch.Tensor): input image tensor with shape :math:`(C, H, W)`.
        fixation_pts (torch.Tensor): fixation points; optional.
        nlayers (int): number of layers in the retinal transform.
        p (float): parameter of the retinal transform.
        alpha (float): parameter of the retinal transform.
        kernel_size (int): size of the isotropic Gaussian kernel.
        sigma (float): standard deviation of the Gaussian kernel.
        k (float): parameter of the retinal transform (foveal width).
        eps (float): parameter of the retinal transform (numerical stabiliser).
        device (str): device to use; optional.

    Returns:
        torch.Tensor: foveated image with shape :math:`(C, H, W)`.
    """
    KORNIA_CHECK_SHAPE(xinput, ["C", "H", "W"])
    KORNIA_CHECK(
        isinstance(nlayers, int) or nlayers < 0,
        f"Invalid iterations, it must be a positive integer. Got: {nlayers}",
    )
    KORNIA_CHECK(
        isinstance(kernel_size, int) or kernel_size <= 0,
        f"Invalid kernel size, it must be a positive integer. Got: {kernel_size}",
    )

    # Get sizes
    nchannels: int
    height: int
    width: int
    nchannels, height, width = xinput.shape

    # Select device
    if device is None:
        device = "cuda" if th.cuda.is_available() else "cpu"
    else:
        if device == "cuda" and not th.cuda.is_available():
            raise RuntimeError("CUDA is not available.")

    # Compute default fixation point
    if fixation_pts is None:
        fixation_pts: Tensor = th.tensor([width / 2, height / 2], device=device)

    # Retinal Pyramid
    aa: list[Tensor] = retina_pyramid(
        xinput=xinput.unsqueeze(0),
        kernel_size=kernel_size,
        sigma=sigma,
        nlayers=nlayers,
        device=device,
    )
    aa: list[Tensor] = [ia.squeeze(0) for ia in aa]

    # Build mesh
    mesh_x_axis: Tensor = th.arange(0, width, 1, device=device)
    mesh_y_axis: Tensor = th.arange(0, height, 1, device=device)
    meshx: Tensor
    meshy: Tensor
    meshx, meshy = th.meshgrid(mesh_x_axis, mesh_y_axis, indexing="xy")

    # Compute Theta
    theta = (
        th.sqrt((meshx - fixation_pts[0][0]) ** 2 + (meshy - fixation_pts[0][1]) ** 2)
        / p
    )

    # Clip Theta
    for fix in fixation_pts[1:]:
        theta = th.minimum(
            theta, th.sqrt((meshx - fix[0]) ** 2 + (meshy - fix[1]) ** 2) / p  # type: ignore
        )

    # Compute R
    rr = alpha / (theta + alpha)

    # Compute Ts
    tt: list[Tensor] = [
        th.exp(-(((2 ** (i - (nlayers // 2))) * rr / sigma) ** 2) * k)  # type: ignore
        for i in range(1, nlayers)
    ] + [th.zeros_like(theta, device=device)]

    # Compute Omega
    omega: Tensor = th.zeros(nlayers, device=device)
    for i in range(1, nlayers):
        omega[i - 1] = th.minimum(
            th.tensor(1.0, device=device), th.sqrt(th.log(th.tensor(2, device=device)) / k) / (2 ** (i - (nlayers // 2))) * sigma  # type: ignore # noqa: E501
        )

    # Compute layer indices
    layer_ind: Tensor = th.zeros_like(rr, device=device)  # type: ignore
    i: int
    for i in range(1, nlayers):
        layer_ind += i * logical_and(rr.gt(omega[i]), rr.le(omega[i - 1])).int()  # type: ignore

    # Compute Bs
    bb = [(0.5 - tt[i]) / (tt[i - 1] - tt[i] + eps) for i in range(1, nlayers)]

    # Compute Ms
    mmlist: list[Tensor] = []
    _: int
    for _ in range(nlayers):
        mmlist.append(th.zeros_like(rr, device=device))  # type: ignore

    mmlist[0] = layer_ind.eq(0).float() + th.multiply(bb[0], layer_ind.eq(1).float())
    for i in range(1, nlayers):
        mmlist[i] = th.multiply(1 - bb[i - 1], layer_ind.eq(i).float()) + th.multiply(
            bb[i] if i < nlayers - 1 else 0, layer_ind.eq(i + 1).float()
        )

    mm: Tensor = th.stack(mmlist)

    # Compute foveated image
    foveated: Tensor = th.zeros_like(aa[0], device=device)
    for m, a in zip(mm, aa):
        i: int
        for i in range(nchannels):
            foveated[i, :, :] += th.multiply(m, a[i, :, :])

    return foveated


_foveate = th.vmap(
    func=foveate_atomic,
    in_dims=(0, 0, None, None, None, None, None, None, None, None),
    out_dims=0,
    randomness="different",
)


def foveate(
    xinput: Tensor,
    fixation_pts: Tensor,
    nlayers: int = 6,
    p: float = 7.5,
    alpha: float = 2.5,
    kernel_size: int = 5,
    sigma: float = 0.248,
    k: float = 1.0,
    eps: float = 1e-5,
    device: Device | None = None,
) -> Tensor:
    r"""Foveate an image via the retinal transform.

    Args:
        xinput (torch.Tensor): input image tensor with shape :math:`(B, C, H, W)`.
        fixation_pts (torch.Tensor): fixation points with shape :math:`(B, C, H, W)`.; optional.
        nlayers (int): number of layers in the retinal transform.
        p (float): parameter of the retinal transform.
        alpha (float): parameter of the retinal transform.
        kernel_size (int): size of the isotropic Gaussian kernel.
        sigma (float): standard deviation of the Gaussian kernel.
        k (float): parameter of the retinal transform (foveal width).
        eps (float): parameter of the retinal transform (numerical stabiliser).
        device (str): device to use; optional.

    Returns:
        torch.Tensor: foveated image with shape :math:`(B, C, H, W)`.
    """
    KORNIA_CHECK_SHAPE(xinput, ["B", "C", "H", "W"])
    return _foveate(
        xinput, fixation_pts, nlayers, p, alpha, kernel_size, sigma, k, eps, device
    )


# ------------------------------------------------------------------------------


class Foveate2D(th.nn.Module):
    def __init__(
        self,
        nlayers: int = 6,
        p: float = 7.5,
        alpha: float = 2.5,
        kernel_size: int = 5,
        sigma: float = 0.248,
        k: float = 1.0,
        eps: float = 1e-5,
        device: Device | None = None,
    ) -> None:
        super().__init__()
        self._nlayers: int = nlayers
        self._p: float = p
        self._alpha: float = alpha
        self._kernel_size: int = kernel_size
        self._sigma: float = sigma
        self._k: float = k
        self._eps: float = eps
        self._device: Device | None = device
        self._fixate_points: Tensor | None = None

    def fixate(self, fixation_pts: Tensor | None = None):
        self._fixate_points: Tensor | None = fixation_pts
        return self

    def forward(self, xinput: Tensor, fixation_pts: Tensor | None = None) -> Tensor:
        if fixation_pts is None:
            if self._fixate_points is None:
                raise ValueError(
                    "Fixation points must be provided: either directly, or via the `fixate` method."
                )
            fixation_pts = self._fixate_points

        return foveate(
            xinput,
            fixation_pts,
            self._nlayers,
            self._p,
            self._alpha,
            self._kernel_size,
            self._sigma,
            self._k,
            self._eps,
            self._device,
        )


class FovealPyramidBlur(th.nn.Module):
    def __init__(
        self,
        pyramid_level: int,
        kernel_size: int = 5,
        sigma: float = 0.248,
        nlayers: int = 6,
        border_type: str = "reflect",
        align_corners: bool = False,
        factor: float = 2.0,
    ) -> None:
        super().__init__()
        self.pyramid_level: int = pyramid_level
        self.kernel_size: int = kernel_size
        self.sigma: float = sigma
        self.nlayers: int = nlayers
        self.border_type: str = border_type
        self.align_corners: bool = align_corners
        self.factor: float = factor

    def forward(self, xinput: Tensor) -> Tensor:
        return retina_pyramid_blur(
            xinput=xinput,
            pyramid_level=self.pyramid_level,
            kernel_size=self.kernel_size,
            sigma=self.sigma,
            nlayers=self.nlayers,
            border_type=self.border_type,
            align_corners=self.align_corners,
            factor=self.factor,
            device=xinput.device,
            dtype=xinput.dtype,
        )
