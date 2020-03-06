"""
Copyright 2019, ETH Zurich

This file is part of L3C-PyTorch.

L3C-PyTorch is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
any later version.

L3C-PyTorch is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with L3C-PyTorch.  If not, see <https://www.gnu.org/licenses/>.
"""
from fjcommon import functools_ext as ft
from torch.nn import functional as F


def pad(img, fac, mode='replicate'):
    """
    pad img such that height and width are divisible by fac
    """
    _, _, h, w = img.shape
    padH = fac - (h % fac)
    padW = fac - (w % fac)
    if padH == fac and padW == fac:
        return img, ft.identity
    if padH == fac:
        padTop = 0
        padBottom = 0
    else:
        padTop = padH // 2
        padBottom = padH - padTop
    if padW == fac:
        padLeft = 0
        padRight = 0
    else:
        padLeft = padW // 2
        padRight = padW - padLeft
    assert (padTop + padBottom + h) % fac == 0
    assert (padLeft + padRight + w) % fac == 0

    padding_tuple = (padLeft, padRight, padTop, padBottom)

    return F.pad(img, padding_tuple, mode), padding_tuple


def undo_pad(img, padLeft, padRight, padTop, padBottom, target_shape=None):
    # the 'or None' makes sure that we don't get 0:0
    img_out = img[..., padTop:(-padBottom or None), padLeft:(-padRight or None)]
    if target_shape:
        h, w = target_shape
        assert img_out.shape[-2:] == (h, w), (img_out.shape[-2:], (h, w), img_.shape,
                                              (padLeft, padRight, padTop, padBottom))
    return img_out
