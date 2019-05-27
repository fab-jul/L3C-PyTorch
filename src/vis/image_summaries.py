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
import numpy as np
import pytorch_ext as pe
from fjcommon.assertions import assert_exc


def imshow(img):
    """ Only meant for local visualizations. Requires HW3 """
    import matplotlib.pyplot as plt
    assert isinstance(img, np.ndarray)
    assert img.dtype == np.uint8
    assert img.ndim == 3
    if img.shape[0] == 3:
        img = img.transpose(1, 2, 0)
    assert img.shape[2] == 3, img.shape
    plt.imshow(img)
    plt.show()


def to_image(t):
    """
    :param t: tensor or np.ndarray, may be of shape NCHW / CHW with C=1 or 3 / HW, dtype float32 or uint8. If float32:
    must be in [0, 1]
    :return: HW3 uint8 np.ndarray
    """
    if not isinstance(t, np.ndarray):
        t = pe.tensor_to_np(t)
    # - t is numpy array
    if t.ndim == 4:
        # - t has batch dimension, only use first
        t = t[0, ...]
    elif t.ndim == 2:
        t = np.expand_dims(t, 0)  # Now 1HW
    assert_exc(t.ndim == 3, 'Invalid shape: {}'.format(t.shape))
    # - t is 3 dimensional CHW numpy array
    if t.dtype != np.uint8:
        assert_exc(t.dtype == np.float32, 'Expected either uint8 or float32, got {}'.format(t.dtype))
        _check_range(t, 0, 1)
        t = (t * 255.).astype(np.uint8)
    # - t is uint8 numpy array
    num_channels = t.shape[0]
    if num_channels == 3:
        t = np.transpose(t, (1, 2, 0))
    elif num_channels == 1:
        t = np.stack([t[0, :, :] for _ in range(3)], -1)
    else:
        raise ValueError('Expected CHW, got {}'.format(t.shape))
    assert_exc(t.ndim == 3 and t.shape[2] == 3, str(t.shape))
    # - t is uint8 numpy array of shape HW3
    return t


def _check_range(a, lo, hi):
    a_lo, a_hi = np.min(a), np.max(a)
    assert_exc(a_lo >= lo and a_hi <= hi, 'Invalid range: [{}, {}]. Expected: [{}, {}]'.format(a_lo, a_hi, lo, hi))


