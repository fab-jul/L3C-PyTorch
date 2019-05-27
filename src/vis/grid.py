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
import functools

import torch
from fjcommon import functools_ext as ft
from torch.nn import functional as F


# TODO: document thoroughly
def prep_for_grid(x, pad_to=None, channelwise=False, insert_empty_indices=None):
    if insert_empty_indices is not None:
        assert isinstance(insert_empty_indices, list)
        assert isinstance(x, list), 'Need list for insert_empty_indices, got {}'.format(x)
    if isinstance(x, tuple) or isinstance(x, list):
        if insert_empty_indices:
            some_x = x[0]
            for idx in sorted(insert_empty_indices, reverse=True):
                x.insert(idx, torch.zeros_like(some_x))
        if pad_to is None:  # we are given a list of tensors, they must be padded!
            pad_to = max(el.size()[-1] for el in x)  # maximum width
        _prep_for_grid = functools.partial(prep_for_grid,
                                           pad_to=pad_to, channelwise=channelwise, insert_empty_indices=None)
        return ft.lconcat(map(_prep_for_grid, x))

    if x.dim() == 2:  # HW
        x = x.unsqueeze(0)  # NHW
    if x.dim() == 3:  # NHW
        assert not channelwise
        x = x.unsqueeze(1)  # NCHW
    assert x.dim() == 4, "Expected NCHW"
    x = x[0, ...]  # now: CHW
    if pad_to:
        w = x.size()[-1]
        pad = (pad_to - w) // 2
        if pad:
            x = F.pad(x, (pad, pad, pad, pad))
    if channelwise:
        return [c.unsqueeze(0) for c in torch.unbind(x, 0)]
    else:
        return [x]
