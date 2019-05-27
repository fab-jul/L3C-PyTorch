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
import torch

from sys import platform
import matplotlib as mpl
if platform != 'darwin':
    mpl.use('Agg')  # No display
import matplotlib.pyplot as plt
import matplotlib.backends.backend_agg as plt_backend_agg
import numpy as np


class PlotToArray(object):
    """
    p = PlotToArray()
    plt = p.prepare()
    # add plot to plt
    im = plt.get_numpy()  # CHW
    """
    def __init__(self):
        self.fig = None

    def prepare(self):
        self.fig = plt.figure(dpi=100)
        return plt

    def get_numpy(self):
        assert self.fig is not None
        return figure_to_image(self.fig)  # CHW

    def get_tensor(self):
        return torch.from_numpy(self.get_numpy())  # CHW


def figure_to_image(figures, close=True):
    if not isinstance(figures, list):
        image = _render_to_rgb(figures, close)
        return image
    else:
        images = [_render_to_rgb(figure, close) for figure in figures]
        return np.stack(images)

def _render_to_rgb(figure, close):
    canvas = plt_backend_agg.FigureCanvasAgg(figure)
    canvas.draw()
    data = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
    w, h = figure.canvas.get_width_height()
    image_hwc = data.reshape([h, w, 4])[..., :3]
    image_chw = np.moveaxis(image_hwc, source=2, destination=0)
    if close:
        plt.close(figure)
    return image_chw
