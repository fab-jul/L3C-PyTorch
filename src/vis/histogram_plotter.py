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


_WIDTH = 0.8


# TODO: rename to something with bar plot


def plot_histogram(datas, plt):
    for i, data in enumerate(datas):
        rel_i = i - len(datas) / 2
        w = _WIDTH/len(datas)
        _plot_histogram(data, plt, w, rel_i * w)
    plt.legend()


def _plot_histogram(data, plt, width, offset):
    name, values = data
    plt.bar(np.arange(len(values)) + offset, values,
            width=width,
            label=name, align='edge')


def _test():
    import matplotlib.pyplot as plt

    f = plt.figure()
    datas = [('gt', [1000, 10, 33, 500, 600, 700]),
             ('outs', [900, 20, 0, 0, 100, 1000]),
             ('ups', 0.5 * np.array([900, 20, 0, 0, 100, 1000])),
             ]
    plot_histogram(datas, plt)
    f.show()


if __name__ == '__main__':
    _test()
