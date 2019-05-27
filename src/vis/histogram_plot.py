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

--------------------------------------------------------------------------------

Class to create histograms of tensors.

"""

from helpers import rolling_buffer
from vis.summarizable_module import SummarizableModule


class HistogramPlot(SummarizableModule):
    def __init__(self, prefix, name, buffer_size, num_inputs_to_buffer=1, per_channel=False, most_mass=5e-5):
        """
        :param prefix:
        :param name: Name in TensorBoard
        :param buffer_size: buffer size
        :param num_inputs_to_buffer: x[:num_inputs_to_buffer, ...] will be stored only
        :param per_channel: if True, create a histo per channel
        """
        super(HistogramPlot, self).__init__()
        self.buffer_size = buffer_size
        self.num_inputs_to_buffer = num_inputs_to_buffer
        self.per_channel = per_channel
        self.buffers = None
        self.num_chan = None  # non-None if self.per_channel and forward has been called at least once
        self.figure_creator = {name: self._plot}
        self.prefix = prefix
        self.name = name
        self.most_mass = most_mass

    def forward(self, x):
        """
        :param x: Tensor
        :returns: x
        """
        if not self.training:  # only during training
            return x
        if self.buffers is None:
            self.buffers = self._new_buffers(x.detach())
        if self.per_channel:
            for c in range(self.num_chan):
                self.buffers[c].add(x[:self.num_inputs_to_buffer, c, ...].detach())
        else:
            self.buffers.add(x[:self.num_inputs_to_buffer, ...].detach())
        # register for plotting
        self.summarizer.register_figures(self.prefix, self.figure_creator)
        return x

    def _plot(self, plt):
        """ Called when summarizer decides to plot. """
        for b in self._iter_over_buffers():
            x, y = b.plot(bins=128, most_mass=self.most_mass)
            plt.plot(x, y)

    def _iter_over_buffers(self):
        if not self.per_channel:  # self.buffers just a RollingBuffer instance
            yield self.buffers
        else:
            yield from self.buffers

    def _new_buffers(self, x):
        if not self.per_channel:
            return rolling_buffer.RollingBufferHistogram(self.buffer_size, self.name)
        self.num_chan = x.shape[1]
        return [rolling_buffer.RollingBufferHistogram(self.buffer_size, self.name)
                for _ in range(self.num_chan)]