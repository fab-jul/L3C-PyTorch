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
from contextlib import contextmanager

from tensorboardX import SummaryWriter
from tensorboardX.summary import Summary, _clean_tag, make_image as make_image_summary

from vis.figure_plotter import PlotToArray
from vis.image_summaries import to_image


class SafeSummaryWriter(SummaryWriter):
    def get_events_file_path(self):
        try:
            return self.file_writer.event_writer._ev_writer._file_prefix
        except AttributeError:
            print('Cannot get events file name...')
            return None

    @staticmethod
    def pre(prefix, tag):
        assert prefix[0] != '/'
        return prefix.rstrip('/') + '/' + tag.lstrip('/')

    def add_image(self, tag, img_tensor, global_step=None, **kwargs):
        """
        Add image img_tensor to summary.
        img_tensor can be np.ndarray or torch.Tensor, 1HW or 3HW or HW
        If img_tensor is uint8:
            add it like it is
        If img_tensor is float32:
            check that it is in [0, 1] and add it
            :param **kwargs:
        """
        if len(img_tensor.shape) == 2:
            img_tensor = img_tensor.reshape(1, *img_tensor.shape)
        self.file_writer.add_summary(SafeSummaryWriter._to_image_summary_safe(tag, img_tensor), global_step)

    @contextmanager
    def add_figure_ctx(self, tag, global_step=None):
        """
        Context manager that yields a plt to draw on, converts it to image
        """
        p = PlotToArray()
        yield p.prepare()  # user draws
        self.add_image(tag, p.get_numpy(), global_step, )

    @staticmethod
    def _to_image_summary_safe(tag, tensor):
        tag = _clean_tag(tag)
        img = make_image_summary(to_image(tensor))
        return Summary(value=[Summary.Value(tag=tag, image=img)])

