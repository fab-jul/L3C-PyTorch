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

File for Net, which contains encoder/decoder of MultiscaleNetwork

"""
from collections import namedtuple

import torch
from torch import nn

import pytorch_ext as pe
import vis.histogram_plot
import vis.summarizable_module
from dataloaders.images_loader import resize_bicubic_batch
from modules import edsr
from modules.quantizer import Quantizer

EncOut = namedtuple('EncOut', ['bn',    # NCH'W'
                               'bn_q',  # quantized bn, NCH'W'
                               'S',     # NCH'W', long
                               'L',     # int
                               'F'      # NCfH'W', float, before Q
                               ])
DecOut = namedtuple('DecOut', ['F',     # NCfHW
                               ])


conv = pe.default_conv


class Net(nn.Module):
    def __init__(self, config_ms, scale):
        super(Net, self).__init__()
        self.config_ms = config_ms
        self.enc = {
            'EDSRLikeEnc': EDSRLikeEnc,
            'BicubicSubsampling': BicubicDownsamplingEnc,
        }[config_ms.enc.cls](config_ms, scale)
        self.dec = {
            'EDSRDec': EDSRDec
        }[config_ms.dec.cls](config_ms, scale)

    def forward(self, x):
        raise NotImplementedError()  # Call .enc and .dec directly


class BicubicDownsamplingEnc(vis.summarizable_module.SummarizableModule):
    def __init__(self, *_):
        super(BicubicDownsamplingEnc, self).__init__()
        # TODO: ugly
        self.rgb_mean = torch.tensor(
                [0.4488, 0.4371, 0.4040], dtype=torch.float32).reshape(3, 1, 1).mul(255.).to(pe.DEVICE)

    def forward(self, x):
        x = x + self.rgb_mean  # back to 0...255
        x = x.clamp(0, 255.).round().type(torch.uint8)
        x = resize_bicubic_batch(x, 0.5).to(pe.DEVICE)
        sym = x.long()
        x = x.float() - self.rgb_mean
        x = x.detach()  # make sure no gradients back to this point
        self.summarizer.register_images('train', {'input_subsampled': lambda: sym.type(torch.uint8)})
        return EncOut(x, x, sym, 256, None)


def new_levels(L, initial_levels):
    lo, hi = initial_levels
    levels = torch.linspace(lo, hi, L)
    return torch.tensor(levels, requires_grad=False)


class EDSRLikeEnc(vis.summarizable_module.SummarizableModule):
    def __init__(self, config_ms, scale):
        super(EDSRLikeEnc, self).__init__()

        self.scale = scale
        self.config_ms = config_ms
        Cf = config_ms.Cf
        kernel_size = config_ms.kernel_size
        C, self.L = config_ms.q.C, config_ms.q.L

        n_resblock = config_ms.enc.num_blocks

        # Downsampling
        self.down = conv(Cf, Cf, kernel_size=5, stride=2)

        # Body
        m_body = [
            edsr.ResBlock(conv, Cf, kernel_size, act=nn.ReLU(True))
            for _ in range(n_resblock)
        ]
        m_body.append(conv(Cf, Cf, kernel_size))
        self.body = nn.Sequential(*m_body)

        # to Quantizer
        to_q = [conv(Cf, C, 1)]
        if self.training:
            to_q.append(
                # start scale from 1, as 0 is RGB
                vis.histogram_plot.HistogramPlot('train', 'histo/enc_{}_after_1x1'.format(scale+1), buffer_size=10,
                                                 num_inputs_to_buffer=1, per_channel=False))
        self.to_q = nn.Sequential(*to_q)

        # We assume q.L levels, evenly distributed between q.levels_range[0] and q.levels_range[1]
        # In theory, the levels could be learned. But in this code, they are assumed to be fixed.
        levels_first, levels_last = config_ms.q.levels_range
        # Wrapping this in a nn.Parameter ensures it is copied to gpu when .to('cuda') is called
        self.levels = nn.Parameter(torch.linspace(levels_first, levels_last, self.L), requires_grad=False)
        self._extra_repr = 'Levels={}'.format(','.join(map('{:.1f}'.format, list(self.levels))))
        self.q = Quantizer(self.levels, config_ms.q.sigma)

    def extra_repr(self):
        return self._extra_repr

    def quantize_x(self, x):
        _, x_hard, _ = self.q(x)
        return x_hard

    def forward(self, x):
        """
        :param x: NCHW
        :return:
        """
        x = self.down(x)
        x = self.body(x) + x
        F = x
        x = self.to_q(x)
        # assert self.summarizer is not None
        x_soft, x_hard, symbols_hard = self.q(x)
        # TODO(parallel): To support nn.DataParallel, this must be changed, as it not a tensor
        return EncOut(x_soft, x_hard, symbols_hard, self.L, F)


class EDSRDec(nn.Module):
    def __init__(self, config_ms, scale):
        super(EDSRDec, self).__init__()

        self.scale = scale
        n_resblock = config_ms.dec.num_blocks

        Cf = config_ms.Cf
        kernel_size = config_ms.kernel_size
        C = config_ms.q.C

        after_q_kernel = 1
        self.head = conv(C, config_ms.Cf, after_q_kernel)
        m_body = [
            edsr.ResBlock(conv, Cf, kernel_size, act=nn.ReLU(True))
            for _ in range(n_resblock)
        ]
        
        m_body.append(conv(Cf, Cf, kernel_size))
        self.body = nn.Sequential(*m_body)
        self.tail = edsr.Upsampler(conv, 2, Cf, act=False)

    def forward(self, x, features_to_fuse=None):
        """
        :param x: NCHW
        :return:
        """
        x = self.head(x)
        if features_to_fuse is not None:
            x = x + features_to_fuse
        x = self.body(x) + x
        x = self.tail(x)
        # TODO(parallel): To support nn.DataParallel, this must be changed, as it not a tensor
        return DecOut(x)


