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
from collections import namedtuple

import numpy as np
import torch
import torch.nn.functional as F
import torchvision

import pytorch_ext as pe
import vis.grid
import vis.summarizable_module
from helpers.pad import pad
from modules.multiscale_network import MultiscaleNetwork, Out
from vis import histogram_plotter, image_summaries


MultiscaleLoss = namedtuple(
        'MultiscaleLoss',
        ['loss_pc',                 # loss to minimize
         'nonrecursive_bpsps',      # bpsp corresponding to non-recursive scales
         'recursive_bpsps'])        # None if not recursive, else all bpsp including recursive



class MultiscaleBlueprint(vis.summarizable_module.SummarizableModule):
    def __init__(self, config_ms):
        super(MultiscaleBlueprint, self).__init__()
        net = MultiscaleNetwork(config_ms)
        net.to(pe.DEVICE)

        self.net = net
        self.losses = net.get_losses()

    def set_eval(self):
        self.net.eval()
        self.losses.loss_dmol_rgb.eval()
        self.losses.loss_dmol_n.eval()

    def forward(self, in_batch, auto_recurse=0) -> Out:
        """
        :param in_batch: NCHW 0..255 float
        :param auto_recurse: int, how many times the last scales should be applied again. Used for RGB Shared.
        :return: layers.multiscale.Out
        """
        return self.net(in_batch, auto_recurse)

    def get_loss(self, out: Out, num_subpixels_before_pad=None) -> MultiscaleLoss:
        """
        :param num_subpixels_before_pad: If given, calculate bpsp with this, instead of num_pixels returned by self.losses.
        This is needed because while testing, we have to pad images. To calculate the correct bpsp, we need to
        calcualte it with respect to the actual (non-padded) number of pixels
        :returns instance of MultiscaleLoss, see above.
        """
        # `costs`: a list, containing the cost of each scale, in nats
        costs, final_cost_uniform, num_subpixels = self.losses.get(out)
        if num_subpixels_before_pad:
            assert num_subpixels_before_pad <= num_subpixels, num_subpixels_before_pad
            num_subpixels = num_subpixels_before_pad
        # conversion between nats and bits per subpixel
        conversion = np.log(2.) * num_subpixels
        costs_bpsp = [cost/conversion for cost in costs]

        self.summarizer.register_scalars(
                'auto',
                {'costs/scale_{}_bpsp'.format(i): cost for i, cost in enumerate(costs_bpsp)})

        # all bpsps corresponding to non-recursive scales, including final (uniform-prior) cost
        nonrecursive_bpsps = costs_bpsp[:out.auto_recursive_from] + [final_cost_uniform / conversion]
        if out.auto_recursive_from is not None:
            # all bpsps corresponding to non-recursive AND recursive scales, including final cost
            recursive_bpsps = costs_bpsp + [out.get_nat_count(-1) / conversion]
        else:
            recursive_bpsps = None

        # loss is everything without final (uniform-prior) scale
        total_bpsp_without_final = sum(costs_bpsp)
        loss_pc = total_bpsp_without_final
        return MultiscaleLoss(loss_pc, nonrecursive_bpsps, recursive_bpsps)

    def sample_forward(self, in_batch, sample_scales, partial_final=None):
        return self.net.sample_forward(in_batch, self.losses, sample_scales, partial_final)

    @staticmethod
    def add_image_summaries(sw, out: Out, global_step, prefix):
        tag = lambda t: sw.pre(prefix, t)
        is_train = prefix == 'train'
        for scale, (S_i, _, P_i, L_i) in enumerate(out.iter_all_scales(), 1):  # start from 1, as 0 is RGB
            sw.add_image(tag('bn/{}'.format(scale)), new_bottleneck_summary(S_i, L_i), global_step)
            # This will only trigger for the final scale, where P_i is the uniform distribution.
            # With this, we can check how accurate the uniform assumption is (hint: not very)
            is_logits = P_i.shape[1] == L_i
            if is_logits and is_train:
                with sw.add_figure_ctx(tag('histo_out/{}'.format(scale)), global_step) as plt:
                    add_ps_summaries(S_i, get_p_y(P_i), L_i, plt)

    @staticmethod
    def bottleneck_images(s, L):
        assert s.dim() == 4, s.shape
        _assert_contains_symbol_indices(s, L)
        s = s.float().div(L)
        return [image_summaries.to_image(s[:, c, ...]) for c in range(s.shape[1])]

    @staticmethod
    def unpack_batch_pad(raw, fac):
        """
        :param raw: uint8, input image.
        :param fac: downscaling factor we will use, used to determine proper padding.
        """
        if len(raw.shape) == 3:
            raw.unsqueeze_(0)  # add batch dim
        assert len(raw.shape) == 4
        raw = MultiscaleBlueprint.pad(raw, fac)
        raw = raw.to(pe.DEVICE)
        img_batch = raw.float()
        s = raw.long()  # symbols
        return img_batch, s

    @staticmethod
    def pad(raw, fac):
        raw, _ = pad(raw, fac, mode=MultiscaleBlueprint.get_padding_mode())
        return raw

    @staticmethod
    def get_padding_mode():
        return 'constant'

    @staticmethod
    def unpack(img_batch):
        idxs = img_batch['idx'].squeeze().tolist()
        raw = img_batch['raw'].to(pe.DEVICE)
        img_batch = raw.float()
        s = raw.long()  # symbols
        return idxs, img_batch, s


def new_bottleneck_summary(s, L):
    """
    Grayscale bottleneck representation: Expects the actual bottleneck symbols.
    :param s: NCHW
    :return: [0, 1] image
    """
    assert s.dim() == 4, s.shape
    _assert_contains_symbol_indices(s, L)
    s = s.float().div(L)
    grid = vis.grid.prep_for_grid(s, channelwise=True)
    assert len(grid) == s.shape[1], (len(grid), s.shape)
    assert [g.max() <= 1 for g in grid], [g.max() for g in grid]
    assert grid[0].dtype == torch.float32, grid.dtype
    return torchvision.utils.make_grid(grid, nrow=5)


def _assert_contains_symbol_indices(t, L):
    """ assert 0 <= t < L """
    assert 0 <= t.min() and t.max() < L, (t.min(), t.max())


def add_ps_summaries(s, p_y, L, plt):
    histo_s = pe.histogram(s, L)
    p_x = histo_s / np.sum(histo_s)

    assert p_x.shape == p_y.shape, (p_x.shape, p_y.shape)

    histogram_plotter.plot_histogram([
        ('p_x', p_x),
        ('p_y', p_y),
    ], plt)


def get_p_y(y):
    """
    :param y: NLCHW float, logits
    :return: L dimensional vector p
    """
    Ldim = 1
    L = y.shape[Ldim]
    y = y.detach()
    p = F.softmax(y, dim=Ldim)
    p = p.transpose(Ldim, -1)
    p = p.contiguous().view(-1, L)  # nL
    p = torch.mean(p, dim=0)  # L
    return pe.tensor_to_np(p)
