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

Wrapper file for

MultiscaleNetwork
    contains
    - networks (Net, encoder/decoder)
    - probability classifiers
    - heads

Losses

Out (output of MultiscaleNetwork)

"""

import itertools

import numpy as np
import torch
from torch import nn

import pytorch_ext as pe
import vis.summarizable_module
from criterion.logistic_mixture import DiscretizedMixLogisticLoss
from modules import edsr
from helpers.global_config import global_config
from modules.head import RGBHead, Head
from modules.net import Net, EncOut
from modules.prob_clf import AtrousProbabilityClassifier


conv = pe.default_conv


class Out(object):
    """
    Stores outputs of network, see notes in __init__.
    """
    def __init__(self, targets_style='S', auto_recursive_from=None):
        """
        :param targets_style: has to be 'S' or 'bn'. If 'S', the predictor predicts symbols, otherwise bottlenecks.
                        For the RGB baselines, targets='S', for L3C, targets='bn'
                        The first predictor (scale 0) always predicts symbols, as it predicts RGB.
        :param auto_recursive_from: int, how many times the last scales should be applied again.
        """
        # --------------------------------------------------------------------------------
        # Notes on relationship with paper notation as shown in Fig. 1:
        # - S[0] = x, S[i] = symbols of z^(i), which are the indices of the levels.
        #       Example: if the levels would be L = {-0.5, 0, 0.5}, and z^(i) = [-0.5, -0.5, 0.5], then
        #                S[i] = [0, 0, 2], the indices of the levels
        # - L[0] = 256, L[i] = number of levels, 25 for L3C, and 256 for the RGB baselines
        # - bn[i] == z^(i)
        # - P[0] = p(x | f^(1)), P[i] = p(z^(i) | f^(i+1))
        # --------------------------------------------------------------------------------

        assert targets_style in ('S', 'bn'), targets_style

        self.S = []   # from fine to coarse, NCHW, sybmols; S[0] = RGB image
        self.L = []   # from fine to coarse, [int], number of symbols of S[i]; S[0] = 256
        self.bn = []  # from fine to coarse, NCHW actual bottleneck, floats; bn[0] = None
        self.P = []   # from fine to coarse, NLCHW, predcitions; P[-1] is uniform.
        #                   P[0] predicts S[0], P[i] predicts bn[i]/S[i] (depending if targets is 'S' or 'bn')

        # Invariant:
        # len(S) == len(L) == len(bn) == len(P) + 1.

        self.auto_recursive_from = auto_recursive_from
        self.targets_style = targets_style

    def append(self, enc_out: EncOut, P, is_training):
        self.S.append(enc_out.S)
        self.L.append(enc_out.L)
        self.P.append(P)
        self.bn.append(enc_out.bn if is_training else enc_out.bn_q)
        assert len(self.S) == len(self.L) == len(self.bn) == len(self.P) + 1

    def append_input_image(self, x):
        self.S.append(x.round().long())
        self.L.append(256)
        self.bn.append(None)  # to match lengths

    def get_uniform_P(self):
        assert len(self.S) == len(self.L) == len(self.P) + 1
        N, C, H, W = self.S[-1].shape
        L = self.L[-1]
        uniform_logits = torch.ones(N, L, C, H, W, dtype=torch.float32, device=pe.DEVICE)
        return uniform_logits

    def iter_all_scales(self):
        """ Iter S|bn, P, L for all layers, from coarse to fine """
        return zip(self.S[1:], self.bn[1:], self.P[1:] + [self.get_uniform_P()], self.L[1:])

    def iter_targets_and_predictions(self, loss_rgb, loss_others):
        """ yield tuples: (loss to use, target, predictions predicting target) for all scales except final
        (uniform-prior) scale"""
        # RGB scale
        yield (loss_rgb,
               self.S[0].float(),
               self.P[0])
        # other scales
        other_targets = (S.float() for S in self.S[1:]) if self.targets_style == 'S' else self.bn[1:]
        yield from zip(itertools.repeat(loss_others),  # repeat loss_others for all scales
                       other_targets,
                       self.P[1:])

    def get_nat_count(self, i):
        """ Get nats required to store scale i with a uniform prior. """
        assert len(self.S) == len(self.L) == len(self.P) + 1
        N, C, H, W = self.S[i].shape
        L = self.L[i]
        return N*C*H*W*np.log(L)


class Losses(vis.summarizable_module.SummarizableModule):
    def __init__(self, config_ms):
        super(Losses, self).__init__()
        self.loss_dmol_rgb = DiscretizedMixLogisticLoss(
                rgb_scale=True, x_min=0, x_max=255, L=256)
        if config_ms.rgb_bicubic_baseline:
            self.loss_dmol_n = self.loss_dmol_rgb
        else:
            x_min, x_max = config_ms.q.levels_range
            self.loss_dmol_n = DiscretizedMixLogisticLoss(
                    rgb_scale=False, x_min=x_min, x_max=x_max, L=config_ms.q.L)

    def get(self, out: Out):
        """
        :param out: instance of `Out` containing all outputs from the network, see `Out`.
        :return: tuple:
            - costs: a list, containing the cost of each scale, in nats
            - final_cost_uniform: nats of final (uniform-prior) scale
            - num_subpixels: number of sub-pixels in the input.
        """
        # for RGB: target_i == symbols
        # for L3C: target_i == bottlenecks
        costs = [loss(target_i, P_i, scale).sum()
                 for scale, (loss, target_i, P_i) in enumerate(out.iter_targets_and_predictions(
                    loss_rgb=self.loss_dmol_rgb, loss_others=self.loss_dmol_n))]

        # Note: S[1] predicts P[0]
        # Final scale that is not autorecursive (usually just the final scale, if we do not apply auto-recursion).
        final_non_recursive_idx = -1 if out.auto_recursive_from is None else out.auto_recursive_from
        final_cost_uniform = out.get_nat_count(final_non_recursive_idx)

        num_subpixels = int(np.prod(out.S[0].shape))
        return costs, final_cost_uniform, num_subpixels


class MultiscaleNetwork(vis.summarizable_module.SummarizableModule):
    def __init__(self, config_ms):
        super(MultiscaleNetwork, self).__init__()

        # Set for the RGB baselines
        self._rgb = config_ms.rgb_bicubic_baseline  # if set, make sure no backprob through sub_mean

        # True for L3C and RGB, not for RGB Shared
        self._fuse_feat = config_ms.dec.skip

        self._show_input = global_config.get('showinp', False)

        # For the first scale, where input is RGB with C=3
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_rgb_mean = edsr.MeanShift(255., rgb_mean, rgb_std)  # to interval -128, 128

        self.scales = config_ms.num_scales
        self.config_ms = config_ms

        # NOTES about naming: See README

        if not config_ms.rgb_bicubic_baseline:
            # Heads are used to make the code work for L3C as well as the RBG baselines.
            # For RGB, each encoder gets a bicubically downsampled RGB image as input, with 3 channels.
            # Otherwise, the encoder gets the final feature before the quantizer, with Cf channels.
            # The Heads map either of these to Cf channels, such that encoders always get a feature map with Cf
            # channels.
            heads = ([RGBHead(config_ms)] +
                     [Head(config_ms, Cin=self.get_Cin_for_scale(scale))
                      for scale in range(self.scales - 1)])
            nets = [Net(config_ms, scale)
                    for scale in range(self.scales)]
            prob_clfs = ([AtrousProbabilityClassifier(config_ms, C=3)] +
                         [AtrousProbabilityClassifier(config_ms, config_ms.q.C)
                          for _ in range(self.scales - 1)])
        else:
            print('*** Multiscale RGB Pyramid')
            # For RGB Baselines, we feed subsampled version of RGB directly to the next subsampler
            # (see Fig A2, A3 in appendix of paper). Thus, the heads are just identity.
            heads = [pe.LambdaModule(lambda x: x, name='ID') for _ in range(self.scales)]
            nets = [Net(config_ms, scale)
                    for scale in range(self.scales)]
            prob_clfs = [AtrousProbabilityClassifier(config_ms, C=3) for _ in range(self.scales)]

        self.heads = nn.ModuleList(heads)
        self.nets = nn.ModuleList(nets)
        self.prob_clfs = nn.ModuleList(prob_clfs)  # len == #scales

        self.extra_repr_str = 'scales={} / {} nets / {} ps'.format(
                self.scales, len(self.nets), len(self.prob_clfs))

    def get_losses(self):
        return Losses(self.config_ms)

    def extra_repr(self):
        return self.extra_repr_str

    def forward(self, x, auto_recurse=0) -> Out:
        """
        :param x: image, NCHW, [0, 255]
        :param auto_recurse: int, how many times the last scales should be applied again.
        :return: Out
        """
        # Visualize input
        # if self._show_input:
        self.summarizer.register_images('train', {'input': x.to(torch.uint8)})
        forward_scales = list(range(self.scales)) + [-1 for _ in range(auto_recurse)]

        out = Out(targets_style='S' if self._rgb else 'bn',  # IF RGB baseline, use symbols as targets for loss
                  auto_recursive_from=self.scales if auto_recurse > 0 else None)
        out.append_input_image(x)

        x = self.sub_rgb_mean(x)  # something like -128..128 but not really
        if self._rgb:
            x = x.detach()
        self._forward_with_scales(out, x, forward_scales)

        return out

    def get_next_scale_intput(self, previous_enc_out: EncOut):
        if self.config_ms.enc.feed_F:  # set to False for the RGB baselines, where there are no features
            return previous_enc_out.F
        else:
            return previous_enc_out.bn

    def get_Cin_for_scale(self, scale):
        if self.config_ms.enc.feed_F:
            return self.config_ms.Cf
        else:
            return self.config_ms.q.C

    def _forward_with_scales(self, out, x, forward_scales):
        """
        Calculate forward pass through scales, save in `out`.
        We first go through the encoders from top to bottom (finest to coarsest scale) and then throught the decoders
        from bottom to top, since decoders depend on coarser outputs.
        Note that for the RGB Shared baseline, _fuse_feat=False, i.e., there is no skip between decoders (since only
        one decoder is learned, the skip would require training infinite decoders or some trick). Thus, the two for
        loops could be simplified to a single for loop, going through both encoders and decoders at the same time.
        :param out: instance of Out
        :param x: input image
        :param forward_scales: list of ints, use -1 to denote coarsest trained scale. Used if forward is called with
        auto_recurse > 0
        """
        inp = x
        enc_outs = []

        for scale in forward_scales:  # from fine to coarse
            net = self.nets[scale]
            head = self.heads[scale]

            inp = head(inp)
            enc_out = net.enc(inp)
            enc_outs.append(enc_out)
            inp = self.get_next_scale_intput(enc_out)  # for next scale

        dec_outs = []
        for i, scale in reversed(list(enumerate(forward_scales))):  # from coarse to fine
            net = self.nets[scale]
            enc_out = enc_outs[i]

            #                                       # do not fuse features from lower scale if:
            if (not self._fuse_feat or              # disabled
                    scale == -1 or                  # autoregressive scale
                    scale == max(forward_scales)):  # final scale
                features_to_fuse = None
            else:
                assert len(dec_outs) > 0
                features_to_fuse = dec_outs[0].F

            dec_inp = enc_out.bn if self.training else enc_out.bn_q
            dec_out = net.dec(dec_inp, features_to_fuse)
            dec_outs.insert(0, dec_out)

        for scale, enc_out, dec_out in zip(forward_scales, enc_outs, dec_outs):
            prob_clf = self.prob_clfs[scale]
            P = prob_clf(dec_out.F)
            out.append(enc_out, P, self.training)

    def get_P(self, scale, bn_q, dec_F_prev=None):
        """
        Used in actual bitencoding
        :param scale: in 0, ..., num_scales-1
        :param bn_q: quantized bn, NCH'W'
        :param dec_F_prev:
        :return:
        """
        assert 0 <= scale < self.config_ms.num_scales, 'Out of range: {}'.format(scale)
        net = self.nets[scale]
        prob_clf = self.prob_clfs[scale]

        features_to_fuse = dec_F_prev
        dec_out = net.dec(bn_q, features_to_fuse)
        return prob_clf(dec_out.F), dec_out.F


    # Sampling -----------------------------------------------------------------------


    def sample_forward(self, x, losses: Losses, sample_scales, partial_final=None, auto_recurse=0):
        """
        :param x: image, NCHW, [0, 255]
        :return: Out
        """
        if auto_recurse != 0:
            raise NotImplementedError(f'Currently not supported for sampling: autorecurse={auto_recurse}')

        print('-' * 40)
        print('- Sampling {}'.format(sample_scales))
        print('-' * 40)

        forward_scales = list(range(self.scales)) + [-1 for _ in range(auto_recurse)]
        x = self.sub_rgb_mean(x)  # something like -128..128 but not really
        return self._sample_forward(x, forward_scales, losses, sample_scales, partial_final)

    def _sample_forward(self, x, forward_scales, losses: Losses, sample_scales, partial_final=None):
        """
        :param x:
        :param forward_scales:
        :param losses:
        :param sample_scales: first is always sampled. If sample = [0], also sample first bottleneck
        :return:
        """
        inp = x

        enc_outs = []
        Cs = [3]
        for scale in forward_scales:  # from fine to coarse
            net = self.nets[scale]
            head = self.heads[scale]

            # translates to NCfHW per scale. Makes sense for shared nets, where we need a special case for RGB.
            inp = head(inp)
            enc_out = net.enc(inp)
            Cs.append(enc_out.bn.shape[1])
            enc_outs.append(enc_out)
            inp = self.get_next_scale_intput(enc_out)

        prev_x = None

        # set if self._fuse_feat
        features_to_fuse = None

        for scale in reversed(forward_scales):
            loss_dmm = losses.loss_dmol_rgb if scale == 0 else losses.loss_dmol_n
            C = Cs[scale]
            net = self.nets[scale]
            prob_clf = self.prob_clfs[scale]

            if scale in sample_scales:
                if prev_x is None:
                    # fast
                    print('Sampling uniformly!')
                    fake_prev_x = torch.zeros_like(enc_outs[-1].bn_q).uniform_(-1, 1)
                    # L = helpers.get_L(scale, self.config_ms)
                    fake_prev_x = self.nets[-1].enc.quantize_x(fake_prev_x)
                    prev_x = fake_prev_x
                    if partial_final:  # partial sampling
                        print('partial sampling')
                        for c in partial_final:
                            prev_x[:, c, ...] = enc_outs[scale].bn_q[:, c, ...]

                print('{}: Feeding sampled to decoder'.format(scale))
                decoder_input = prev_x
            else:
                print('{}: Feeding encoder output to decoder'.format(scale))
                decoder_input = enc_outs[scale].bn_q

            dec_out = net.dec(decoder_input, features_to_fuse)
            if self._fuse_feat:
                features_to_fuse, = dec_out  # unpack F
            P = prob_clf(dec_out[0])  # unpack F, torch.jit thing

            if scale == 0 or scale - 1 in sample_scales:
                print('{}: sampling N{}HW for next scale'.format(scale, C))
                prev_x = loss_dmm.sample(P, C=C)

        return prev_x


def _to_set_assert_unique(l):
    n = len(l)
    s = set(l)
    assert len(s) == n, '{} != {}'.format(len(s), n)
    return s

