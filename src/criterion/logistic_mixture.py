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

This class is based on the TensorFlow code of PixelCNN++:
    https://github.com/openai/pixel-cnn/blob/master/pixel_cnn_pp/nn.py
In contrast to that code, we predict mixture weights pi for each channel, i.e., mixture weights are "non-shared".
Also, x_min, x_max and L are parameters, and we implement a function to get the CDF of a channel.

# ------
# Naming
# ------

Note that we use the following names through the code, following the code PixelCNN++:
    - x: targets, e.g., the RGB image for scale 0
    - l: for the output of the network;
      In Fig. 2 in our paper, l is the final output, denoted with p(z^(s-1) | f^(s)), i.e., it contains the parameters
      for the mixture weights.
"""

from collections import namedtuple

import torch
import torch.nn.functional as F
import torchvision
from fjcommon import functools_ext as ft

import vis.grid
import vis.summarizable_module
from modules import quantizer

# Note that for RGB, we predict the parameters mu, sigma, pi and lambda. Since RGB has C==3 channels, it so happens that
# the total number of channels needed to predict the 4 parameters is 4 * C * K (for K mixtures, see final paragraphs of
# Section 3.4 in the paper). Note that for an input of, e.g., C == 4 channels, we would need 3 * C * K + 6 * K channels
# to predict all parameters. To understand this, see Eq. (7) in the paper, where it can be seen that for \tilde \mu_4,
# we would need 3 lambdas.
# We do not implement this case here, since it would complicate the code unnecessarily.
_NUM_PARAMS_RGB = 4  # mu, sigma, pi, lambda
_NUM_PARAMS_OTHER = 3  # mu, sigma, pi

_LOG_SCALES_MIN = -7.
_MAX_K_FOR_VIS = 10


CDFOut = namedtuple('CDFOut', ['logit_probs_c_sm',
                               'means_c',
                               'log_scales_c',
                               'K',
                               'targets'])


def non_shared_get_Kp(K, C):
    """ Get Kp=number of channels to predict. See note where we define _NUM_PARAMS_RGB above """
    if C == 3:  # finest scale
        return _NUM_PARAMS_RGB * C * K
    else:
        return _NUM_PARAMS_OTHER * C * K


def non_shared_get_K(Kp, C):
    """ Inverse of non_shared_get_Kp, get back K=number of mixtures """
    if C == 3:
        return Kp // (_NUM_PARAMS_RGB * C)
    else:
        return Kp // (_NUM_PARAMS_OTHER * C)


# --------------------------------------------------------------------------------


class DiscretizedMixLogisticLoss(vis.summarizable_module.SummarizableModule):
    def __init__(self, rgb_scale: bool, x_min=0, x_max=255, L=256):
        """
        :param rgb_scale: Whether this is the loss for the RGB scale. In that case,
            use_coeffs=True
            _num_params=_NUM_PARAMS_RGB == 4, since we predict coefficients lambda. See note above.
        :param x_min: minimum value in targets x
        :param x_max: maximum value in targets x
        :param L: number of symbols
        """
        super(DiscretizedMixLogisticLoss, self).__init__()
        self.rgb_scale = rgb_scale
        self.x_min = x_min
        self.x_max = x_max
        self.L = L
        # whether to use coefficients lambda to weight means depending on previously outputed means.
        self.use_coeffs = rgb_scale
        # P means number of different variables contained in l, l means output of network
        self._num_params = _NUM_PARAMS_RGB if rgb_scale else _NUM_PARAMS_OTHER

        # NOTE: in contrast to the original code, we use a sigmoid (instead of a tanh)
        # The optimizer seems to not care, but it would probably be more principaled to use a tanh
        # Compare with L55 here: https://github.com/openai/pixel-cnn/blob/master/pixel_cnn_pp/nn.py#L55
        self._nonshared_coeffs_act = torch.sigmoid

        # Adapted bounds for our case.
        self.bin_width = (x_max - x_min) / (L-1)
        self.x_lower_bound = x_min + 0.001
        self.x_upper_bound = x_max - 0.001

        self._extra_repr = 'DMLL: x={}, L={}, coeffs={}, P={}, bin_width={}'.format(
                (self.x_min, self.x_max), self.L, self.use_coeffs, self._num_params, self.bin_width)

    def to_sym(self, x):
        return quantizer.to_sym(x, self.x_min, self.x_max, self.L)

    def to_bn(self, S):
        return quantizer.to_bn(S, self.x_min, self.x_max, self.L)

    def extra_repr(self):
        return self._extra_repr

    @staticmethod
    def to_per_pixel(entropy, C):
        N, H, W = entropy.shape
        return entropy.sum() / (N*C*H*W)  # NHW -> scalar

    def cdf_step_non_shared(self, l, targets, c_cur, C, x_c=None) -> CDFOut:
        assert c_cur < C

        # NKHW         NKHW     NKHW
        logit_probs_c, means_c, log_scales_c, K = self._extract_non_shared_c(c_cur, C, l, x_c)

        logit_probs_c_softmax = F.softmax(logit_probs_c, dim=1)  # NKHW, pi_k
        return CDFOut(logit_probs_c_softmax, means_c, log_scales_c, K, targets.to(l.device))

    def sample(self, l, C):
        return self._non_shared_sample(l, C)

    def forward(self, x, l, scale=0):
        """
        :param x: labels, i.e., NCHW, float
        :param l: predicted distribution, i.e., NKpHW, see above
        :return: log-likelihood, as NHW if shared, NCHW if non_shared pis
        """
        assert x.min() >= self.x_min and x.max() <= self.x_max, '{},{} not in {},{}'.format(
                x.min(), x.max(), self.x_min, self.x_max)

        # Extract ---
        #  NCKHW      NCKHW  NCKHW
        x, logit_pis, means, log_scales, K = self._extract_non_shared(x, l)

        # visualize pi, means, variances
        self.summarizer.register_images(
                'val', {f'dmll/{scale}/c{c}': lambda c=c: _visualize_params(logit_pis, means, log_scales, c)
                        for c in range(x.shape[1])})

        centered_x = x - means  # NCKHW

        # Calc P = cdf_delta
        # all of the following is NCKHW
        inv_stdv = torch.exp(-log_scales)  # <= exp(7), is exp(-sigma), inverse std. deviation, i.e., sigma'
        plus_in = inv_stdv * (centered_x + self.bin_width/2)  # sigma' * (x - mu + 0.5)
        cdf_plus = torch.sigmoid(plus_in)  # S(sigma' * (x - mu + 1/255))
        min_in = inv_stdv * (centered_x - self.bin_width/2)  # sigma' * (x - mu - 1/255)
        cdf_min = torch.sigmoid(min_in)  # S(sigma' * (x - mu - 1/255)) == 1 / (1 + exp(sigma' * (x - mu - 1/255))
        # the following two follow from the definition of the logistic distribution
        log_cdf_plus = plus_in - F.softplus(plus_in)  # log probability for edge case of 0
        log_one_minus_cdf_min = -F.softplus(min_in)  # log probability for edge case of 255
        # NCKHW, P^k(c)
        cdf_delta = cdf_plus - cdf_min  # probability for all other cases, essentially log_cdf_plus + log_one_minus_cdf_min

        # NOTE: the original code has another condition here:
        #   tf.where(cdf_delta > 1e-5,
        #            tf.log(tf.maximum(cdf_delta, 1e-12)),
        #            log_pdf_mid - np.log(127.5)
        #            )
        # which handles the extremly low porbability case. Since this is only there to stabilize training,
        # and we get fine training without it, I decided to drop it
        #
        # so, we have the following if, where I put in the x_upper_bound and x_lower_bound values for RGB
        # if x < 0.001:                         cond_C
        #       log_cdf_plus                    out_C
        # elif x > 254.999:                     cond_B
        #       log_one_minus_cdf_min           out_B
        # else:
        #       log(cdf_delta)                  out_A
        out_A = torch.log(torch.clamp(cdf_delta, min=1e-12))
        # NOTE, we adapt the bounds for our case
        cond_B = (x > self.x_upper_bound).float()
        out_B = (cond_B * log_one_minus_cdf_min + (1. - cond_B) * out_A)
        cond_C = (x < self.x_lower_bound).float()
        # NCKHW, =log(P^k(c))
        log_probs = cond_C * log_cdf_plus + (1. - cond_C) * out_B

        # combine with pi, NCKHW, (-inf, 0]
        log_probs_weighted = log_probs.add(
                log_softmax(logit_pis, dim=2))  # (-inf, 0]

        # final log(P), NCHW
        return -log_sum_exp(log_probs_weighted, dim=2)  # NCHW

    def _extract_non_shared(self, x, l):
        """
        :param x: targets, NCHW
        :param l: output of net, NKpHW, see above
        :return:
            x NC1HW,
            logit_probs NCKHW (probabilites of scales, i.e., \pi_k)
            means NCKHW,
            log_scales NCKHW (variances),
            K (number of mixtures)
        """
        N, C, H, W = x.shape
        Kp = l.shape[1]

        K = non_shared_get_K(Kp, C)

        # we have, for each channel: K pi / K mu / K sigma / [K coeffs]
        # note that this only holds for C=3 as for other channels, there would be more than 3*K coeffs
        # but non_shared only holds for the C=3 case
        l = l.reshape(N, self._num_params, C, K, H, W)

        logit_probs = l[:, 0, ...]  # NCKHW
        means = l[:, 1, ...]  # NCKHW
        log_scales = torch.clamp(l[:, 2, ...], min=_LOG_SCALES_MIN)  # NCKHW, is >= -7
        x = x.reshape(N, C, 1, H, W)

        if self.use_coeffs:
            assert C == 3  # Coefficients only supported for C==3, see note where we define _NUM_PARAMS_RGB
            coeffs = self._nonshared_coeffs_act(l[:, 3, ...])  # NCKHW, basically coeffs_g_r, coeffs_b_r, coeffs_b_g
            means_r, means_g, means_b = means[:, 0, ...], means[:, 1, ...], means[:, 2, ...]  # each NKHW
            coeffs_g_r,  coeffs_b_r, coeffs_b_g = coeffs[:, 0, ...], coeffs[:, 1, ...], coeffs[:, 2, ...]  # each NKHW
            means = torch.stack(
                    (means_r,
                     means_g + coeffs_g_r * x[:, 0, ...],
                     means_b + coeffs_b_r * x[:, 0, ...] + coeffs_b_g * x[:, 1, ...]), dim=1)  # NCKHW again

        assert means.shape == (N, C, K, H, W), (means.shape, (N, C, K, H, W))
        return x, logit_probs, means, log_scales, K

    def _extract_non_shared_c(self, c, C, l, x=None):
        """
        Same as _extract_non_shared but only for c-th channel, used to get CDF
        """
        assert c < C, f'{c} >= {C}'

        N, Kp, H, W = l.shape
        K = non_shared_get_K(Kp, C)

        l = l.reshape(N, self._num_params, C, K, H, W)
        logit_probs_c = l[:, 0, c, ...]  # NKHW
        means_c = l[:, 1, c, ...]  # NKHW
        log_scales_c = torch.clamp(l[:, 2, c, ...], min=_LOG_SCALES_MIN)  # NKHW, is >= -7

        if self.use_coeffs and c != 0:
            unscaled_coeffs = l[:, 3, ...]  # NCKHW, coeffs_g_r, coeffs_b_r, coeffs_b_g
            if c == 1:
                assert x is not None
                coeffs_g_r = torch.sigmoid(unscaled_coeffs[:, 0, ...])  # NKHW
                means_c += coeffs_g_r * x[:, 0, ...]
            elif c == 2:
                assert x is not None
                coeffs_b_r = torch.sigmoid(unscaled_coeffs[:, 1, ...])  # NKHW
                coeffs_b_g = torch.sigmoid(unscaled_coeffs[:, 2, ...])  # NKHW
                means_c += coeffs_b_r * x[:, 0, ...] + coeffs_b_g * x[:, 1, ...]

        #      NKHW           NKHW     NKHW
        return logit_probs_c, means_c, log_scales_c, K

    def _non_shared_sample(self, l, C):
        """ sample from model """
        N, Kp, H, W = l.shape
        K = non_shared_get_K(Kp, C)
        l = l.reshape(N, self._num_params, C, K, H, W)

        logit_probs = l[:, 0, ...]  # NCKHW

        # sample mixture indicator from softmax
        u = torch.zeros_like(logit_probs).uniform_(1e-5, 1. - 1e-5)  # NCKHW
        sel = torch.argmax(
                logit_probs - torch.log(-torch.log(u)),  # gumbel sampling
                dim=2)  # argmax over K, results in NCHW, specifies for each c: which of the K mixtures to take
        assert sel.shape == (N, C, H, W), (sel.shape, (N, C, H, W))

        sel = sel.unsqueeze(2)  # NC1HW

        means = torch.gather(l[:, 1, ...], 2, sel).squeeze(2)
        log_scales = torch.clamp(torch.gather(l[:, 2, ...], 2, sel).squeeze(2), min=_LOG_SCALES_MIN)

        # sample from the resulting logistic, which now has essentially 1 mixture component only.
        # We use inverse transform sampling. i.e. X~logistic; generate u ~ Unfirom; x = CDF^-1(u),
        #  where CDF^-1 for the logistic is CDF^-1(y) = \mu + \sigma * log(y / (1-y))
        u = torch.zeros_like(means).uniform_(1e-5, 1. - 1e-5)  # NCHW
        x = means + torch.exp(log_scales) * (torch.log(u) - torch.log(1. - u))  # NCHW

        if self.use_coeffs:
            assert C == 3

            clamp = lambda x_: torch.clamp(x_, 0, 255.)

            # Be careful about coefficients! We need to use the correct selection mask, namely the one for the G and
            #  B channels, as we update the G and B means! Doing torch.gather(l[:, 3, ...], 2, sel) would be completly
            #  wrong.
            coeffs = torch.sigmoid(l[:, 3, ...])
            sel_g, sel_b = sel[:, 1, ...], sel[:, 2, ...]
            coeffs_g_r = torch.gather(coeffs[:, 0, ...], 1, sel_g).squeeze(1)
            coeffs_b_r = torch.gather(coeffs[:, 1, ...], 1, sel_b).squeeze(1)
            coeffs_b_g = torch.gather(coeffs[:, 2, ...], 1, sel_b).squeeze(1)

            # Note: In theory, we should go step by step over the channels and update means with previously sampled
            # xs. But because of the math above (x = means + ...), we can just update the means here and it's all good.
            x0 = clamp(x[:, 0, ...])
            x1 = clamp(x[:, 1, ...] + coeffs_g_r * x0)
            x2 = clamp(x[:, 2, ...] + coeffs_b_r * x0 + coeffs_b_g * x1)
            x = torch.stack((x0, x1, x2), dim=1)
        return x


def log_prob_from_logits(logit_probs):
    """ numerically stable log_softmax implementation that prevents overflow """
    # logit_probs is NKHW
    m, _ = torch.max(logit_probs, dim=1, keepdim=True)
    return logit_probs - m - torch.log(torch.sum(torch.exp(logit_probs - m), dim=1, keepdim=True))


# TODO(pytorch): replace with pytorch internal in 1.0, there is a bug in 0.4.1
def log_softmax(logit_probs, dim):
    """ numerically stable log_softmax implementation that prevents overflow """
    m, _ = torch.max(logit_probs, dim=dim, keepdim=True)
    return logit_probs - m - torch.log(torch.sum(torch.exp(logit_probs - m), dim=dim, keepdim=True))


def log_sum_exp(log_probs, dim):
    """ numerically stable log_sum_exp implementation that prevents overflow """
    m, _        = torch.max(log_probs, dim=dim)
    m_keep, _   = torch.max(log_probs, dim=dim, keepdim=True)
    # == m + torch.log(torch.sum(torch.exp(log_probs - m_keep), dim=dim))
    return log_probs.sub_(m_keep).exp_().sum(dim=dim).log_().add(m)


def _visualize_params(logits_pis, means, log_scales, channel):
    """
    :param logits_pis:  NCKHW
    :param means: NCKHW
    :param log_scales: NCKHW
    :param channel: int
    :return:
    """
    assert logits_pis.shape == means.shape == log_scales.shape
    logits_pis = logits_pis[0, channel, ...].detach()
    means = means[0, channel, ...].detach()
    log_scales = log_scales[0, channel, ...].detach()

    pis = torch.softmax(logits_pis, dim=0)  # Kdim==0 -> KHW

    mixtures = ft.lconcat(
            zip(_iter_Kdim_normalized(pis, normalize=False),
                _iter_Kdim_normalized(means),
                _iter_Kdim_normalized(log_scales)))
    grid = vis.grid.prep_for_grid(mixtures)
    img = torchvision.utils.make_grid(grid, nrow=3)
    return img


def _iter_Kdim_normalized(t, normalize=True):
    """ normalizes t, then iterates over Kdim (1st dimension) """
    K = t.shape[0]

    if normalize:
        lo, hi = float(t.min()), float(t.max())
        t = t.clamp(min=lo, max=hi).add_(-lo).div_(hi - lo + 1e-5)

    for k in range(min(_MAX_K_FOR_VIS, K)):
        yield t[k, ...]  # HW

