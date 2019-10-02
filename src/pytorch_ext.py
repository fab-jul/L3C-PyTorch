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

General PyTorch related stuff

"""

import os
from collections import defaultdict

import numpy as np
import math

import torch

from torch import nn as nn
from torch.utils.data import Dataset

import itertools


CUDA_AVAILABLE = torch.cuda.is_available()

# IGNORE_CUDA = os.environ.get('IGNORE_CUDA', '0') == '1'
# if IGNORE_CUDA:
#     print('*** IGNORE_CUDA=1')

DEVICE = torch.device("cuda:0" if CUDA_AVAILABLE else "cpu")


# This has an effect to all parts reading pytorch_ext.DEVICE *after* it has been set.
def set_device(cuda_available):
    global DEVICE
    DEVICE = torch.device("cuda:0" if cuda_available else "cpu")


# Conv -------------------------------------------------------------------------


def default_conv(in_channels, out_channels, kernel_size, bias=True, rate=1, stride=1):
    padding = kernel_size // 2 if rate == 1 else rate
    return nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride, dilation=rate,
            padding=padding, bias=bias)


def initialize_with_filter(conv_or_deconv, f):
    assert conv_or_deconv.weight.size() == f.size(), 'Must match: {}, {}'.format(
            conv_or_deconv.weight.size(), f.size())
    conv_or_deconv.weight.data = f
    return conv_or_deconv


def initialize_with_id(conv_or_deconv, with_noise=False):
    n, Cout, H, W = conv_or_deconv.weight.shape
    assert n == Cout and H == W == 1, 'Invalid shape: {}'.format(conv_or_deconv.weight.shape)
    eye = torch.eye(n).reshape(n, n, 1, 1)
    if with_noise:  # nicked from torch.nn.modules.conv:reset_parameters
        stdv = 1. / math.sqrt(n)
        noise = torch.empty_like(eye)
        noise.uniform_(-stdv, stdv)
        eye.add_(noise)

    print(eye[:10, :10, 0, 0])
    initialize_with_filter(conv_or_deconv, eye)


# Numpy -----------------------------------------------------------------------


def tensor_to_np(t):
    return t.detach().cpu().numpy()


def histogram(t, L):
    """
    A: If t is a list of tensors/np.ndarrays, B is executed for all, yielding len(ts) histograms, which are summed
    per bin
    B: convert t to numpy, count bins.
    :param t: tensor or list of tensor, each expected to be in [0, L)
    :param L: number of symbols
    :return: length-L array, containing at l the number of values mapping to to symbol l
    """
    if isinstance(t, list):
        ts = t
        histograms = np.stack((histogram(t, L) for t in ts), axis=0)  # get array (len(ts) x L)
        return np.sum(histograms, 0)
    assert 0 <= t.min() and t.max() < L, (t.min(), t.max())
    a = tensor_to_np(t)
    counts, _ = np.histogram(a, np.arange(L+1))  # +1 because np.histogram takes bin edges, including rightmost edge
    return counts


# Gradients --------------------------------------------------------------------


def get_total_grad_norm(params, norm_type=2):
    # nicked from torch.nn.utils.clip_grad_norm
    with torch.no_grad():
        total_norm = 0
        for p in params:
            if p.grad is None:
                continue
            param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm.item() ** norm_type
        total_norm = total_norm ** (1. / norm_type)
        return total_norm


def get_average_grad_norm(params, norm_type=2):
    """
    :param params: Assumed to be generator
    :param norm_type:
    """
    # nicked from torch.nn.utils.clip_grad_norm
    with torch.no_grad():
        average_norm = 0
        num_params = 0
        for p in params:
            if p.grad is None:
                continue
            average_norm += p.grad.data.norm(norm_type)
            num_params += 1
        if num_params == 0:
            return 0
        return average_norm / float(num_params)


# Datasets --------------------------------------------------------------------


class TruncatedDataset(Dataset):
    def __init__(self, dataset, num_elemens):
        assert len(dataset) >= num_elemens, 'Cannot truncate to {}: dataset has {} elements'.format(
                num_elemens, len(dataset))
        self.dataset = dataset
        self.num_elemens = num_elemens

    def __len__(self):
        return self.num_elemens

    def __getitem__(self, item):
        return self.dataset[item]


# Helpful modules --------------------------------------------------------------


class LambdaModule(nn.Module):
    def __init__(self, forward_lambda, name=''):
        super(LambdaModule, self).__init__()
        self.forward_lambda = forward_lambda
        self.description = 'LambdaModule({})'.format(name)

    def __repr__(self):
        return self.description

    def forward(self, x):
        return self.forward_lambda(x)


class ChannelToLogitsTranspose(nn.Module):
    def __init__(self, Cout, Lout):
        super(ChannelToLogitsTranspose, self).__init__()
        self.Cout = Cout
        self.Lout = Lout

    def forward(self, x):
        N, C, H, W = x.shape
        # unfold channel dimension to (Cout, Lout)
        # this fills up the Cout dimension first!
        x = x.view(N, self.Lout, self.Cout, H, W)
        return x

    def __repr__(self):
        return 'ChannelToLogitsTranspose(Cout={}, Lout={})'.format(self.Cout, self.Lout)


class LogitsToChannelTranspose(nn.Module):
    def __init__(self):
        super(LogitsToChannelTranspose, self).__init__()

    def forward(self, x):
        N, L, C, H, W = x.shape
        # fold channel, L dimension back to channel dim
        x = x.view(N, C * L, H, W)
        return x

    def __repr__(self):
        return 'LogitsToChannelTranspose()'


def channel_to_logits(x, Cout, Lout):
    N, C, H, W = x.shape
    # unfold channel dimension to (Cout, Lout)
    # this fills up the Cout dimension first!
    x = x.view(N, Lout, Cout, H, W)
    return x


def logits_to_channel(x):
    N, L, C, H, W = x.shape
    # fold channel, L dimension back to channel dim
    x = x.view(N, C * L, H, W)
    return x


class OneHot(nn.Module):
    """
    Take long tensor x of some shape (N,d1,d2,...,dN) containing integers in [0, L),
    produces one hot encoding `out` of out_shape (N, d1, ..., L, ..., dN), where out_shape[Ldim] = L, containing
        out[n, i, ..., l, ..., j] == {1 if x[n, i, ..., j] == l
                                      0 otherwise
    """
    def __init__(self, L, Ldim=1):
        super(OneHot, self).__init__()
        self.L = L
        self.Ldim = Ldim

    def forward(self, x):
        return one_hot(x, self.L, self.Ldim)


def one_hot(x, L, Ldim):
    """ add dim L at Ldim """
    assert Ldim >= 0 or Ldim == -1, f'Only supporting Ldim >= 0 or Ldim == -1: {Ldim}'
    out_shape = list(x.shape)
    if Ldim == -1:
        out_shape.append(L)
    else:
        out_shape.insert(Ldim, L)
    x = x.unsqueeze(Ldim)  # x must match # dims of outshape
    assert x.dim() == len(out_shape), (x.shape, out_shape)
    oh = torch.zeros(*out_shape, dtype=torch.float32, device=x.device)
    oh.scatter_(Ldim, x, 1)
    return oh


# ------------------------------------------------------------------------------


def assert_equal(t1, t2, show_num_wrong=3, names=None, msg=''):
    if t1.shape != t2.shape:
        raise AssertionError('Different shapes! {} != {}'.format(t1.shape, t2.shape))
    wrong = t1 != t2
    if not wrong.any():
        return
    if names is None:
        names = ('t1', 't2')
    wrong_idxs = wrong.nonzero()
    num_wrong = len(wrong_idxs)
    show_num_wrong = min(show_num_wrong, num_wrong)
    wrong_idxs = itertools.islice((tuple(i.tolist()) for i in wrong_idxs),
                                  show_num_wrong)
    err_msg = ' // '.join('{}: {}!={}'.format(idx, t1[idx], t2[idx])
                          for idx in wrong_idxs)
    raise AssertionError(('{} != {}: {}, and {}/{} other(s) '.format(
            names[0], names[1], err_msg, num_wrong - show_num_wrong, np.prod(t1.shape)) + msg).strip())


class BatchSummarizer(object):
    """
    Summarize values from multiple batches
    """
    def __init__(self, writer, global_step):
        self.writer = writer
        self.global_step = global_step
        self._values = defaultdict(list)

    def append(self, key, values):
        self._values[key].append(values)

    def output_summaries(self, strip_for_str='val'):
        strs = []
        for key, values in self._values.items():
            avg = sum(values) / len(values)
            self.writer.add_scalar(key, avg, self.global_step)
            key = key.replace(strip_for_str, '').strip('/')
            strs.append('{}={:.3f}'.format(key, avg))
        self.writer.file_writer.flush()
        return strs
