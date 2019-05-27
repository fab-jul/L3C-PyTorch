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

Contains the neat SummarizableModule class. It's a replacement for nn.Module. If in a tree of nn.Modules, the root
is a SummarizableModule and some leaves are also, you can call `register_summarizer` on the root, and this will add
an instance of `Summarizer` to every SummarizableModule in the tree.
Then, the module can call stuff like:

    self.summarizer.register_scalars('train', {'lr': self.lr})

    def _plot(plt):
        plt.plot(x, y)

    self.summarizer.register_figures('val', {'niceplot': _plot})

"""
from contextlib import contextmanager

import torch
from fjcommon.assertions import assert_exc
from fjcommon.no_op import NoOp
from torch import nn as nn


class _GlobalStepDependable(object):
    def __init__(self):
        self.global_step = None
        self.enabled_prefix = None

    def enable(self, prefix, global_step):
        """ Enable logging of prefix """
        assert_exc(isinstance(prefix, str), 'prefix must be str, got {}'.format(prefix))
        assert_exc(prefix[-1] != '/')
        self.enabled_prefix = prefix
        self.global_step = global_step

    def disable(self):
        self.enabled_prefix = None

    @contextmanager
    def maybe_enable(self, prefix, flag, global_step):
        if flag:
            self.enable(prefix, global_step)
        yield
        self.disable()


def normalize_to_0_1(t):
    return t.add(-t.min()).div(t.max() - t.min() + 1e-5)


class Summarizer(_GlobalStepDependable):
    def __init__(self, sw):
        super(Summarizer, self).__init__()
        self.sw = sw

    def register_scalars(self, prefix, values):
        """
        :param prefix: Prefix to use in TensorBoard
        :param values: A dictionary of name -> value, where value can be callable (useful if it is expensive)
        """
        if self.enabled_prefix is None:
            return
        if prefix == 'auto':
            prefix = self.enabled_prefix
        if prefix == self.enabled_prefix:
            for name, value in values.items():
                self.sw.add_scalar(prefix + '/' + name, _convert_if_callable(value), self.global_step)

    def register_figures(self, prefix, creators):
        """
        :param prefix: Prefix to use in TensorBoard
        :param creators: plot_name -> (plt -> None)
        """
        if self.enabled_prefix is None:
            return
        if prefix == 'auto':
            prefix = self.enabled_prefix
        if prefix == self.enabled_prefix:
            for name, creator in creators.items():
                with self.sw.add_figure_ctx(prefix + '/' + name, self.global_step) as plt:
                    creator(plt)

    def register_images(self, prefix, imgs, normalize=False):  # , early_only=False):
        """
        :param prefix: Prefix to use in TensorBoard
        :param imgs: A dictionary of name -> img, where img can be callable (useful if it is expensive)
        :param normalize: If given, will normalize imgs to [0,1]
        """
        if self.enabled_prefix is None:
            return
        if prefix == 'auto':
            prefix = self.enabled_prefix
        if prefix == self.enabled_prefix:
            for name, img in imgs.items():
                img = _convert_if_callable(img)
                if normalize:
                    img = normalize_to_0_1(img)
                self.sw.add_image(prefix + '/' + name, img, self.global_step)


def _convert_if_callable(v):
    if hasattr(v, '__call__'):  # python 3 only
        return v()
    return v


class SummarizableModule(nn.Module):
    def __init__(self):
        super(SummarizableModule, self).__init__()
        self.summarizer = NoOp

    def forward(self, *input):
        raise NotImplementedError

    def register_summarizer(self, summarizer: Summarizer):
        for m in iter_modules_of_class(self, SummarizableModule):
            m.summarizer = summarizer


def iter_modules_of_class(root_module: nn.Module, cls):
    """
    Helpful for extending nn.Module. How to use:
    1. define new nn.Module subclass with some new instance methods, cls
    2. make your root module inherit from cls
    3. make some leaf module inherit from cls
    """
    for m in root_module.modules():
        if isinstance(m, cls):
            yield m


# Tests ------------------------------------------------------------------------


def test_submodules():
    class _T(nn.Module):
        def __init__(self):
            super(_T, self).__init__()
            self.foo = []

        def register_foo(self, f):
            self.foo.append(f)

        def get_all(self):
            for m_ in iter_modules_of_class(self, _T):
                yield from m_.foo

    class _SomethingWithTs(nn.Module):
        def __init__(self):
            super(_SomethingWithTs, self).__init__()
            self.a_t = _T()

    class _M(_T):  # first T
        def __init__(self):
            super(_M, self).__init__()
            self.conv = nn.Conv2d(1, 2, 3)
            self.t = _T()  # here
            self.t.register_foo(1)
            self.list = nn.ModuleList(
                    [nn.Conv2d(1, 2, 3),
                     _T()])  # here
            inner = _T()
            self.seq = nn.Sequential(
                    nn.Conv2d(1, 2, 3),
                    inner,  # here
                    _SomethingWithTs())  # here
            inner.register_foo(2)

    m = _M()
    all_ts = list(iter_modules_of_class(m, _T))
    assert len(all_ts) == 5, all_ts
    assert list(m.get_all()) == [1, 2]
