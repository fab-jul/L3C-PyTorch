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

from fjcommon.assertions import assert_exc

from vis.figure_plotter import PlotToArray


SPEC_SEP = '_'


def from_spec(s, initial_lr, optims, epoch_len):
    """
    grammar: one of
        none
        exp FAC (iITR|eEPOCH)
        cos lrmax lrmin (iITR|eEPOCH)       # time to finish

    Example:
        exp_0.5_e8_warm_20_0.75_e1
    """
    if s == 'none':
        return ConstantLRSchedule()

    schedule_kind, s = s.split(SPEC_SEP, 1)

    def _parse_cos_spec():
        lrmax, lrmin, T = s.split(SPEC_SEP)
        kind, T = T[0], T[1:]
        assert_exc(kind in ('i', 'e'), 'Invalid spec: {}'.format(s))
        T_itr = int(T) if kind == 'i' else None
        T_epoch = float(T) if kind == 'e' else None
        return CosineDecayLRSchedule(optims, float(lrmax), float(lrmin), T_itr, T_epoch, epoch_len)

    p = {'exp': lambda: _parse_exp_spec(s, optims, initial_lr, epoch_len),
         'cos': _parse_cos_spec
         }[schedule_kind]
    return p()


# ------------------------------------------------------------------------------


class LRSchedule(object):
    def __init__(self, optims):
        self.optims = optims
        self._current_lr = None

    def _get_lr(self, i):
        raise NotImplementedError()

    def update(self, i):
        lr = self._get_lr(i)
        if lr == self._current_lr:
            return
        for optim in self.optims:
            for pg in optim.param_groups:
                pg['lr'] = lr
        self._current_lr = lr


class ConstantLRSchedule(object):
    def update(self, i):  # no-op
        pass



class ExponentialDecayLRSchedule(LRSchedule):
    def __init__(self, optims, initial, decay_fac,
                 decay_interval_itr=None, decay_interval_epoch=None, epoch_len=None,
                 warm_restart=None,
                 warm_restart_schedule=None):
        super(ExponentialDecayLRSchedule, self).__init__(optims)
        assert_exc((decay_interval_itr is not None) ^ (decay_interval_epoch is not None), 'Need either iter or epoch')
        if decay_interval_epoch:
            assert epoch_len is not None
            decay_interval_itr = int(decay_interval_epoch * epoch_len)
            if warm_restart:
                warm_restart = int(warm_restart * epoch_len)
        self.initial = initial
        self.decay_fac = decay_fac
        self.decay_every_itr = decay_interval_itr

        self.warm_restart_itr = warm_restart
        self.warm_restart_schedule = warm_restart_schedule

        self.last_warm_restart = 0

    def _get_lr(self, i):
        if i > 0 and self.warm_restart_itr and ((i - self.last_warm_restart) % self.warm_restart_itr) == 0:
            if i != self.last_warm_restart:
                self._warm_restart()
                self.last_warm_restart = i
        i -= self.last_warm_restart
        num_decays = i // self.decay_every_itr
        return self.initial * (self.decay_fac ** num_decays)

    def _warm_restart(self):
        print('WARM restart')
        if self.warm_restart_schedule:
            self.initial = self.warm_restart_schedule.initial
            self.decay_fac = self.warm_restart_schedule.decay_fac
            self.decay_every_itr = self.warm_restart_schedule.decay_every_itr
            self.warm_restart_itr = self.warm_restart_schedule.warm_restart_itr
            self.warm_restart_schedule = self.warm_restart_schedule.warm_restart_schedule


class CosineDecayLRSchedule(LRSchedule):
    def __init__(self, optims, lrmax, lrmin, T_itr, T_epoch, epoch_len):
        super(CosineDecayLRSchedule, self).__init__(optims)
        self.lrmax = lrmax
        self.lrmin = lrmin
        if T_itr is None:
            assert epoch_len is not None
            T_itr = int(T_epoch * epoch_len)
        self.Ti = T_itr
        self.epoch_len = epoch_len

    def _get_lr(self, i):
        Tcur = (i % self.Ti) / (2 * self.Ti)
        return self.lrmin + (self.lrmax - self.lrmin)*(np.cos(np.pi * Tcur))


def _parse_exp_spec(s, optims, initial_lr, epoch_len):
    if s.count(SPEC_SEP) > 2:
        fac, interval, warm, warm_start, warm_fac, warm_interval = s.split(SPEC_SEP)
        assert warm == 'warm'
        warm_start = int(warm_start)
        warm_schedule = _parse_exp_spec(SPEC_SEP.join([warm_fac, warm_interval]), optims, initial_lr, epoch_len)
    else:
        fac, interval = s.split(SPEC_SEP)
        warm_start, warm_schedule = None, None
    kind, interval = interval[0], interval[1:]
    assert_exc(kind in ('i', 'e'), 'Invalid spec: {}'.format(s))
    decay_interval_itr = int(interval) if kind == 'i' else None
    decay_interval_epoch = float(interval) if kind == 'e' else None
    return ExponentialDecayLRSchedule(
            optims, initial_lr, float(fac), decay_interval_itr, decay_interval_epoch, epoch_len,
            warm_restart=warm_start, warm_restart_schedule=warm_schedule)

