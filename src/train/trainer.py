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

General Trainer class, subclassed by MultiscaleTrainer

"""

from collections import namedtuple

import torch

import torchvision
from fjcommon import timer
from fjcommon.no_op import NoOp

from helpers import logdir_helpers
import vis.safe_summary_writer
from helpers.saver import Saver
from helpers.global_config import global_config
import itertools

from train.train_restorer import TrainRestorer


LogConfig = namedtuple('LogConfig', ['log_train', 'log_val', 'log_train_heavy'])


class TimedIterator(object):
    def __init__(self, it):
        self.t = timer.TimeAccumulator()
        self.it = iter(it)

    def __iter__(self):
        return self

    def __next__(self):
        with self.t.execute():
            return next(self.it)


# TODO: allow "restart last epoch" or sth
class TrainingSetIterator(object):
    """ Implements skipping to a certain iteration """
    def __init__(self, skip_to_itr, dl_train):
        self.skip_to_itr = skip_to_itr
        self.dl_train = dl_train
        self.epoch_len = len(self.dl_train)

    def epochs_to_skip(self):
        if self.skip_to_itr:
            skip_epochs, skip_batches = self.skip_to_itr // self.epoch_len, self.skip_to_itr % self.epoch_len
            return skip_epochs, skip_batches
        return 0, 0

    def iterator(self, epoch):
        """ :returns an iterator over tuples (itr, batch) """
        skip_epochs, skip_batches = self.epochs_to_skip()
        if epoch < skip_epochs:
            print('Skipping epoch {}'.format(epoch))
            return []  # nothing to iterate
        if epoch > skip_epochs or (epoch == skip_epochs and skip_batches == 0):  # iterate like normal
            return enumerate(self.dl_train, epoch * len(self.dl_train))
        # if we get to here, we are in the first epoch which we should not skip, so skip `skip_batches` batches
        it = iter(self.dl_train)
        for i in range(skip_batches):
            print('\rDropping batch {: 10d}...'.format(i), end='')
            if not global_config.get('drop_batches', False):
                # would be nice to not load images but this is hard to do as DataLoader caches Dataset's respondes,
                # might even be immutable?
                next(it)  # drop batch
        print(' -- dropped {} batches'.format(skip_batches))
        return enumerate(it, epoch * len(self.dl_train) + skip_batches)


class AbortTrainingException(Exception):
    pass


class Trainer(object):
    def __init__(self, dl_train, dl_val, optims, net, sw: vis.safe_summary_writer.SafeSummaryWriter,
                 max_epochs, log_config: LogConfig, saver: Saver=None, skip_to_itr=None):

        assert isinstance(optims, list)

        self.dl_train = dl_train
        self.dl_val = dl_val
        self.optims = optims
        self.net = net
        self.sw = sw
        self.max_epochs = max_epochs
        self.log_config = log_config
        self.saver = saver if saver is not None else NoOp

        self.skip_to_itr = skip_to_itr

    def continue_from(self, ckpt_dir):
        pass

    def train(self):
        log_train, log_val, log_train_heavy = self.log_config

        dl_train_it = TrainingSetIterator(self.skip_to_itr, self.dl_train)

        _print_unused_global_config()

        try:
            for epoch in (range(self.max_epochs) if self.max_epochs else itertools.count()):
                self.print_epoch_sep(epoch)
                self.prepare_for_epoch(epoch)
                t = TimedIterator(dl_train_it.iterator(epoch))
                for i, img_batch in t:
                    for o in self.optims:
                        o.zero_grad()
                    should_log = (i > 0 and i % log_train == 0)
                    should_log_heavy = (i > 0 and (i / log_train_heavy) % log_train == 0)
                    self.train_step(i, img_batch,
                                    log=should_log,
                                    log_heavy=should_log_heavy,
                                    load_time='[{:.2e} s/batch load]'.format(t.t.mean_time_spent()) if should_log else None)
                    self.saver.save(self.modules_to_save(), i)

                    if i > 0 and i % log_val == 0:
                        self._eval(i)
        except AbortTrainingException as e:
            print('Caught {}'.format(e))
            return

    def _eval(self, i):
        self.net.eval()
        with torch.no_grad():
            self.validation_loop(i)
        self.net.train()

    def debug(self):
        print('Debug ---')
        _print_unused_global_config()
        self.prepare_for_epoch(0)
        self.train_step(0, next(iter(self.dl_train)),
                        log=True, log_heavy=True, load_time=0)
        self._eval(101)

    def print_epoch_sep(self, epoch):
        print('-' * 80)
        print(' EPOCH {}'.format(epoch))
        print('-' * 80)

    def modules_to_save(self):
        """ used to save and restore. Should return a dictionary module_name -> nn.Module """
        raise NotImplementedError()

    def train_step(self, i, img_batch, log, log_heavy, load_time=None):
        raise NotImplementedError()

    def validation_loop(self, i):
        raise NotImplementedError()

    def prepare_for_epoch(self, epoch):
        pass

    def add_filter_summaray(self, tag, p, global_step):
        if len(p.shape) == 1:  # bias
            C, = p.shape
            p = p.reshape(C, 1).expand(C, C).reshape(C, C, 1, 1)

        try:
            _, _, H, W = p.shape
        except ValueError:
            if global_step == 0:
                print('INFO: Cannot unpack {} ({})'.format(p.shape, tag))
            return

        if H == W == 1:  # 1x1 conv
            p = p[:, :, 0, 0]
            filter_vis = torchvision.utils.make_grid(p, normalize=True)
            self.sw.add_image(tag, filter_vis, global_step)

    @staticmethod
    def update_lrs(epoch, optims_lrs_facs_interval):
        raise DeprecationWarning('use lr_schedule.py')
        for optimizer, init_lr, decay_fac, interval_epochs in optims_lrs_facs_interval:
            if decay_fac is None:
                continue
            Trainer.exp_lr_scheduler(optimizer, epoch, init_lr, decay_fac, interval_epochs)

    @staticmethod
    def exp_lr_scheduler(optimizer, epoch, init_lr, decay_fac=0.1, interval_epochs=7):
        raise DeprecationWarning('use lr_schedule.py')
        lr = init_lr * (decay_fac ** (epoch // interval_epochs))
        print('LR = {}'.format(lr))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def get_lrs(self):
        for optim in self.optims:
            for param_group in optim.param_groups:
                yield param_group['lr']

    def maybe_restore(self, restorer: TrainRestorer):
        """
        :return: skip_to_itr
        """
        if restorer is None:
            return None  # start from 0
        restore_itr = restorer.restore_desired_ckpt(self.modules_to_save())  # TODO: allow arbitrary ckpts
        if restorer.restart_at_zero:
            return 0
        return restore_itr

    @staticmethod
    def get_log_dir(log_dir_root, rel_paths, restorer, strip_ext='.cf'):
        if not restorer or not restorer.restore_continue:
            log_dir = logdir_helpers.create_unique_log_dir(
                    rel_paths, log_dir_root, strip_ext=strip_ext, postfix=global_config.values())
            print('Created {}...'.format(log_dir))
        else:
            log_dir = restorer.get_log_dir()
            print('Using {}...'.format(log_dir))
        return log_dir


def _print_unused_global_config(ignore=None):
    """ For safety, print parameters that were passed with -p but never used during construction of graph. """
    if not ignore:
        ignore = []
    unused = [u for u in global_config.get_unused_params() if u not in ignore]
    if unused:
        raise ValueError('Unused params:\n- ' + '\n- '.join(unused))

