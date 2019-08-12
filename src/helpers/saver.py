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
import os
import time
import re
import shutil

import pytorch_ext as pe
from os.path import basename
import torch
from torch.optim import optimizer
from fjcommon.no_op import NoOp
from fjcommon import timer
from fjcommon.assertions import assert_exc


class _CheckpointTracker(object):
    """ out_dir is usally set via set_out_dir """
    def __init__(self, out_dir=None, ckpt_name_fmt='ckpt_{:010d}.pt', tmp_postfix='.tmp'):
        assert len(tmp_postfix)
        assert '.' in tmp_postfix
        m = re.search(r'{:0(\d+?)d}', ckpt_name_fmt)
        assert m, 'Expected ckpt_name_fmt to have an int specifier such as or {:09d} or {:010d}.'
        max_itr = 10 ** int(m.group(1)) - 1
        if max_itr < 10000000:  # ten million, should be enough
            print(f'Maximum iteration supported: {max_itr}')
        assert os.sep not in ckpt_name_fmt
        self.ckpt_name_fmt = ckpt_name_fmt
        self.ckpt_prefix = ckpt_name_fmt.split('{')[0]
        assert len(self.ckpt_prefix), 'Expected ckpt_name_fmt to start with a prefix before the format part!'
        self.tmp_postfix = tmp_postfix

        self._out_dir = None
        if out_dir is not None:
            self.set_out_dir(out_dir)

    def set_out_dir(self, out_dir):
        assert self._out_dir is None
        os.makedirs(out_dir, exist_ok=True)
        self._out_dir = out_dir

    def get_all_ckpts(self):
        """
        :return: All checkpoints in `self._out_dir`, sorted ascendingly by global_step.
        """
        return [os.path.join(self._out_dir, f)
                for f in sorted(os.listdir(self._out_dir))
                if f.startswith(self.ckpt_prefix)]

    def itr_ckpt(self):
        for ckpt_p in self.get_all_ckpts():
            yield self.get_itr_from_ckpt_p(ckpt_p), ckpt_p

    def get_ckpt_for_itr(self, itr):
        """
        Gets ckpt_itrc where itrc <= itr, i.e., the latest ckpt before `itr`.
        Special values: itr == -1 -> newest ckpt
        """
        ckpts = list(self.itr_ckpt())
        assert_exc(len(ckpts) > 0, 'No ckpts found in {}'.format(self._out_dir))
        if itr == -1:
            return ckpts[-1]
        first_itrc, _ = ckpts[0]
        assert_exc(first_itrc <= itr, 'Earliest ckpt {} is after {}'.format(first_itrc, itr))
        for itrc, ckpt_p in reversed(ckpts):
            if itrc <= itr:
                return itrc, ckpt_p
        raise ValueError('Unexpected, {}, {}'.format(itr, ckpts))

    def get_latest_ckpt(self):
        """
        :return: Most recent checkpoint. May be a temporary checkpoint.
        """
        return self.get_all_ckpts()[-1]

    def get_lastest_persistent_ckpt(self):
        """
        :return: Most recent persistent checkpoint. May be a temporary checkpoint.
        """
        candidates = [p for p in self.get_all_ckpts() if not p.endswith(self.tmp_postfix)]
        if len(candidates) == 0:
            raise ValueError('No persistent checkpoints')
        return candidates[-1]

    def _get_out_p(self, global_step, is_tmp):
        postfix = self.tmp_postfix if is_tmp else ''
        return os.path.join(self._out_dir, self.ckpt_name_fmt.format(global_step) + postfix)

    def get_itr_from_ckpt_p(self, ckpt_p):
        file_name = os.path.splitext(os.path.basename(ckpt_p))[0]
        assert self.ckpt_prefix in file_name
        itr_part = file_name.replace(self.ckpt_prefix, '')
        itr_part_digits_only = int(''.join(c for c in itr_part if c.isdigit()))
        return itr_part_digits_only



class Saver(_CheckpointTracker):
    """
    Saves ckpts:
    - ckpt_XXXXXXXX.pt.tmp
    If keep_tmp_last=None:
        Every `keep_every`-th ckpt is renamed to
        - ckpt_XXXXXXXX.pt
        and kept, the intermediate ones are removed. We call this a persistent checkpoint.
    else:
        Let C be the most recent persistent checkpoint.
        In addition to C being kept, the last `keep_tmp_last` temporary checkpoints before C are also kept.
        This means that always `keep_tmp_last` more checkpoints are kept than if keep_tmp_last=None
    """
    def __init__(self,
                 keep_tmp_itr: int, keep_every=10, keep_tmp_last=None,
                 out_dir=None, ckpt_name_fmt='ckpt_{:010d}.pt', tmp_postfix='.tmp',
                 verbose=False):
        """
        :param keep_every: keep every `keep_every`-th checkpoint, making it a persistent checkpoint
        :param keep_tmp_itr: keep checkpoint every `keep_tmp_itr` iterations.
        :param keep_tmp_last: Also keep the last `keep_tmp_last` temporary checkpoints before a persistent checkpoint.
        :param ckpt_name_fmt: filename, must include a format spec and some prefix before the format
        :param tmp_postfix: non-empty string to append to temporary checkpoints
        :param verbose: if True, print rename and remove info.
        """
        self.keep_every = keep_every
        self.keep_tmp_last = keep_tmp_last
        self.keep_tmp_itr = keep_tmp_itr
        self.ckpts_since_last_permanent = 0
        self.print = print if verbose else NoOp
        self.save_time_acc = timer.TimeAccumulator()
        super(Saver, self).__init__(out_dir, ckpt_name_fmt, tmp_postfix)

    def save(self, modules, global_step, force=False):
        """
        Save iff (force given or global_step % keep_tmp_itr == 0)
        :param modules: dictionary name -> nn.Module
        :param global_step: current step
        :return: bool, Whether previous checkpoints were removed
        """
        if not (force or (global_step % self.keep_tmp_itr == 0)):
            return False
        assert self._out_dir is not None
        current_ckpt_p = self._save(modules, global_step)
        self.ckpts_since_last_permanent += 1
        if self.ckpts_since_last_permanent == self.keep_every:
            self._remove_previous(current_ckpt_p)
            self.ckpts_since_last_permanent = 0
            return True
        return False

    def _save(self, modules, global_step):
        out_p = self._get_out_p(global_step, is_tmp=True)
        with self.save_time_acc.execute():
            torch.save({key: m.state_dict() for key, m in modules.items()}, out_p)
        return out_p

    def _remove_previous(self, current_ckpt_p):
        assert self.tmp_postfix in current_ckpt_p
        current_ckpt_p_non_tmp = current_ckpt_p.replace(self.tmp_postfix, '')
        self.print('{} -> {}'.format(basename(current_ckpt_p), basename(current_ckpt_p_non_tmp)))
        os.rename(current_ckpt_p, current_ckpt_p_non_tmp)
        keep_tmp_last = self.get_all_ckpts()[-(self.keep_tmp_last+1):] if self.keep_tmp_last else []
        for p in self.get_all_ckpts():
            if self.tmp_postfix in p and p not in keep_tmp_last:
                self.print('Removing {}...'.format(basename(p)))
                os.remove(p)
        self.print('Average save time: {:.3f}s'.format(self.save_time_acc.mean_time_spent()))


class Restorer(_CheckpointTracker):
    def restore_latest_persistent(self, net):
        return self.restore(net, self.get_lastest_persistent_ckpt())

    def restore(self, modules, ckpt_p, strict=True, restore_restart=False):
        print('Restoring {}... (strict={})'.format(ckpt_p, strict))
        map_location = None if pe.CUDA_AVAILABLE else 'cpu'
        state_dicts = torch.load(ckpt_p, map_location=map_location)
        # ---
        for key, m in modules.items():
            # optim implements its own load_state_dict which does not have the `strict` keyword...
            if isinstance(m, optimizer.Optimizer):
                if restore_restart:
                    print('Not restoring optimizer, --restore_restart given...')
                else:
                    try:
                        m.load_state_dict(state_dicts[key])
                    except ValueError as e:
                        raise ValueError('Error while restoring Optimizer:', str(e))
            else:
                try:
                    m.load_state_dict(state_dicts[key], strict=strict)
                except RuntimeError as e:  # loading error
                    for n, module in sorted(m.named_modules()):
                        print(n, module)
                    raise e
        return self.get_itr_from_ckpt_p(ckpt_p)



