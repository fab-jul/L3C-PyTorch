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
import numpy as np
import torch
import os
import time
import pytorch_ext as pe
from collections import defaultdict
from collections import namedtuple


_NO_CUDA_SYNC_OVERWRITE = int(os.environ.get('NO_CUDA_SYNC', 0)) == 1


if _NO_CUDA_SYNC_OVERWRITE or not pe.CUDA_AVAILABLE:
    sync = lambda: None
else:
    sync = torch.cuda.synchronize



class StackLogger(object):
    Entry = namedtuple('Entry', ['fmt_str', 'logs'])
    CombineCtx = namedtuple('CombineCtx', ['prefixes_of_created_entries', 'fmt_str'])

    def __init__(self, default_fmt_str='{}'):
        self.default_fmt_str = default_fmt_str

        self.logs = defaultdict(list)
        self._order = []
        self._prefixes = []

        self._combine_ctx = None
        self._global_skip = False

    @contextmanager
    def skip(self, flag):
        self._global_skip = flag
        yield
        self._global_skip = False

    @contextmanager
    def prefix_scope(self, p):
        self._prefixes.append(p)
        yield
        del self._prefixes[-1]

    @contextmanager
    def combine(self, fmt_str):
        if self._combine_ctx is not None:
            raise ValueError('Already in combine!')
        self._combine_ctx = set(), fmt_str
        yield
        self._combine_ctx = None

    @contextmanager
    def prefix_scope_combine(self, prefix, fmt_str):
        with self.prefix_scope(prefix):
            with self.combine(fmt_str):
                yield

    def log(self, name, msg):
        if self._global_skip:
            return
        prefix = ' '.join(self._prefixes + [name])
        if prefix not in self.logs:
            self._order.append(prefix)
        logs = self._get_log_entry_list(prefix)
        logs.append(msg)

    def _get_log_entry_list(self, prefix):
        if self._combine_ctx is None:
            # Always create a new entry
            return self.logs[prefix]

        prefixes_of_created_entries, fmt_str = self._combine_ctx
        if prefix not in prefixes_of_created_entries:
            prefixes_of_created_entries.add(prefix)
            return self._append_entry(prefix, fmt_str).logs

        assert len(self.logs[prefix]) > 0
        return self.logs[prefix][-1].logs

    def _append_entry(self, prefix, fmt_str):
        log_entry = StackLogger.Entry(fmt_str, logs=[])
        self.logs[prefix].append(log_entry)
        return log_entry


class StackTimeLogger(StackLogger):
    def __init__(self, default_fmt_str='{:.5f}'):
        super(StackTimeLogger, self).__init__(default_fmt_str)


    def get_mean_strs(self):
        for prefix in self._order:
            entries = self.logs[prefix]
            first_entry = entries[0]
            if isinstance(first_entry, StackLogger.Entry):
                num_values_per_entry = len(first_entry.logs)
                means = np.zeros(num_values_per_entry, dtype=np.float)
                for e in entries:
                    means += np.array(e.logs)
                means = means / len(entries)
                yield self._to_str(prefix, first_entry.fmt_str, means)
            else:  # entries is just a list
                mean = np.mean(entries)
                yield self._to_str(prefix, self.default_fmt_str, mean)

    def get_last_strs(self):
        for prefix in self._order:
            entries = self.logs[prefix]
            last_entry = entries[-1]
            if isinstance(last_entry, StackLogger.Entry):
                yield self._to_str(prefix, last_entry.fmt_str, last_entry.logs)
            else:
                yield self._to_str(prefix, self.default_fmt_str, last_entry)

    @contextmanager
    def run(self, name):
        sync()
        start = time.time()
        yield
        sync()
        duration = time.time() - start
        self.log(name, duration)

    @staticmethod
    def _to_str(prefix, fmt_str, values):
        should_iter = isinstance(values, (list, np.ndarray))
        values_str = (fmt_str.format(values) if not should_iter
                      else '/'.join(fmt_str.format(i, v)
                                    for i, v in enumerate(values)))
        return prefix + ': ' + values_str


def test_stack_time_logger():
    t = StackTimeLogger()
    for i in [1, 2]:
        with t.prefix_scope('foo'):
            with t.prefix_scope('bar'):
                with t.run('setup'):
                    time.sleep(0.1)
                with t.combine('c{}: {:.5f}'):
                    for c in range(5):
                        with t.run('run'):
                            time.sleep(c * 0.01)
    from pprint import pprint
    pprint(t.logs)
    pprint(t._order)
    print('\n'.join(t.get_mean_strs()))
    print('...')
    print('\n'.join(t.get_last_strs()))

