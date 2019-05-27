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
import argparse
import os
import shutil
from functools import total_ordering

import numpy as np
from fjcommon import os_ext
from fjcommon.assertions import assert_exc

from helpers.paths import has_image_ext, img_name


@total_ordering
class Testset(object):
    """
    Class the holds a reference to paths of images inside a folder
    """
    def __init__(self, root_dir_or_img, max_imgs=None, skip_hidden=False, append_id=None):
        """
        :param root_dir_or_img: Either a directory with images or the path of a single image.
        :param max_imgs: If given, subsample deterministically to only contain max_imgs
        :param skip_hidden: If given, skip images starting with '.'
        :param append_id: If given, append `append_id` to self.id
        :raises ValueError if root_dir is not a directory or does not contain images
        """
        self.root_dir_or_img = root_dir_or_img

        if os.path.isdir(root_dir_or_img):
            root_dir = root_dir_or_img
            self.name = os.path.basename(root_dir.rstrip('/'))
            self.ps = sorted(p for p in os_ext.listdir_paths(root_dir) if has_image_ext(p))
            if skip_hidden:
                self.ps = self._filter_hidden(self.ps)
            if max_imgs and max_imgs < len(self.ps):
                print('Subsampling to use {} imgs of {}...'.format(max_imgs, self.name))
                idxs = np.linspace(0, len(self.ps) - 1, max_imgs, dtype=np.int)
                self.ps = np.array(self.ps)[idxs].tolist()
                assert len(self.ps) == max_imgs
            assert_exc(len(self.ps) > 0, 'No images found in {}'.format(root_dir), ValueError)
            self.id = '{}_{}'.format(self.name, len(self.ps))
            self._str = 'Testset({}): in {}, {} images'.format(self.name, root_dir, len(self.ps))
        else:
            img = root_dir_or_img
            assert_exc(os.path.isfile(img), 'Does not exist: {}'.format(img), FileNotFoundError)
            self.name = os.path.basename(img)
            self.ps = [img]
            self.id = img
            self._str = 'Testset([{}]): 1 image'.format(self.name)
        if append_id:
            self.id += append_id

    def search_path(self):
        return self.root_dir_or_img

    def filter_filenames(self, filter_filenames):
        filename = lambda p: os.path.splitext(os.path.basename(p))[0]
        self.ps = [p for p in self.ps
                   if filename(p) in filter_filenames]
        assert_exc(len(self.ps) > 0, 'No files after filtering for {}'.format(filter_filenames))

    def iter_img_names(self):
        return map(img_name, self.ps)

    def iter_orig_paths(self):
        return self.ps

    def __len__(self):
        return len(self.ps)

    def __str__(self):
        return self._str

    def __repr__(self):
        return 'Testset({}): {} paths'.format(self.name, len(self.ps))

    # to enable sorting
    def __lt__(self, other):
        return self.id < other.id

    @staticmethod
    def _filter_hidden(ps):
        count_a = len(ps)
        ps = [p for p in ps if not os.path.basename(p).startswith('.')]
        count_b = len(ps)
        if count_b < count_a:
            print(f'NOTE: Filtered {count_a - count_b} hidden file(s).')
        return ps


def main():
    p = argparse.ArgumentParser('Copy deterministic subset of images to another directory.')
    p.add_argument('root_dir')
    p.add_argument('max_imgs', type=int)
    p.add_argument('out_dir')
    p.add_argument('--dry')
    p.add_argument('--verbose', '-v', action='store_true')
    flags = p.parse_args()
    os.makedirs(flags.out_dir, exist_ok=True)

    t = Testset(flags.root_dir, flags.max_imgs)

    def cp(p1, p2):
        if os.path.isfile(p2):
            print('Exists, skipping: {}'.format(p2))
            return
        if flags.verbose:
            print('cp {} -> {}'.format(p1, p2))
        if not flags.dry:
            shutil.copy(p1, p2)

    for p in t.iter_orig_paths():
        cp(p, os.path.join(flags.out_dir, os.path.basename(p)))