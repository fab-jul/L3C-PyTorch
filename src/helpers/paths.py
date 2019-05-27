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
import glob

from PIL import Image
from torchvision import transforms as transforms

from helpers import logdir_helpers

from fjcommon.assertions import assert_exc

import pytorch_ext as pe


CKPTS_DIR_NAME = 'ckpts'
VAL_DIR_NAME = 'val'


IMG_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif'}


def get_ckpts_dir(experiment_dir, ensure_exists=True):
    ckpts_p = os.path.join(experiment_dir, CKPTS_DIR_NAME)
    if ensure_exists:
        assert_exc(os.path.isdir(ckpts_p), 'Not found: {}'.format(ckpts_p))
    return ckpts_p


def get_experiment_dir(log_dir, experiment_spec):
    """
    experiment_spec: if is a logdate, find correct full path in log_dir, otherwise assume logdir/experiment_spec exists
    :return experiment dir, no slash at the end. containing /ckpts
    """
    if logdir_helpers.is_log_date(experiment_spec):  # assume that log_dir/restore* matches
        assert_exc(log_dir is not None, 'Can only infer experiment_dir from log_date if log_dir is not None')
        restore_dir_glob = os.path.join(log_dir, experiment_spec + '*')
        restore_dir_possible = glob.glob(restore_dir_glob)
        assert_exc(len(restore_dir_possible) == 1, 'Expected one match for {}, got {}'.format(
                restore_dir_glob, restore_dir_possible))
        experiment_spec = restore_dir_possible[0]
    else:
        experiment_spec = os.path.join(log_dir, experiment_spec)
    experiment_dir = experiment_spec.rstrip(os.path.sep)
    assert_exc(os.path.isdir(experiment_dir), 'Invalid experiment_dir: {}'.format(experiment_dir))
    return experiment_dir


def img_name(img_p):
    return os.path.splitext(os.path.basename(img_p))[0]


def has_image_ext(p):
    return os.path.splitext(p)[1].lower() in IMG_EXTENSIONS

