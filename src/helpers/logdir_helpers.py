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
import glob
from collections import namedtuple
from datetime import datetime, timedelta

import fasteners
import re
import os
from os import path

_LOG_DATE_FORMAT = "%m%d_%H%M"
_RESTORE_PREFIX = 'r@'


def create_unique_log_dir(config_rel_paths, log_dir_root, line_breaking_chars_pat=r'[-]',
                          postfix=None, restore_dir=None, strip_ext=None):
    """
    0117_1704 repr@soa3_med_8e*5_deePer_b50_noHM_C16 repr@v2_res_shallow r@0115_1340
    :param config_rel_paths: paths to the configs, relative to the config root dir
    :param log_dir_root: In this directory, all log dirs are stored. Created if needed.
    :param line_breaking_chars_pat:
    :param postfix: appended to the returned log dir
    :param restore_dir: if given, expected to be a log dir. the JOB_ID of that will be appended
    :param strip_ext: if given, do not store extension `strip_ext` of config_rel_paths
    :return: path to a newly created directory
    """
    if any('@' in config_rel_path for config_rel_path in config_rel_paths):
        raise ValueError('"@" not allowed in paths, got {}'.format(config_rel_paths))

    if strip_ext:
        assert all(strip_ext in c for c in config_rel_paths)
        config_rel_paths = [c.replace(strip_ext, '') for c in config_rel_paths]

    def prep_path(p):
        p = p.replace(path.sep, '@')
        return re.sub(line_breaking_chars_pat, '*', p)

    postfix_dir_name = ' '.join(map(prep_path, config_rel_paths))
    if restore_dir:
        _, restore_job_component = _split_log_dir(restore_dir)
        restore_job_id = log_date_from_log_dir(restore_job_component)
        postfix_dir_name += ' {restore_prefix}{job_id}'.format(
                restore_prefix=_RESTORE_PREFIX, job_id=restore_job_id)
    if postfix:
        if isinstance(postfix, list):
            postfix = ' '.join(postfix)
        postfix_dir_name += ' ' + postfix
    return _mkdir_threadsafe_unique(log_dir_root, datetime.now(), postfix_dir_name)


LogDirComps = namedtuple('LogDirComps', ['config_paths', 'postfix'])


def parse_log_dir(log_dir, configs_dir, base_dirs, append_ext=''):
    """
    Given a log_dir produced by `create_unique_log_dir`, return the full paths of all configs used.
    The log dir has thus the following format
            {now} {netconfig} {probconfig} [r@XXXX_YYYY] [{postfix} {postfix}]

    :param log_dir: the log dir to parse
    :param configs_dir: the root config dir, where all the configs live
    :param base_dirs: Prefixed to the paths of the configs, e.g., ['ae', 'pc']
    :return: all config paths, as well as the postfix if one was given
    """
    base_dirs = [path.join(configs_dir, base_dir) for base_dir in base_dirs]
    log_dir = path.basename(log_dir.strip(path.sep))

    comps = log_dir.split(' ')
    assert is_log_date(comps[0]), 'Invalid log_dir: {}'.format(log_dir)

    assert len(comps) > len(base_dirs), 'Expected a base dir for every component, got {} and {}'.format(
            comps, base_dirs)
    config_components = comps[1:(1+len(base_dirs))]
    has_restore = any(_RESTORE_PREFIX in c for c in comps)
    postfix = comps[1+len(base_dirs)+has_restore:]

    def get_real_path(base, prepped_p):
        p_glob = prepped_p.replace('@', path.sep)
        p_glob = path.join(base, p_glob) + append_ext  # e.g., ae_configs/p_glob.cf
        glob_matches = glob.glob(p_glob)
        # We always only replace one character with *, so filter for those.
        # I.e. lr1e-5 will become lr1e*5, which will match lr1e-5 but also lr1e-4.5
        glob_matches_of_same_len = [g for g in glob_matches if len(g) == len(p_glob)]
        if len(glob_matches_of_same_len) != 1:
            raise ValueError('Cannot find config on disk: {} (matches: {})'.format(p_glob, glob_matches_of_same_len))
        return glob_matches_of_same_len[0]

    return LogDirComps(
            config_paths=tuple(get_real_path(base_dir, comp)
                               for base_dir, comp in zip(base_dirs, config_components)),
            postfix=tuple(postfix) if postfix else None)


# ------------------------------------------------------------------------------


def _split_log_dir(log_dir):
    """
    given
        some/path/to/job/dir/0101_1818 ae_config pc_config/ckpts
    or
        some/path/to/job/dir/0101_1818 ae_config pc_config
    returns
        tuple some/path/to/job/dir, 0101_1818 ae_config pc_config
    """
    log_dir_root = []
    job_component = None

    for comp in log_dir.split(path.sep):
        try:
            log_date_from_log_dir(comp)
            job_component = comp
            break  # this component is an actual log dir. stop and return components
        except ValueError:
            log_dir_root.append(comp)

    assert job_component is not None, 'Invalid log_dir: {}'.format(log_dir)
    return path.sep.join(log_dir_root), job_component


def _mkdir_threadsafe_unique(log_dir_root, log_date, postfix_dir_name):
    os.makedirs(log_dir_root, exist_ok=True)
    # Make sure only one process at a time writes into log_dir_root
    with fasteners.InterProcessLock(os.path.join(log_dir_root, 'lock')):
        return _mkdir_unique(log_dir_root, log_date, postfix_dir_name)


def _mkdir_unique(log_dir_root, log_date, postfix_dir_name):
    log_date_str = log_date.strftime(_LOG_DATE_FORMAT)
    if _log_dir_with_log_date_exists(log_dir_root, log_date):
        print('Log dir starting with {} exists...'.format(log_date_str))
        return _mkdir_unique(log_dir_root, log_date + timedelta(minutes=1), postfix_dir_name)

    log_dir = path.join(log_dir_root, '{log_date_str} {postfix_dir_name}'.format(
        log_date_str=log_date_str,
        postfix_dir_name=postfix_dir_name).strip())
    os.makedirs(log_dir)
    return log_dir


def _log_dir_with_log_date_exists(log_dir_root, log_date):
    log_date_str = log_date.strftime(_LOG_DATE_FORMAT)
    all_log_dates = set()
    for log_dir in os.listdir(log_dir_root):
        try:
            all_log_dates.add(log_date_from_log_dir(log_dir))
        except ValueError:
            continue
    return log_date_str in all_log_dates


def log_date_from_log_dir(log_dir):
    # extract {log_date} from LOG_DIR/{log_date} {netconfig} {probconfig}
    possible_log_date = os.path.basename(log_dir).split(' ')[0]
    if not is_log_date(possible_log_date):
        raise ValueError('Invalid log dir: {}'.format(log_dir))
    return possible_log_date


def is_log_dir(log_dir):
    try:
        log_date_from_log_dir(log_dir)
        return True
    except ValueError:
        return False


def is_log_date(possible_log_date):
    try:
        datetime.strptime(possible_log_date, _LOG_DATE_FORMAT)
        return True
    except ValueError:
        return False
