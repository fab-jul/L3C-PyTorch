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

test.py log_dir log_dates images

test.py log_dir log_dates images --sample=samples

test.py log_dir log_dates images --write_to_files=l3c_out

This code uses a cache: If some experiment has already been tested for some iteration and crop and dataset,
we just print that (see TestID in multiscale_tester.py).

"""

import torch.backends.cudnn
torch.backends.cudnn.benchmark = True

import argparse
from operator import itemgetter

from helpers.aligned_printer import AlignedPrinter
from helpers.testset import Testset
from test.multiscale_tester import MultiscaleTester



def main():
    p = argparse.ArgumentParser()

    p.add_argument('log_dir', help='Directory of experiments. Will create a new folder, LOG_DIR_test, to save test '
                                   'outputs.')
    p.add_argument('log_dates', help='A comma-separated list, where each entry is a log_date, such as 0104_1345. '
                                     'These experiments will be tested.')
    p.add_argument('images', help='A comma-separated list, where each entry is either a directory with images or '
                                  'the path of a single image. Will test on all these images.')
    p.add_argument('--match_filenames', '-fns', nargs='+', metavar='FILTER',
                   help='If given, remove any images in the folders given by IMAGES that do not match any '
                        'of specified filter.')
    p.add_argument('--max_imgs_per_folder', '-m', type=int, metavar='MAX',
                   help='If given, only use MAX images per folder given in IMAGES. Default: None')
    p.add_argument('--crop', type=int, help='Crop all images to CROP x CROP squares. Default: None')

    p.add_argument('--names', '-n', type=str,
                   help='Comma separated list, if given, must be as long as LOG_DATES. Used for output. If not given, '
                        'will just print LOG_DATES as names.')

    p.add_argument('--overwrite_cache', '-f', action='store_true',
                   help='Ignore cached test outputs, and re-create.')
    p.add_argument('--reset_entire_cache', action='store_true',
                   help='Remove cache.')

    p.add_argument('--restore_itr', '-i', default='-1',
                   help='Which iteration to restore. -1 means latest iteration. Will use closest smaller if exact '
                        'iteration is not found. Default: -1')

    p.add_argument('--recursive', default='0',
                   help='Either an number or "auto". If given, the rgb configs with num_scales == 1 will '
                        'automatically be evaluated recursively (i.e., the RGB baseline). See _parse_recursive_flag '
                        'in multiscale_tester.py. Default: 0')

    p.add_argument('--sample', type=str, metavar='SAMPLE_OUT_DIR',
                   help='Sample from model. Store results in SAMPLE_OUT_DIR.')

    p.add_argument('--write_to_files', type=str, metavar='WRITE_OUT_DIR',
                   help='Write images to files in folder WRITE_OUT_DIR, with arithmetic coder. If given, the cache is '
                        'ignored and no test output is printed. Requires torchac to be installed, see README. Files '
                        'that already exist in WRITE_OUT_DIR are overwritten.')
    p.add_argument('--compare_theory', action='store_true',
                   help='If given with --write_to_files, will compare actual bitrate on disk to theoretical bitrate '
                        'given by cross entropy.')
    p.add_argument('--time_report', type=str, metavar='TIME_REPORT_PATH',
                   help='If given with --write_to_files, write a report of time needed for each component to '
                        'TIME_REPORT_PATH.')

    p.add_argument('--sort_output', '-s', choices=['testset', 'exp', 'itr', 'res'], default='testset',
                   help='How to sort the final summary. Possible values: "testset" to sort by '
                        'name of the testset // "exp" to sort by experiment log_date // "itr" to sort by iteration // '
                        '"res" to sort by result, i.e., show smaller first. Default: testset')

    flags = p.parse_args()

    if flags.compare_theory and not flags.write_to_files:
        raise ValueError('Cannot have --compare_theory without --write_to_files.')
    if flags.write_to_files and flags.sample:
        raise ValueError('Cannot have --write_to_files and --sample.')
    if flags.time_report and not flags.write_to_files:
        raise ValueError('--time_report only valid with --write_to_files.')

    testsets = [Testset(images_dir_or_image.rstrip('/'), flags.max_imgs_per_folder,
                        # Append flags.crop to ID so that it creates unique entry in cache
                        append_id=f'_crop{flags.crop}' if flags.crop else None)
                for images_dir_or_image in flags.images.split(',')]
    if flags.match_filenames:
        for ts in testsets:
            ts.filter_filenames(flags.match_filenames)

    splitter = ',' if ',' in flags.log_dates else '|'  # support tensorboard strings, too
    results = []
    log_dates = flags.log_dates.split(splitter)
    for log_date in log_dates:
        for restore_itr in map(int, flags.restore_itr.split(',')):
            print('Testing {} at {} ---'.format(log_date, restore_itr))
            tester = MultiscaleTester(log_date, flags, restore_itr)
            results += tester.test_all(testsets)

    # if --names was passed: will print 'name (log_date)'. otherwise, will just print 'log_date'
    if flags.names:
        names = flags.names.split(splitter) if flags.names else log_dates
        names_to_log_date = {log_date: f'{name} ({log_date})'
                             for log_date, name in zip(log_dates, names)}
    else:
        # set names to log_dates if --names is not given, i.e., we just output log_date
        names_to_log_date = {log_date: log_date for log_date in log_dates}
    if not flags.write_to_files:
        print('*** Summary:')
        with AlignedPrinter() as a:
            sortby = {'testset': 0, 'exp': 1, 'itr': 2, 'res': 3}[flags.sort_output]
            a.append('Testset', 'Experiment', 'Itr', 'Result')
            for testset, log_date, restore_itr, result in sorted(results, key=itemgetter(sortby)):
                a.append(testset.id,  names_to_log_date[log_date], str(restore_itr), result)


if __name__ == '__main__':
    main()
