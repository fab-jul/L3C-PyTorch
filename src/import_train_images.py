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
import multiprocessing
import os
import random
import shutil
import time
import warnings
from os.path import join

import PIL
import numpy as np
import skimage.color
from PIL import Image

from helpers.paths import IMG_EXTENSIONS

# TO SPEED THINGS UP: run on CPU cluster! We use task_array for this.
# task_array is not released. It's used by us to batch process on our servers. Feel free to replace with whatever you
# use. Make sure to set NUM_TASKS (number of concurrent processes) and set job_enumerate to a function that takes an
# iterable and only yield elements to be processed by the current process.
try:
    from task_array import NUM_TASKS, job_enumerate
except ImportError:
    NUM_TASKS = 1
    job_enumerate = enumerate

warnings.filterwarnings("ignore")


random.seed(123)


_NUM_PROCESSES = int(os.environ.get('NUM_PROCESS', 16))
_DEFAULT_MAX_SCALE = 0.8


get_fn = lambda p_: os.path.splitext(os.path.basename(p_))[0]


def iter_images(root_dir, num_folder_levels=0):
    fns = sorted(os.listdir(root_dir))
    for fn in fns:
        if num_folder_levels > 0:
            dir_p = os.path.join(root_dir, fn)
            if os.path.isdir(dir_p):
                print('Recursing into', fn)
                yield from iter_images(dir_p, num_folder_levels - 1)
            continue
        _, ext = os.path.splitext(fn)
        if ext.lower() in IMG_EXTENSIONS:
            yield os.path.join(root_dir, fn)


class Helper(object):
    def __init__(self, out_dir_clean, out_dir_discard, min_res: int):
        print(f'Creating {out_dir_clean}, {out_dir_discard}...')
        os.makedirs(out_dir_clean, exist_ok=True)
        os.makedirs(out_dir_discard, exist_ok=True)
        self.out_dir_clean = out_dir_clean
        self.out_dir_discard = out_dir_discard

        print('Getting images already processed...', end=" ", flush=True)
        self.images_cleaned = set(map(get_fn, os.listdir(out_dir_clean)))
        self.images_discarded = set(map(get_fn, os.listdir(out_dir_discard)))
        print(f'-> Found {len(self.images_cleaned) + len(self.images_discarded)} images.')

        self.min_res = min_res

    def process_all_in(self, input_dir):
        images_dl = iter_images(input_dir)  # generator of paths

        # files this job should compress
        files_of_job = [p for _, p in job_enumerate(images_dl)]
        # files that were compressed already by somebody (i.e. this job earlier)
        processed_already = self.images_cleaned | self.images_discarded
        # resulting files to be compressed
        files_of_job = [p for p in files_of_job if get_fn(p) not in processed_already]

        N = len(files_of_job)
        if N == 0:
            print('Everything processed / nothing to process.')
            return

        num_process = 2 if NUM_TASKS > 1 else _NUM_PROCESSES
        print(f'Processing {N} images using {num_process} processes in {NUM_TASKS} tasks...')

        start = time.time()
        predicted_time = None
        with multiprocessing.Pool(processes=num_process) as pool:
            for i, clean in enumerate(pool.imap_unordered(self.process, files_of_job)):
                if i > 0 and i % 100 == 0:
                    time_per_img = (time.time() - start) / (i + 1)
                    time_remaining = time_per_img * (N - i)
                    if not predicted_time:
                        predicted_time = time_remaining
                    print(f'\r{time_per_img:.2e} s/img | '
                          f'{i / N * 100:.1f}% | '
                          f'{time_remaining / 60:.1f} min remaining', end='', flush=True)

    def process(self, p_in):
        fn, ext = os.path.splitext(os.path.basename(p_in))
        if fn in self.images_cleaned:
            return 1
        if fn in self.images_discarded:
            return 0
        try:
            im = Image.open(p_in)
        except OSError as e:
            print(f'\n*** Error while opening {p_in}: {e}')
            return 0
        im_out = random_resize_or_discard(im, self.min_res)
        if im_out is not None:
            p_out = join(self.out_dir_clean, fn + '.png')  # Make sure to use .png!
            im_out.save(p_out)
            return 1
        else:
            p_out = join(self.out_dir_discard, os.path.basename(p_in))
            shutil.copy(p_in, p_out)
            return 0


def random_resize_or_discard(im, min_res: int):
    """Randomly resize image with `random_resize` and check if it should be discarded."""
    im_resized = random_resize(im, min_res)
    if im_resized is None:
        return None
    if should_discard(im_resized):
        return None
    return im_resized


def random_resize(im, min_res: int, max_scale=_DEFAULT_MAX_SCALE):
    """Scale longer side to `min_res`, but only if that scales by <= max_scale."""
    W, H = im.size
    D = min(W, H)
    scale_min = min_res / D
    # Image is too small to downscale by a factor smaller MAX_SCALE.
    if scale_min > max_scale:
        return None

    # Get a random scale for new size.
    scale = random.uniform(scale_min, max_scale)
    new_size = round(W * scale), round(H * scale)
    try:
        # Using LANCZOS!
        return im.resize(new_size, resample=PIL.Image.LANCZOS)
    except OSError as e:  # Happens for corrupted images
        print('*** Caught im.resize error', e)
        return None


def should_discard(im):
    """Return true iff the image is high in saturation or value, or not RGB."""
    # Modes found in train_0:
    # Counter({'RGB': 152326, 'L': 4149, 'CMYK': 66})
    if im.mode != 'RGB':
        return True
    im_rgb = np.array(im)
    im_hsv = skimage.color.rgb2hsv(im_rgb)
    mean_hsv = np.mean(im_hsv, axis=(0, 1))
    _, s, v = mean_hsv
    if s > 0.9:
        return True
    if v > 0.8:
        return True
    return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument('base_dir',
                   help='Directory of images, or directory of DIRS.')
    p.add_argument('dirs', nargs='*',
                   help='If given, must be subdirectories in BASE_DIR. Will be processed. '
                        'If not given, assume BASE_DIR is already a directory of images.')
    p.add_argument('--out_dir_clean', required=True)
    p.add_argument('--out_dir_discard', required=True)
    p.add_argument('--resolution', type=int, default=512,
                   help='Randomly rescale each image to be at least '
                        'RANDOM_SCALE long on the longer side.')

    flags = p.parse_args()

    # If --dirs not given, just assume `base_dir` is already the directory of images.
    if not flags.dirs:
        flags.dirs = [os.path.basename(flags.base_dir)]
        flags.base_dir = os.path.dirname(flags.base_dir)

    h = Helper(flags.out_dir_clean, flags.out_dir_discard, flags.resolution)
    for i, d in enumerate(flags.dirs):
        print(f'*** {d}: {i}/{len(flags.dirs)}')
        h.process_all_in(join(flags.base_dir, d))

    print('\n\nDONE')  # For cluster logs.


if __name__ == '__main__':
    main()
