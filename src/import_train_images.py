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

# TODO: maybe release.
# from dataloaders.cached_listdir_imgs import iter_images
from helpers.paths import IMG_EXTENSIONS

# task_array is not released. It's used by us to batch process on our servers. Feel free to replace with whatever you
# use. Make sure to set NUM_TASKS (number of concurrent processes) and set job_enumerate to a function that takes an
# iterable and only yield elements to be processed by the current process.
try:
    from task_array import NUM_TASKS, job_enumerate
except ImportError:
    NUM_TASKS = 1
    job_enumerate = enumerate

warnings.filterwarnings("ignore")


_NUM_PROCESSES = int(os.environ.get('NUM_PROCESS', 16))


get_fn = lambda p_: os.path.splitext(os.path.basename(p_))[0]


# TODO: copied from dataloaders
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
    def __init__(self, out_dir_clean, out_dir_discard, resolution: int,
                 crop4, crop16, random_scale):
        print(f'Creating {out_dir_clean}, {out_dir_discard}...')
        os.makedirs(out_dir_clean, exist_ok=True)
        os.makedirs(out_dir_discard, exist_ok=True)
        self.out_dir_clean = out_dir_clean
        self.out_dir_discard = out_dir_discard

        print('Getting processed images...')
        self.images_cleaned = set(map(get_fn, os.listdir(out_dir_clean)))
        self.images_discarded = set(map(get_fn, os.listdir(out_dir_discard)))
        print(f'Found {len(self.images_cleaned) + len(self.images_discarded)} processed images.')

        self.resolution = resolution

        self.crop4 = crop4
        self.crop16 = crop16
        self.random_scale = random_scale

    def process_all_in(self, input_dir, filter_imgs_list=None):
        images_dl = iter_images(input_dir)  # generator of paths

        # files this job should comperss
        files_of_job = [p for _, p in job_enumerate(images_dl)]
        # files that were compressed already by somebody (i.e. this job earlier)
        processed_already = self.images_cleaned | self.images_discarded
        # resulting files to be compressed
        files_of_job = [p for p in files_of_job if get_fn(p) not in processed_already]

        # TODO: Temporary to work around no discarding
        if filter_imgs_list:
            with open(filter_imgs_list, 'r') as f:
                ps_orig = f.read().split('\n')
            fns_to_use = set(map(get_fn, ps_orig))
            print('Filtering with', len(fns_to_use), 'filenames. Before:', len(files_of_job))
            files_of_job = [p for p in files_of_job if get_fn(p) in fns_to_use]
            print('Filtered, now', len(files_of_job))

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
        if predicted_time:
            print(f'Actual time: {(time.time() - start) / 60:.1f} // predicted {predicted_time / 60:.1f}')

    def process(self, p_in):
        fn, ext = os.path.splitext(os.path.basename(p_in))
        if fn in self.images_cleaned:
            return 1
        if fn in self.images_discarded:
            return 0
        try:
            im = Image.open(p_in)
            if self.crop4 or self.crop16:
                _crop_fn = _crop4 if self.crop4 else _crop16
                for i, im_crop in enumerate(_crop_fn(im)):
                    p_out = join(self.out_dir_clean, f'{fn}_{i}.png')
                    print(p_out)
                    im_crop.save(p_out)
                return 1
            if self.random_scale:
                im_out = random_resize(im, min_res=self.random_scale)
                if im_out is None:
                    return 0
                im_out.save(join(self.out_dir_clean, fn + '.png'))
                return 1
            # TODO: old code:
            raise NotImplementedError('Currently only random scale supported!')
            # im2 = resize_or_discard(im, self.resolution)
            # if im2 is not None:
            #     im2.save(join(self.out_dir_clean, fn + '.png'))
            #     return 1
            # else:
            #     p_out = join(self.out_dir_discard, os.path.basename(p_in))
            #     shutil.copy(p_in, p_out)
            #     return 0
        except OSError as e:
            print(e)
            return 0


def _crop16(im):
    for im_cropped in _crop4(im):
        yield from _crop4(im_cropped)


def _crop4(im):
    w, h = im.size
    #               (left, upper, right, lower)
    imgs = [im.crop((0, 0, w//2, h//2)),  # top left
            im.crop((0, h//2, w//2, h)),  # bottom left
            im.crop((w//2, 0, w, h//2)),  # top right
            im.crop((w//2, h//2, w, h)),  # bottom right
            ]

    assert sum(np.prod(i.size) for i in imgs) == np.prod(im.size)
    return imgs


# TODO: old code:
# def resize_or_discard(im, res: int, verbose=False):
#     im2 = resize(im, res)
#     if im2 is None:
#         return None
#     if should_discard(im2):
#         return None
#     return im2
#
#
# def rescale(im, scale):
#     W, H = im.size
#
#     W2 = round(W * scale)
#     H2 = round(H * scale)
#
#     try:
#         # TODO
#         return im.resize((W2, H2), resample=Image.LANCZOS)
#     except OSError as e:
#         print('*** im.resize error', e)
#         return None
#
#
# def resize(im, res):
#     """ scale longer side to `res`. """
#     W, H = im.size
#     D = max(W, H)
#     scaling_factor = float(res) / D
#     # image is already the target resolution, so no downscaling possible...
#     if scaling_factor > 0.95:
#         return None
#     W2 = round(W * scaling_factor)
#     H2 = round(H * scaling_factor)
#     try:
#         # TODO
#         return im.resize((W2, H2), resample=Image.BICUBIC)
#     except OSError as e:
#         print('*** im.resize error', e)
#         return None


MAX_SCALE = 0.8


def random_resize(im, min_res):
    """Scale longer side to `min_res`, but only if that scales by < MAX_SCALE."""
    W, H = im.size
    D = min(W, H)
    scale_min = min_res / D
    # Image is too small to downscale by a factor smaller MAX_SCALE.
    if scale_min > MAX_SCALE:
        return None

    # Get a random scale for new size.
    scale = random.uniform(scale_min, MAX_SCALE)
    new_size = round(W * scale), round(H * scale)
    try:
        # Using LANCZOS!
        return im.resize(new_size, resample=PIL.Image.LANCZOS)
    except OSError as e:  # Happens for corrupted images
        print('*** Caught im.resize error', e)
    return None


# TODO
def should_discard(im):
    # modes found in train_0:
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
    p.add_argument('base_dir', help='Directory of images.')
    p.add_argument('dirs', nargs='*', help='If given, must be subdiroectries in BASE_DIR. Will be processed.')
    p.add_argument('--out_dir_clean', required=True)
    p.add_argument('--out_dir_discard', required=True)
    # TODO
    # p.add_argument('--resolution', '-r', type=int, default=768)
    p.add_argument('--crop4', action='store_true', help='Crop images into 4 parts')
    p.add_argument('--crop16', action='store_true', help='Crop images into 16 parts')
    p.add_argument('--random_scale', type=int, help='If given, randomly rescale each image to be at least '
                                                    'RANDOM_SCALE long on the longer side.')
    p.add_argument('--filter_with_list', type=str, help='If given, only process images in the specified file.')

    flags = p.parse_args()
    if flags.filter_with_list and not os.path.isfile(flags.filter_with_list):
        raise ValueError('Must be file: {}'.format(flags.filter_with_list))

    h = Helper(flags.out_dir_clean, flags.out_dir_discard, None,  # TODO: flags.resolution,
               flags.crop4, flags.crop16, flags.random_scale)

    # If --dirs not given, just assume `base_dir` is already the directory of images.
    if not flags.dirs:
        flags.dirs = [os.path.basename(flags.base_dir)]
        flags.base_dir = os.path.dirname(flags.base_dir)

    for d in flags.dirs:
        h.process_all_in(join(flags.base_dir, d),
                         flags.filter_with_list)

    # For cluster logs.
    print('\n\nDONE')


if __name__ == '__main__':
    main()
