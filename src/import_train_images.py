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
import random
from os.path import join
import PIL
from PIL import Image
import skimage.color
import numpy as np
import argparse
import warnings
# task_array is not released. It's used by us to batch process on our servers. Feel free to replace with whatever you
# use. Make sure to set NUM_TASKS (number of concurrent processes) and set job_enumerate to a function that takes an
# iterable and only yield elements to be processed by the current process.
try:
    from task_array import NUM_TASKS, job_enumerate
except ImportError:
    NUM_TASKS = 1
    job_enumerate = enumerate

warnings.filterwarnings("ignore")


QUALITY = 95
MAX_SCALE = 0.95


def main():
    p = argparse.ArgumentParser()
    p.add_argument('base_dir')
    p.add_argument('dirs', nargs='+')
    p.add_argument('--resolution', '-r', type=str, default='768', help='can be randX_Y')
    p.add_argument('--no_clean', '-nc', action='store_true')
    p.add_argument('--out_dir', '-o')
    p.add_argument('--seed')
    flags = p.parse_args()

    if flags.seed:
        print('Seeding {}'.format(flags.seed))
        random.seed(flags.seed)

    for d in flags.dirs:
        process(join(flags.base_dir, d), flags.out_dir, flags.resolution, not flags.no_clean)


def get_res(res) -> int:
    try:
        return int(res)
    except ValueError:
        pass
    if not res.startswith('rand'):
        raise ValueError('Expected res to be either int or `randX_Y`')
    X, Y = map(int, res.replace('rand', '').split('_'))
    return random.randint(X, Y)


def process(input_dir, out_dir, res: str, should_clean):
    inp_base, inp_name = os.path.split(input_dir.rstrip(os.path.sep))
    if out_dir is None:
        out_dir = inp_base
    output_dir = os.path.join(out_dir, inp_name + '_' + res + ('_clean' if should_clean else '_all'))
    os.makedirs(output_dir, exist_ok=True)
    # if should_clean, we do not discard because of HSV, but still might because of size!
    discard_dir = os.path.join(out_dir, inp_name + '_' + res + ('_discard' if should_clean else '_discard_size'))
    os.makedirs(discard_dir, exist_ok=True)

    images_dl = os.listdir(input_dir)
    N = len(images_dl) // NUM_TASKS
    images_done = set(os.listdir(output_dir)) | set(os.listdir(discard_dir))

    clean = 0
    discarded = 0
    skipped = 0
    for i, imfile in job_enumerate(images_dl):
        if imfile in images_done:
            skipped += 1
            continue
        im = Image.open(join(input_dir, imfile))
        res = get_res(res)
        im2 = resize_or_discard(im, res, should_clean=should_clean)
        if im2 is not None:
            fn, ext = os.path.splitext(imfile)
            im2.save(join(output_dir, fn + '.jpg'), quality=QUALITY)
            clean += 1
        else:
            im.save(join(discard_dir, imfile))
            discarded += 1
        print('\r{} -> {} // Skipped {}/{}, Resized {}/{}, Discarded {}/{}'.format(
                os.path.basename(input_dir), os.path.basename(output_dir),
                skipped, N, clean, N, discarded, N), end='')
    # Done
    print('\n{} -> {} // Skipped {}/{}, Resized {}/{}, Discarded {}/{} // Completed'.format(
            input_dir, output_dir,
            skipped, N, clean, N, discarded, N))


def resize_or_discard(im, res: int, verbose=False, should_clean=True):
    im2 = resize(im, res, verbose)
    if im2 is None:
        return None
    if should_clean and should_discard(im2, verbose):
        return None
    return im2


def resize(im, res, verbose=False, max_scale=MAX_SCALE):
    W, H = im.size
    D = max(W, H)
    s = float(res) / D
    if max_scale and s > max_scale:
        if verbose:
            print('Too big: {}'.format((W, H)))
        return None
    W2 = round(W * s)
    H2 = round(H * s)
    try:
        return im.resize((W2, H2), resample=PIL.Image.BICUBIC)
    except OSError as e:
        print(e)
        return None


def should_discard(im, verbose=False):
    im_rgb = np.array(im)
    if im_rgb.ndim != 3 or im_rgb.shape[2] != 3:
        if verbose:
            print('Invalid shape: {}'.format(im_rgb.shape))
        return True
    im_hsv = skimage.color.rgb2hsv(im_rgb)
    mean_hsv = np.mean(im_hsv, axis=(0, 1))
    h, s, v = mean_hsv
    if s > 0.9:
        if verbose:
            print('Invalid s: {}'.format(s))
        return True
    if v > 0.8:
        if verbose:
            print('Invalid v: {}'.format(v))
        return True
    return False


def get_hsv(im):
    im_rgb = np.array(im)
    im_hsv = skimage.color.rgb2hsv(im_rgb)
    mean_hsv = np.mean(im_hsv, axis=(0, 1))
    h, s, v = mean_hsv
    return h, s, v


if __name__ == '__main__':
    main()
