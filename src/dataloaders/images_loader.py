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
import glob
import os
import pickle

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset

from helpers.paths import has_image_ext
from helpers.testset import Testset


class NoImagesFoundException(Exception):
    def __init__(self, p):
        self.p = p


class IndexImagesDataset(Dataset):
    """
    A Dataset class for images, that also returns the index of the image in the dataset.
    """
    @staticmethod
    def to_tensor_uint8_transform():
        """ Convert PIL to uint8 tensor, CHW. """
        return to_tensor_not_normalized

    @staticmethod
    def to_grb(t):
        assert t.shape[0] == 3
        with torch.no_grad():
            return torch.stack((t[1, ...], t[0, ...], t[2, ...]), dim=0)

    @staticmethod
    def to_float_tensor_transform():
        return transforms.Lambda(lambda img: img.float().div(255.))

    def copy(self):
        return IndexImagesDataset(
                self.images_spec, self.to_tensor_transform, self.cache_p, self.min_size)

    def __init__(self, images, to_tensor_transform, fixed_first=None):
        """
        :param images: Instance of Testset or ImagesCached
        :param to_tensor_transform: A function that takes a PIL RGB image and returns a torch.Tensor
        """
        self.to_tensor_transform = to_tensor_transform

        if isinstance(images, ImagesCached):
            self.files = images.get_images_sorted_cached()
            self.id = '{}_{}_{}'.format(images.images_spec, images.min_size, len(self.files))
        elif isinstance(images, Testset):
            self.files = images.ps
            self.id = images.id
        else:
            raise ValueError('Expected ImagesCached or Testset, got images={}'.format(images))

        if fixed_first:
            assert os.path.isfile(fixed_first)
            self.files = [fixed_first] + self.files

        if len(self.files) == 0:
            raise NoImagesFoundException(images.search_path())

    def __len__(self):
        return len(self.files)

    def __str__(self):
        return 'IndexImagesDataset({} images, id={})'.format(len(self.files), self.id)

    def __getitem__(self, idx):
        path = self.files[idx]
        with open(path, 'rb') as f:
            pil = Image.open(f).convert('RGB')
            raw = self.to_tensor_transform(pil)  # 3HW uint8s
        return {'idx': idx,
                'raw': raw}


def to_tensor_not_normalized(pic):
    """ copied from PyTorch functional.to_tensor, removed final .float().div(255.) """
    if isinstance(pic, np.ndarray):
        # handle numpy array
        img = torch.from_numpy(pic.transpose((2, 0, 1)))
        return img

    # handle PIL Image
    if pic.mode == 'I':
        img = torch.from_numpy(np.array(pic, np.int32, copy=False))
    elif pic.mode == 'I;16':
        img = torch.from_numpy(np.array(pic, np.int16, copy=False))
    elif pic.mode == 'F':
        img = torch.from_numpy(np.array(pic, np.float32, copy=False))
    elif pic.mode == '1':
        img = 255 * torch.from_numpy(np.array(pic, np.uint8, copy=False))
    else:
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
    # PIL image mode: L, P, I, F, RGB, YCbCr, RGBA, CMYK
    if pic.mode == 'YCbCr':
        nchannel = 3
    elif pic.mode == 'I;16':
        nchannel = 1
    else:
        nchannel = len(pic.mode)
    img = img.view(pic.size[1], pic.size[0], nchannel)
    # put it from HWC to CHW format
    # yikes, this transpose takes 80% of the loading time/CPU
    img = img.transpose(0, 1).transpose(0, 2).contiguous()
    return img


class ImagesCached(object):
    """ Caches contents of folders, for slow filesystems. """
    def __init__(self, images_spec, cache_p=None, min_size=None):
        """
        :param images_spec: str, interpreted as
            - a glob, if it contains a *
            - a single image, if it ends in one of IMG_EXTENSIONS
            - a directory otherwise
        :param cache_p: path to a cache or None. If given, check there when `get_images_sorted_cached` is called
        :param min_size: if given, make sure to only return/cache images of the given size
        :return:
        """
        self.images_spec = os.path.expanduser(images_spec)
        self.cache_p = os.path.expanduser(cache_p)
        self.min_size = min_size
        self.cache = ImagesCached._get_cache(self.cache_p)

    def __str__(self):
        return f'ImagesCached(images_spec={self.images_spec})'

    def search_path(self):
        return self.images_spec

    def get_images_sorted_cached(self):
        key = self.images_spec, self.min_size
        if key in self.cache:
            return self.cache[key]
        if self.cache_p:
            print(f'WARN: Given cache_p={self.cache_p}, but key not found:\n{key}')
            available_keys = sorted(self.cache.keys(), key=lambda img_size: img_size[0])
            print('Found:\n' + '\n'.join(map(str, available_keys)))
        return sorted(self._iter_imgs_unordered_filter_size())

    def update(self, force, verbose):
        """
        Writes/updates to cache_p
        :param force: overwrites
        """
        if not force and (self.images_spec, self.min_size) in self.cache:
            print('Cache already contains {}'.format(self.images_spec))
            return
        print('Updating cache for {}...'.format(self.images_spec))
        images = sorted(self._iter_imgs_unordered_filter_size(verbose))
        if len(images) == 0:
            print('No images found...')
            return
        print('Found {} images...'.format(len(images)))
        self.cache[(self.images_spec, self.min_size)] = images
        print('Writing cache...')
        with open(self.cache_p, 'wb') as f:
            pickle.dump(self.cache, f)

    @staticmethod
    def print_all(cache_p):
        """
        Print all cache entries to console
        :param cache_p: Path of the cache to print
        """
        cache = ImagesCached._get_cache(cache_p)
        for key in list(cache.keys()):
            if len(cache[key]) == 0:
                del cache[key]
        for (p, min_size), imgs in cache.items():
            min_size_str = ' (>={})'.format(min_size) if min_size else ''
            print('{}{}: {} images'.format(p, min_size_str, len(imgs)))

    def _iter_imgs_unordered_filter_size(self, verbose=False):
        for p in self._iter_imgs_unordered(self.images_spec):
            if self.min_size:
                img = Image.open(p)
                img_min_dim = min(img.size)
                if img_min_dim < self.min_size:
                    print('Skipping {} ({})...'.format(p, img.size))
                    continue
            if verbose:
                print(p)
            yield p

    @staticmethod
    def _get_cache(cache_p):
        if not cache_p:
            return {}
        if not os.path.isfile(cache_p):
            print(f'cache_p={cache_p} does not exist.')
            return {}
        with open(cache_p, 'rb') as f:
            return pickle.load(f)

    @staticmethod
    def _iter_imgs_unordered(images_spec):
        if '*' in images_spec:
            matches = glob.glob(images_spec)
            _, ext = os.path.splitext(images_spec)
            if not ext:
                matches = (p for p in matches if has_image_ext(p))
            elif not has_image_ext(images_spec):
                raise ValueError('Unrecognized extension {} in glob ({})'.format(ext, images_spec))
            if not matches:
                raise ValueError('No matches for glob {}'.format(images_spec))
            yield from matches
            return

        if has_image_ext(images_spec):
            yield from [images_spec]
            return

        # At this point, images_spec should be a path to a directory
        if not os.path.isdir(images_spec):
            raise NotADirectoryError(images_spec)

        print('Recursively traversing {}...'.format(images_spec))
        for root, _, fnames in os.walk(images_spec, followlinks=True):
            for fname in fnames:
                if has_image_ext(fname):
                    p = os.path.join(root, fname)
                    if os.path.getsize(p) == 0:
                        print('WARN / 0 bytes /', p)
                        continue
                    yield p

def main():
    # See README
    p = argparse.ArgumentParser()
    mode_parsers = p.add_subparsers(dest='mode')
    show_p = mode_parsers.add_parser('show')
    show_p.add_argument('cache_p')
    update_p = mode_parsers.add_parser('update')
    update_p.add_argument('images_spec')
    update_p.add_argument('cache_p')
    update_p.add_argument('--min_size', type=int)
    update_p.add_argument('--force', '-f', action='store_true')
    update_p.add_argument('--verbose', '-v', action='store_true', help='Print found paths. Might be slow!')

    flags = p.parse_args()
    if flags.mode == 'show':
        ImagesCached.print_all(flags.cache_p)
    elif flags.mode == 'update':
        ImagesCached(flags.images_spec, flags.cache_p, flags.min_size).update(flags.force, flags.verbose)
    else:
        p.print_usage()


# Resize bicubic


def resize_bicubic_batch(t, fac):
    assert len(t.shape) == 4
    N = t.shape[0]
    return torch.stack([resize_bicubic(t[n, ...], fac) for n in range(N)], dim=0)


def resize_bicubic(t, fac):
    img = _tensor_to_image(t)  # to PIL
    h, w = img.size
    img = img.resize((int(h * fac), int(w * fac)), Image.BICUBIC)
    t = to_tensor_not_normalized(img)  # back to 3HW uint8 tensor
    return t


def _tensor_to_image(t):
    assert t.shape[0] == 3, t.shape
    return Image.fromarray(t.permute(1, 2, 0).detach().cpu().numpy())


if __name__ == '__main__':
    main()

