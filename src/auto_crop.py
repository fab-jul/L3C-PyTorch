"""
Helpers for cropping image depending on resolution.

TODO: replace recursive code with something more adaptive, right now we only do
    - no crops / 2x2 / 4x4 / 16x16 / etc.
    and the stitching code is complicated, but would be nice to have e.g. 3x3.
"""
import math
import os

import itertools

import torch

from blueprints.multiscale_blueprint import MultiscaleLoss
import functools
import operator


def prod(it):
    return functools.reduce(operator.mul, it, 1)


# Images with H * W > prod(_NEEDS_CROP_DIM) will be split into crops
# We set this empirically such that crops fit into our TITAN X (Pascal) with 12GB VRAM.
# You can set this from the console using AC_NEEDS_CROP_DIM, e.g.,
#
#   AC_NEEDS_CROP_DIM=2000,2000 python test.py ...
#
# But expect OOM errors for big values.
_NEEDS_CROP_DIM_DEFAULT = '2000,1500'
_NEEDS_CROP_DIM = os.environ.get('AC_NEEDS_CROP_DIM', _NEEDS_CROP_DIM_DEFAULT)
if _NEEDS_CROP_DIM != _NEEDS_CROP_DIM_DEFAULT:
    print('*** AC_NEEDS_CROP_DIM =', _NEEDS_CROP_DIM)
_NEEDS_CROP_DIM = prod(map(int, _NEEDS_CROP_DIM.split(',')))
print('*** AC_NEEDS_CROP_DIM =', _NEEDS_CROP_DIM)


def _assert_valid_image(i):
    if len(i.shape) != 4 or i.shape[1] != 3:
        raise ValueError(f'Expected BCHW image, got {i.shape}')


def needs_crop(img, needs_crop_dim=_NEEDS_CROP_DIM):
    _assert_valid_image(img)
    H, W = img.shape[-2:]
    return H * W > needs_crop_dim


def _crop16(im):
    for im_cropped in _crop4(im):
        yield from _crop4(im_cropped)


def iter_crops(img, needs_crop_dim=_NEEDS_CROP_DIM):
    _assert_valid_image(img)

    if not needs_crop(img, needs_crop_dim):
        yield img
        return
    for img_crop in _crop4(img):
        yield from iter_crops(img_crop, needs_crop_dim)


def _crop4(img):
    _assert_valid_image(img)
    H, W = img.shape[-2:]
    imgs = [img[..., :H//2, :W//2],  # Top left
            img[..., :H//2, W//2:],  # Top right
            img[..., H//2:, :W//2],  # Bottom left
            img[..., H//2:, W//2:]]  # Bottom right
    # Validate that we got all pixels
    assert sum(prod(img.shape[-2:]) for img in imgs) == \
           prod(img.shape[-2:])
    return imgs


def _get_crop_idx_mapping(side):
    """Helper method to get the order of crops.

    :param side: how many crops live on each side.

    Example. Say you have an image that gets devided into 16 crops, i.e., the image gets cut into 16 parts:

    [[ 0,  1,  2,  3],
     [ 4,  5,  6,  7],
     [ 8,  9, 10, 11],
     [12, 13, 14, 15]],

    However, due to our recursive cropping code, this results in crops that are ordered like this:

    index of crop:
     0  1  2  3  4 ...
    corresponds to part in image:
     0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15

    This method returns the inverse, going from the index of the crop back to the index in the image.
    """
    a = torch.arange(side * side).reshape(1, 1, side, side)
    a = torch.cat((a, a, a), dim=1)
    # Create mapping
    #   Index of crop in original image -> index of crop in the order it was extracted,
    # E.g. 2 -> 4  means it's the 2nd crop, but in the image, it's at position 4 (see above).
    crops = {i: crop[0, 0, ...].flatten().item()
             for i, crop in enumerate(iter_crops(a, 1))}
    return crops


def stitch(parts):
    side = int(math.sqrt(len(parts)))
    if side * side != len(parts):
        raise ValueError(f'Invalid number of parts {len(parts)}')

    rows = []

    # Sort by original position in image
    crops_idx_mapping = _get_crop_idx_mapping(side)
    parts_sorted = (
        part for _, part in sorted(
        enumerate(parts), key=lambda ip: crops_idx_mapping[ip[0]]))

    parts_itr = iter(parts_sorted)  # Turn into iterator so we can easily grab elements
    for _ in range(side):
        parts_row = itertools.islice(parts_itr, side)  # Get `side` number of parts
        row = torch.cat(list(parts_row), dim=3)  # cat on W dimension
        rows.append(row)

    assert next(parts_itr, None) is None, f'Iterator should be empty, got {len(rows)} rows'
    img = torch.cat(rows, dim=2)  # cat on H dimension

    # Validate.
    B, C, H_part, W_part = parts[0].shape
    expected_shape = (B, C, H_part * side, W_part * side)
    assert img.shape == expected_shape, f'{img.shape} != {expected_shape}'

    return img


class CropLossCombinator(object):
    """Used to combine the bpsp of different crops into one. Supports crops of varying dimensions."""
    def __init__(self):
        self._num_bits_total = 0.
        self._num_subpixels_total = 0

    def add(self, bpsp, num_subpixels_crop):
        bits = bpsp * num_subpixels_crop
        self._num_bits_total += bits
        self._num_subpixels_total += num_subpixels_crop

    def get_bpsp(self):
        assert self._num_subpixels_total > 0
        return self._num_bits_total / self._num_subpixels_total


def test_auto_crop():
    import torch
    import pytorch_ext as pe

    for H, W, num_crops_expected in [(10000, 6000, 64),
                                     (4928, 3264, 16),
                                     (2048, 2048, 4),
                                     (1024, 1024, 1),
                                     ]:
        img = (torch.rand(1, 3, H, W) * 255).round().long()
        print(img.shape)
        if num_crops_expected > 1:
            assert needs_crop(img)
            crops = list(iter_crops(img, 2048 * 1024))
            assert len(crops) == num_crops_expected
            pe.assert_equal(stitch(crops), img)
        else:
            pe.assert_equal(next(iter_crops(img, 2048 * 1024)), img)

