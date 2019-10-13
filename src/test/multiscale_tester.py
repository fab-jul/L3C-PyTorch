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
import ast
import os
import sys
import pytorch_ext as pe
from PIL import Image
import pickle
import shutil
import time
from contextlib import contextmanager

import fasteners
import numpy as np
import torch
from fjcommon import functools_ext as ft, config_parser, no_op
from fjcommon.assertions import assert_exc
from torchvision import transforms

from blueprints.multiscale_blueprint import MultiscaleBlueprint
from dataloaders.images_loader import IndexImagesDataset
from helpers import paths, saver, logdir_helpers
from helpers.config_checker import DEFAULT_CONFIG_DIR
from helpers.global_config import global_config
from collections import namedtuple

from test import cuda_timer
from test.image_saver import ImageSaver

# used for the Shared RGB basline
_DEFAULT_RECURSIVE_FOR_RGB = 3


_FILE_EXT = '.l3c'


_CLEAN_CACHE_PERIODICALLY = int(os.environ.get('CUDA_CLEAN_CACHE', '0')) == 1


class EncodeError(Exception):
    pass


class DecodeError(Exception):
    pass


class TestOutputCache(object):
    def __init__(self, test_log_dir):
        assert os.path.isdir(test_log_dir)
        self.test_log_dir = test_log_dir
        self.lock_file = os.path.join(test_log_dir, '.cache.lock')
        self.pickle_file = os.path.join(test_log_dir, 'cache.pkl')

    @contextmanager
    def _acquire_lock(self):
        with fasteners.InterProcessLock(self.lock_file):
            yield

    def __contains__(self, test_id):
        with self._acquire_lock():
            return test_id in self._read()

    def __setitem__(self, test_id, results):
        with self._acquire_lock():
            cache = self._read()
            cache[test_id] = results
            self._write(cache)

    def __getitem__(self, test_id):
        with self._acquire_lock():
            cache = self._read()
            return cache[test_id]

    def _read(self):
        if not os.path.isfile(self.pickle_file):
            return {}
        with open(self.pickle_file, 'rb') as f:
            return pickle.load(f)

    def _write(self, cache):
        with open(self.pickle_file, 'wb') as f:
            return pickle.dump(cache, f)


# Uniquely identifies a test run of some experiment
# dataset_id comes from Testset.id, which is 'FOLDERNAME_NUMIMGS'
TestID = namedtuple('TestID', ['dataset_id', 'restore_itr'])


class TestResult(object):
    def __init__(self, metric_name):
        self.metric_name = metric_name
        self.per_img_results = {}

    def __setitem__(self, filename, result):
        self.per_img_results[filename] = result.item()  # unpack Torch Tensors

    def mean(self):
        return np.mean(list(self.per_img_results.values()))



def _parse_recursive_flag(recursive, config_ms):
    if not config_ms.rgb_bicubic_baseline:
        return 0
    if recursive == 'auto':
        if config_ms.rgb_bicubic_baseline and config_ms.num_scales == 1:  # RGB - shared
            return _DEFAULT_RECURSIVE_FOR_RGB
    try:
        return int(recursive)
    except ValueError:
        return 0


def _clean_cuda_cache(i):
    if i % 25 == 0 and torch.cuda.is_available():
        print()
        print('{:,} {:,} {:,} {:,}'.format(
                torch.cuda.max_memory_allocated(),
                torch.cuda.memory_allocated(),
                torch.cuda.max_memory_cached(),
                torch.cuda.memory_cached()))
        torch.cuda.empty_cache()


def check_correct_torchac_backend_available():
    try:
        from torchac import torchac
    except ImportError as e:
        raise ValueError(f'Caught {e}. --write_to_files requires torchac. See README.')
    if pe.CUDA_AVAILABLE and not torchac.CUDA_SUPPORTED:
        raise ValueError('Found CUDA but torachac_backend_gpu not compiled. '
                         'Either compile it or use CUDA_VISIBLE_DEVICES="". See also README')
    if not pe.CUDA_AVAILABLE and not torchac.CPU_SUPPORTED:
        raise ValueError('No CUDA found but torchac_backend_cpu not compiled. '
                         'Compile it or set CUDA_VISIBLE_DEVICES appropriately. See also README.')


class MultiscaleTester(object):
    def __init__(self, log_date, flags, restore_itr, l3c=False):
        """
        :param flags:
            log_dir
            img
            filter_filenames
            max_imgs_per_folder
            # out_dir
            crop
            recursive
            sample
            write_to_files
            compare_theory
            time_report
            overwrite_cache
        """
        self.flags = flags

        test_log_dir_root = self.flags.log_dir.rstrip(os.path.sep) + '_test'
        global_config.reset()

        config_ps, experiment_dir = MultiscaleTester.get_configs_experiment_dir('ms', self.flags.log_dir, log_date)
        self.log_date = logdir_helpers.log_date_from_log_dir(experiment_dir)
        (self.config_ms, _), _ = ft.unzip(map(config_parser.parse, config_ps))
        global_config.update_config(self.config_ms)

        self.recursive = _parse_recursive_flag(self.flags.recursive, config_ms=self.config_ms)
        if self.flags.write_to_files and self.recursive:
            raise NotImplementedError('--write_to_file not implemented for --recursive')

        if self.recursive:
            print(f'--recursive={self.recursive}')

        blueprint = MultiscaleBlueprint(self.config_ms)
        blueprint.set_eval()
        self.blueprint = blueprint

        self.restorer = saver.Restorer(paths.get_ckpts_dir(experiment_dir))
        self.restore_itr, ckpt_p = self.restorer.get_ckpt_for_itr(restore_itr)
        self.restorer.restore({'net': self.blueprint.net}, ckpt_p, strict=True)

        # test_log_dir/0311_1057 cr oi_012
        self.test_log_dir = os.path.join(
                test_log_dir_root, os.path.basename(experiment_dir))
        if self.flags.reset_entire_cache and os.path.isdir(self.test_log_dir):
            print(f'Removing test_log_dir={self.test_log_dir}...')
            time.sleep(1)
            shutil.rmtree(self.test_log_dir)
        os.makedirs(self.test_log_dir, exist_ok=True)
        self.test_output_cache = TestOutputCache(self.test_log_dir)

        self.times = cuda_timer.StackTimeLogger() if self.flags.write_to_files else None

        # Import only if needed, as it imports torchac
        if self.flags.write_to_files:
            check_correct_torchac_backend_available()
            from bitcoding.bitcoding import Bitcoding
            self.bc = Bitcoding(self.blueprint, times=self.times, compare_with_theory=self.flags.compare_theory)
        elif l3c:  # Called from l3c.py
            from bitcoding.bitcoding import Bitcoding
            self.bc = Bitcoding(self.blueprint, times=no_op.NoOp)

    def _padding_fac(self):
        if self.recursive:
            return 2 ** (self.recursive + 1)
        return 2 ** self.config_ms.num_scales

    @staticmethod
    def get_configs_experiment_dir(prefix, log_dir, log_date):
        experiment_dir = paths.get_experiment_dir(log_dir, log_date)
        log_dir_comps = logdir_helpers.parse_log_dir(
                experiment_dir, DEFAULT_CONFIG_DIR, [prefix, 'dl'], append_ext='.cf')
        config_ps = log_dir_comps.config_paths
        global_config.add_from_flag(log_dir_comps.postfix)
        return config_ps, experiment_dir

    def test_all(self, testsets):
        results = [self.test(testset) for testset in testsets]
        if self.flags.write_to_files:  # no results generated
            return [None]
        return [(testset, self.log_date, self.restore_itr, f'{result.metric_name}={result.mean()}')
                for testset, result in zip(testsets, results)]

    def test(self, testset):
        test_id = TestID(testset.id, self.restore_itr)
        return_cache = (not self.flags.overwrite_cache and
                        not self.flags.write_to_files and
                        test_id in self.test_output_cache)
        if return_cache:
            print(f'*** Found cached: {test_id}')
            return self.test_output_cache[test_id]

        print('Testing {}'.format(testset))
        ds = self.get_test_dataset(testset)
        with torch.no_grad():
            result = self._test(ds)
            if not result:  # because self.flags.write_to_files
                return None

        self.test_output_cache[test_id] = result
        return result

    def get_test_dataset(self, testset):
        to_tensor_transform = [IndexImagesDataset.to_tensor_uint8_transform()]
        if self.flags.crop:
            print('*** WARN: Cropping to {}'.format(self.flags.crop))
            to_tensor_transform.insert(0, transforms.CenterCrop(self.flags.crop))
        return IndexImagesDataset(
                testset,
                to_tensor_transform=transforms.Compose(to_tensor_transform))

    def _test(self, ds):
        # If we write to file, we do not store any TestResult
        test_result = (TestResult(metric_name='bpsp recursive' if self.recursive else 'bpsp')
                       if not self.flags.write_to_files
                       else None)

        # If we sample, we store the result with a ImageSaver
        if self.flags.sample:
            image_saver = ImageSaver(os.path.join(self.flags.sample, self.log_date))
            print('Will store samples in {}.'.format(image_saver.out_dir))
        else:
            image_saver = None

        log = ''
        one_line_output = not self.flags.sample

        for i, img in enumerate(ds):
            filename = os.path.splitext(os.path.basename(ds.files[img['idx']]))[0]

            if _CLEAN_CACHE_PERIODICALLY:
                _clean_cuda_cache(i)

            # We have to pad images not divisible by (2 ** num_scales), because we downsample num_scales-times.
            # To get the correct bpsp, we have to use, num_subpixels_before_pad,
            #   see `get_loss` in multiscale_blueprint.py
            num_subpixels_before_pad = np.prod(img["raw"].shape)
            img_batch, img_batch_raw = self.blueprint.unpack_batch_pad(img, fac=self._padding_fac())

            if self.flags.write_to_files:
                print('***', filename)
                with self.times.skip(i == 0):
                    out_dir = self.flags.write_to_files
                    os.makedirs(out_dir, exist_ok=True)
                    out_p = os.path.join(out_dir, filename + _FILE_EXT)
                    info = self._write_to_file(img_batch_raw, out_p)
                    print(info)  # prints bpsp
                    print('*' * 80)
                    continue

            out = self.blueprint.forward(img_batch, self.recursive)
            loss_out = self.blueprint.get_loss(out, num_subpixels_before_pad=num_subpixels_before_pad)

            if self.recursive:
                test_result[filename] = sum(loss_out.recursive_bpsps)
            else:
                test_result[filename] = sum(loss_out.nonrecursive_bpsps)

            if self.flags.sample:
                self._sample(loss_out.nonrecursive_bpsps, img_batch, image_saver, '{}_{}'.format(i, filename))

            log = f'{self.log_date}: {filename} ({i: 10d}): mean {test_result.metric_name}={test_result.mean()}'
            _print(log, one_line_output)
        _print(log, one_line_output, final=True)
        return test_result

    def _write_to_file(self, img, out_p):
        """
        :param img: 1CHW, long
        :param out_p: string
        :return: info string
        """
        if os.path.isfile(out_p):
            os.remove(out_p)

        with self.times.run('=== bc.encode'):
            info = self.bc.encode(img, pout=out_p)

        with self.times.run('=== bc.decode'):
            img_o = self.bc.decode(pin=out_p)

        pe.assert_equal(img, img_o)

        print('\n'.join(self.times.get_last_strs()))
        if self.flags.time_report:
            with open(self.flags.time_report, 'w') as f:
                f.write('Average times:\n')
                f.write('\n'.join(self.times.get_mean_strs()))

        return info

    def encode(self, img_p, pout, overwrite=False, use_patches=False, patch_size = 32):
        pout_dir = os.path.dirname(os.path.abspath(pout))
        assert_exc(os.path.isdir(pout_dir), f'pout directory ({pout_dir}) does not exists!', EncodeError)
        if overwrite and os.path.isfile(pout):
            print(f'Removing {pout}...')
            os.remove(pout)
        assert_exc(not os.path.isfile(pout), f'{pout} exists. Consider --overwrite', EncodeError)

        if not use_patches:
            img = self._read_img(img_p)
            img = img.to(pe.DEVICE)

            self.bc.encode(img, pout=pout)
        else:
            self.encode_patches(img_p, pout, patch_size)
        print('---\nSaved:', pout)

    def encode_patches(self, img_p, pout, patch_size):
        img = self._read_img(img_p)
        img = img.to(pe.DEVICE)

        width = img.shape[2] // patch_size
        height = img.shape[3] // patch_size

        file_names = []
        for i in range(width):
            for j in range(height):
                patch = img[:, :, i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size]
                file_names.append("/tmp/patch_{}_{}.l3c".format(i, j))
                self.bc.encode(patch, file_names[-1])

        content = bytes()
        lens = []
        for file_name in file_names:
            with open(file_name, "rb") as fp:
                curr_content = fp.read()
                lens.append(len(curr_content))
                content += curr_content

        meta = [width, height] + lens
        arr = bytearray()
        arr.extend((str(meta)+"\n").encode("latin-1"))
        arr.extend(content)
        with open(pout, "wb+") as fp:
            fp.write(arr)

    def decode(self, pin, png_out_p, use_patches=False, patch_size=32):
        """
        Decode L3C-encoded file at `pin` to a PNG at `png_out_p`.
        """
        pout_dir = os.path.dirname(os.path.abspath(png_out_p))
        assert_exc(os.path.isdir(pout_dir), f'png_out_p directory ({pout_dir}) does not exists!', DecodeError)
        assert_exc(png_out_p.endswith('.png'), f'png_out_p must end in .png, got {png_out_p}', DecodeError)
        if not use_patches:
            decoded = self.bc.decode(pin)

            self._write_img(decoded, png_out_p)
        else:
            self.decode_patches(pin, png_out_p, patch_size)
        print(f'---\nDecoded: {png_out_p}')

    def decode_patches(self, pin, png_out_p, patch_size):
        with open(pin, "rb") as fp:
            content = fp.read()
            fp.close()
        linebreak = "\n".encode("latin-1")
        idx = content.index(linebreak)
        meta = content[:idx].decode("ascii")
        meta = ast.literal_eval(meta)
        width, height = meta[:2]
        lens = meta[2:]
        acc_lens = [0]
        for i in range(0, len(lens)):
            acc_lens.append(acc_lens[-1] + lens[i])

        # only decode actual content!
        content = content[idx+1:]
        # prepare output image tensor with full resolution
        img = torch.zeros((1, 3, width*patch_size, height*patch_size))
        for i in range(width):
            for j in range(height):
                idx = i * height + j
                curr = content[acc_lens[idx]:acc_lens[idx+1]]
                
                # use temporary file to use established bitcoding procedure
                tmp_path = "/tmp/patch_tmp.l3c"
                with open(tmp_path, "wb+") as fp:
                    fp.write(curr)

                img[:, :, i*patch_size:(i+1)*patch_size, 
                    j*patch_size:(j+1)*patch_size] = self.bc.decode(tmp_path)
        
        self._write_img(img, png_out_p)

    def _read_img(self, img_p):
        img = np.array(Image.open(img_p)).transpose(2, 0, 1)  # Turn into CHW
        C, H, W = img.shape
        # check number of channels
        if C == 4:
            print('*** WARN: Will discard 4th (alpha) channel.')
            img = img[:3, ...]
        elif C != 3:
            raise EncodeError(f'Image has {C} channels, expected 3 or 4.')
        # Convert to 1CHW torch tensor
        img = torch.from_numpy(img).unsqueeze(0).long()
        # Check padding
        padding = self._padding_fac()
        if H % padding != 0 or W % padding != 0:
            print(f'*** WARN: image shape ({H}X{W}) not divisible by {padding}. Will pad...')
            img = MultiscaleBlueprint.pad(img, fac=padding)
        return img

    @staticmethod
    def _write_img(decoded, png_out_p):
        """
        :param decoded: 1CHW tensor
        :param png_out_p: str
        """
        # TODO: should undo padding
        assert decoded.shape[0] == 1 and decoded.shape[1] == 3, decoded.shape
        img = pe.tensor_to_np(decoded.squeeze(0))  # CHW
        img = img.transpose(1, 2, 0).astype(np.uint8)  # Make HW3
        img = Image.fromarray(img)
        img.save(png_out_p)

    def _sample(self, bpsps, img_batch, image_saver, save_prefix):
        # Make sure folder does not already contain samples for this file.
        if image_saver.file_starting_with_exists(save_prefix):
            raise FileExistsError('Previous sample outputs found in {}. Please remove.'.format(
                    image_saver.out_dir))
        # Store ground truth for comparison
        image_saver.save_img(img_batch, '{}_{:.3f}_gt.png'.format(save_prefix, sum(bpsps)))
        for style, sample_scales in (('rgb', []),               # Sample RGB scale (final scale)
                                     ('rgb+bn0', [0]),          # Sample RGB + z^(1)
                                     ('rgb+bn0+bn1', [0, 1])):  # Sample RGB + z^(1) + z^(2)
            sampled = self.blueprint.sample_forward(img_batch, sample_scales)
            bpsp_sample = sum(bpsps[len(sample_scales) + 1:])
            image_saver.save_img(sampled, '{}_{}_{:.3f}.png'.format(save_prefix, style, bpsp_sample))



def _print(s, oneline, final=False):
    if oneline:
        if not final:
            print('\r' + s, end='')
        else:
            print('\n' + s)
    else:
        print(s)


