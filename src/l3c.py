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
import torch.backends.cudnn
torch.backends.cudnn.benchmark = True

import argparse

from test.multiscale_tester import MultiscaleTester, EncodeError, DecodeError


class _FakeFlags(object):
    def __init__(self, flags):
        self.flags = flags

    def __getattr__(self, item):
        try:
            return self.flags.__dict__[item]
        except KeyError:
            return None


def main():
    p = argparse.ArgumentParser(description='Encoder/Decoder for L3C')

    p.add_argument('log_dir', help='Directory of experiments.')
    p.add_argument('log_date', help='A log_date, such as 0104_1345.')

    p.add_argument('--restore_itr', '-i', default=-1, type=int,
                   help='Which iteration to restore. -1 means latest iteration. Default: -1')

    mode = p.add_subparsers(title='mode', dest='mode')

    enc = mode.add_parser('enc', help='Encode image. enc IMG_P OUT_P [--overwrite | -f]\n'
                                      '    IMG_P: Path to an Image, readable by PIL.\n'
                                      '    OUT_P: Path to where to save the bitstream.\n'
                                      '    OVERWRITE: If given, overwrite OUT_P.'
                                      'Example:\n'
                                      '    python l3c.py LOG_DIR LOG_DATE enc some/img.jpg out/img.l3c')
    dec = mode.add_parser('dec', help='Decode image. dec IMG_P OUT_P_PNG\n'
                                      '    IMG_P: Path to an L3C-encoded Image, readable by PIL.\n'
                                      '    OUT_P_PNG: Path to where to save the decode image to, as a PNG.\n'
                                      'Example:\n'
                                      '    python l3c.py LOG_DIR LOG_DATE dec out/img.l3c decoded.png')

    enc.add_argument('img_p')
    enc.add_argument('out_p')
    enc.add_argument('--overwrite', '-f', action='store_true')

    dec.add_argument('img_p')
    dec.add_argument('out_p_png')

    flags = p.parse_args()

    print('Testing {} at {} ---'.format(flags.log_date, flags.restore_itr))
    tester = MultiscaleTester(flags.log_date, _FakeFlags(flags), flags.restore_itr, l3c=True)

    if flags.mode == 'enc':
        try:
            tester.encode(flags.img_p, flags.out_p, flags.overwrite)
        except EncodeError as e:
            print('*** EncodeError:', e)
    else:
        try:
            tester.decode(flags.img_p, flags.out_p_png)
        except DecodeError as e:
            print('*** DecodeError:', e)


if __name__ == '__main__':
    main()
