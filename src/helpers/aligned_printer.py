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
import itertools


class AlignedPrinter(object):
    """ Print Rows nicely as a table. """
    def __init__(self):
        self.rows = []
        self.maxs = []

    def append(self, *row):
        self.rows.append(row)
        self.maxs = [max(max_cur, len(row_entry))
                     for max_cur, row_entry in
                     itertools.zip_longest(self.maxs, row, fillvalue=0)]

    def print(self):
        for row in self.rows:
            for width, row_entry in zip(self.maxs, row):
                print('{row_entry:{width}}'.format(row_entry=row_entry, width=width), end='   ')
            print()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.print()
