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


DEFAULT_CONFIG_DIR = 'configs'


class ConfigsRepo(object):
    def __init__(self, config_dir=DEFAULT_CONFIG_DIR):
        self.config_dir = config_dir

    def check_configs_available(self, *config_ps):
        for p in config_ps:
            assert self.config_dir in p, 'Expected {} to contain {}!'.format(p, self.config_dir)
            if not os.path.isfile(p):
                raise FileNotFoundError(p)
