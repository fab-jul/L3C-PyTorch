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


Support for global config parameters shared over the whole program. The goal is to easily add parameters into some
nested module without passing it all the way through, for fast prototyping.
Also supports updating Configs returned by config_parser.py.


Usage

```
from global_config import global_config

parser = argparse.ArgumentParser()
parser.add_argument('-p', action='append', nargs=1)
...
flags = parser.parse_args()
...
global_config.add_from_flag(flags.p)

# in some module

from global_config import global_config

# check if loss is set, if not, use the default of 'mse'
loss_fct = global_config.get('loss', default_value='mse')
if loss_fct == 'mse':
    loss = MSE
elif loss_fct == 'ce':
    loss = CrossEntropy

# start the training
python train.py -p loss=ce
```


"""
from contextlib import contextmanager


class _GlobalConfig(object):
    def __init__(self, default_unset_value=False, type_classes=None):
        """
        :param default_unset_value: Value to use if global_config['key'] is called on a key that is not set
        :param type_classes: supported type classes that values are tried to convert to, see `_eval_value`
        """
        if type_classes is None:
            type_classes = [int, float]
        self.type_classes = type_classes
        self.default_unset_value = default_unset_value
        self._values = {}
        self._used_params = set()

    def add_from_flag(self, param_flag):
        """ Add from a list containing key=value, or key (eqivalent to key=True). """
        if param_flag is None:
            return
        for param_spec in param_flag:
            if isinstance(param_spec, list):
                assert len(param_spec) == 1
                param_spec = param_spec[0]
            self.add_param_from_spec(param_spec)

    def update_config(self, config):
        """ Update a fjcommon._Config returned by fjcommon.config_parser """
        for k, v in config.all_params_and_values():
            if k in self:
                print('Updating config.{} = {}'.format(k, self[k]))
                config.set_attr(k, self[k])
                self.declare_used(k)

    def add_param_from_spec(self, spec):
        if '=' not in spec:
            spec = '{}=True'.format(spec)
        key, value = spec.split('=')
        key = key.strip()
        value = self._eval_value(value.strip())
        self[key] = value

    def _eval_value(self, value):
        # try if `value` is True, False, or None, and return the actual instance
        if value in ('True', 'False', 'None'):
            return {'True':  True,
                    'False': False,
                    'None':  None}[value]

        # try casting to classes in type_classes
        for type_cls in self.type_classes:
            try:
                return type_cls(value)
            except ValueError:
                continue

        # finally, just interpret as string type
        if ' ' in value:
            raise ValueError('values are not allowed to contain spaces! {}'.format(value))
        if '/' in value or '~' in value:
            raise ValueError('values are not allowed to contain "/" or "~"! {}'.format(value))
        return value

    def __setitem__(self, key, value):
        self._values[key] = value

    def __getitem__(self, key):
        return self.get(key, self.default_unset_value)

    def __contains__(self, item):
        return item in self._values

    def get(self, key, default_value, incompatible=None):
        """
        Check if `key` is set
        :param default_value: value to return if `key` is not set
        :param incompatible: list of keys which are not allowed to bet set if `key` is set
        :return: values[key] if key is set, `default_value` otherwise
        """
        if incompatible and key in self._values:
            self._ensure_not_specified(key, incompatible)
        self._used_params.add(key)
        return self._values.get(key, default_value)

    def declare_used(self, *keys):
        """ Hack to mark all keys in `keys` as used, even if global_config[key] was never called """
        self._used_params.update(keys)

    def get_unused_params(self):
        return [k for k in self._values.keys() if k not in self._used_params]

    def values(self):
        return list(self._values_to_spec())

    def values_str(self, joiner=' '):
        return joiner.join(self.values())

    def reset(self):  # ugly, needed because global...
        """ Reset global_config. """
        self._values = {}
        self._used_params = set()

    @contextmanager
    def reset_after(self):
        yield
        self.reset()

    def _values_to_spec(self):
        for k, v in sorted(self._values.items()):
            if v is True:
                yield k
            else:
                yield '{}={}'.format(k, v)

    def _ensure_not_specified(self, key, incompatible):
        """ Raises ValueError if any of the keys in `incompatible` were specified """
        assert isinstance(incompatible, list)
        errors = [k for k in incompatible if k in self]
        if errors:
            raise ValueError(f"Got {key}, incompatible with: {','.join(errors)}")

    def __str__(self):
        if len(self._values) == 0:
            return 'GlobalConfig()'
        return 'GlobalConfig(\n\t{})'.format('\n\t'.join(self.values()))


global_config = _GlobalConfig()

