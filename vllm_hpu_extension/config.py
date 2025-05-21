###############################################################################
# Copyright (C) 2024-2025 Intel Corporation
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.
###############################################################################

from packaging.version import Version
from packaging.specifiers import SpecifierSet
import os
import itertools


class Config:
    def __init__(self, *values, **kwargs):
        self._data = dict(itertools.chain(*[v.items() for v in values] + [kwargs.items()]))

    def __getattr__(self, name):
        return self.get(name)

    def get(self, key):
        assert key in self._data, f'Unknown run-time configuration parameter: {key}!'
        value = self._data.get(key)
        if callable(value):
            value = value(self)
            self._data[key] = value
        return value

    def get_all(self, keys=None):
        keys = keys or self._data.keys()
        return {k: self.get(k) for k in keys}


def All(*parts):
    return lambda cfg: all(p(cfg) for p in parts)


def Not(fn):
    return lambda cfg: not fn(cfg)


def Eq(key, value):
    return lambda cfg: (cfg.get(key) == value)


def Active(name, value=True):
    return Eq(name, value)


def FirstActive(*options):
    return lambda cfg: next((o for o in options if cfg.get(o)), None)


def Kernel(loader_fn):
    return lambda _: loader_fn() is not None


def Lazy():
    return Eq('mode', 'lazy')


def Hardware(target_hw):
    return Eq('hw', target_hw)


def ModelType(target_hw):
    return Eq('model_type', target_hw)


def VersionRange(*specifiers):
    specifiers = [SpecifierSet(s) for s in specifiers]
    return lambda cfg: any(Version(cfg.build) in s for s in specifiers)


def choice(*options):
    def choice_impl(x):
        assert x in options, f'{x} is not in allowed options: {options}!'
        return x
    return choice_impl


def boolean(x):
    return x.lower()[0] in ['t', '1']


class Flag:
    def __init__(self, name, constructor):
        self.name = name
        self.constructor = constructor

    def __call__(self, config):
        value = os.environ.get(self.name)
        if value is not None:
            return self.constructor(value)
        return None


class Parameter:
    def __init__(self, name, dependencies, env_var=None, env_var_type=boolean):
        self.name = name
        self.env_var = env_var if env_var is not None else 'VLLM_' + name.upper()
        self.env_var_type = env_var_type
        self.dependencies = dependencies

    def to_flag(self):
        return Flag(self.env_var, self.env_var_type)

    def __call__(self, config):
        if (override := config.get(self.env_var)) is not None:
            return override
        if callable(self.dependencies):
            return self.dependencies(config)
        return self.dependencies
