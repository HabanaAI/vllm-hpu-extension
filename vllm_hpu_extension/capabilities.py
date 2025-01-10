###############################################################################
# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.
###############################################################################

from packaging.version import Version
from packaging.specifiers import SpecifierSet
from functools import cache

from vllm_hpu_extension.environment import get_environment


class Check:
    def __init__(self, *required_params):
        self.required_params = required_params

    def __call__(self, **kwargs):
        if any(kwargs[rp] is None for rp in self.required_params):
            return False
        return self.check(**kwargs)


class VersionRange(Check):
    def __init__(self, *specifiers):
        super().__init__('build')
        self.specifiers = [SpecifierSet(s) for s in specifiers]

    def check(self, build, **_):
        version = Version(build)
        return any(version in s for s in self.specifiers)


class Hardware(Check):
    def __init__(self, target_hw):
        super().__init__('hw')
        self.target_hw = target_hw

    def check(self, hw, **_):
        return hw == self.target_hw


class Capabilities:
    def __init__(self, features, environment):
        self.all = set(features.keys())
        self.enabled = set(name for name, check in features.items() if check(**environment))
        self.disabled = self.all - self.enabled

    def is_enabled(self, *names):
        return all(n in self.enabled for n in names)

    def is_disabled(self, *names):
        return all(n in self.disabled for n in names)

    def __repr__(self):
        feature_list = [('+' if self.is_enabled(f) else '-') + f for f in sorted(self.all)]
        return f'[{(" ").join(feature_list)}]'

    def _check(self, name):
        if name.startswith('-'):
            return self.is_disabled(name[1:])
        if name.startswith('+'):
            return self.is_enabled(name[1:])
        return self.is_enabled(name)

    def __contains__(self, names):
        return all(self._check(name) for name in names.split(','))


@cache
def capabilities():
    supported_features = {
        "gaudi": Hardware("gaudi"),
        "gaudi2": Hardware("gaudi2"),
        "gaudi3": Hardware("gaudi3"),
        "cpu": Hardware("cpu"),
    }
    environment = get_environment()
    capabilities = Capabilities(supported_features, environment)
    print(f'Detected capabilities: {capabilities}')
    return capabilities
