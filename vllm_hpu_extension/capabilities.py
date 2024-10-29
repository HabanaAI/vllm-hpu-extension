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


class VersionRange:
    def __init__(self, *specifiers):
        self.specifiers = [SpecifierSet(s) for s in specifiers]

    def __call__(self, build, **_):
        version = Version(build)
        return any(version in s for s in self.specifiers)


class Capabilities:
    def __init__(self, features, environment):
        self.all = set(features.keys())
        self.enabled = set(name for name, check in features.items() if check(**environment))

    def is_enabled(self, *names):
        return all(n in self.enabled for n in names)

    def __repr__(self):
        feature_list = [('+' if self.is_enabled(f) else '-') + f for f in sorted(self.all)]
        return f'[{(" ").join(feature_list)}]'

    def __contains__(self, names):
        return self.is_enabled(*names.split(','))


@cache
def capabilities():
    supported_features = {
        "index_copy": VersionRange(">=1.19.0-272"),
    }
    environment = get_environment()
    capabilities = Capabilities(supported_features, environment)
    print(f'Detected capabilities: {capabilities}')
    return capabilities
