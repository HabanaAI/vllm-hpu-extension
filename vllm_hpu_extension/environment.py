###############################################################################
# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.
###############################################################################

from pathlib import Path
import re


def get_build():
    import habana_frameworks.torch as htorch
    # This is ugly as hell, but it's the only reliable way of querying build number
    # from python that doesn't involve parsing external process output.
    # version.py can be found in gpu-migration package which means we cannot load it
    # directly as this would trigger enabling that feature. Instead we parse the file.
    version_re = re.compile(r'__version__\s+=\s+"(?P<version>.+)"')
    path = Path(htorch.__file__).parent
    for p in path.rglob("version.py"):
        if m := version_re.search(open(p).read()):
            return m.group('version')


def get_environment(**overrides):
    overrides = {k: lambda: v for k, v in overrides.items()}
    getters = {
        "build": get_build
    }
    return {k: g() for k, g, in (getters | overrides).items()}
