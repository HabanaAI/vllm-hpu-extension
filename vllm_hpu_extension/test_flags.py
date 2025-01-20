###############################################################################
# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.
###############################################################################

import pytest
from unittest.mock import patch, MagicMock

from vllm_hpu_extension.flags import VersionRange, Flags, EnvFlag, Value, Kernel, FeatureTest, Not


def test_operators():
    assert VersionRange("<1.19.0.100")(build="1.19.0.99")
    assert not VersionRange("<1.19.0.100")(build="1.19.0.100")
    assert not VersionRange("<1.19.0.100")(build="1.19.0.101")

    assert VersionRange("<=1.19.0.100")(build="1.19.0.99")
    assert VersionRange("<=1.19.0.100")(build="1.19.0.100")
    assert not VersionRange("<=1.19.0.100")(build="1.19.0.101")

    assert not VersionRange("==1.19.0.100")(build="1.19.0.99")
    assert VersionRange("==1.19.0.100")(build="1.19.0.100")
    assert not VersionRange("==1.19.0.100")(build="1.19.0.101")

    assert not VersionRange(">1.19.0.100")(build="1.19.0.99")
    assert not VersionRange(">1.19.0.100")(build="1.19.0.100")
    assert VersionRange(">1.19.0.100")(build="1.19.0.101")

    assert not VersionRange(">=1.19.0.100")(build="1.19.0.99")
    assert VersionRange(">=1.19.0.100")(build="1.19.0.100")
    assert VersionRange(">=1.19.0.100")(build="1.19.0.101")


def test_two_ranges_same_release():
    assert not VersionRange(">1.19.0.100,<1.19.0.200")(build="1.19.0.50")
    assert VersionRange(">1.19.0.100,<1.19.0.200")(build="1.19.0.150")
    assert not VersionRange(">1.19.0.100,<1.19.0.200")(build="1.19.0.250")


def test_multiple_ranges_same_release():
    ver_check = VersionRange(">=1.19.0.100,<1.19.0.200", ">=1.19.0.300,<1.19.0.400")
    assert not ver_check(build="1.19.0.50")
    assert ver_check(build="1.19.0.100")
    assert ver_check(build="1.19.0.150")
    assert not ver_check(build="1.19.0.200")
    assert not ver_check(build="1.19.0.250")
    assert ver_check(build="1.19.0.300")
    assert ver_check(build="1.19.0.350")
    assert not ver_check(build="1.19.0.400")
    assert not ver_check(build="1.19.0.450")


def test_other_releases():
    ver_check = VersionRange(">=1.19.0.100")
    assert ver_check(build="1.19.1.50")
    assert ver_check(build="1.20.0.50")

    ver_check = VersionRange("<1.19.0.100")
    assert not ver_check(build="1.19.1.50")
    assert not ver_check(build="1.20.0.50")

    ver_check = VersionRange(">=1.19.0.100,<1.20.0")
    assert ver_check(build="1.19.1.50")
    assert not ver_check(build="1.20.0.50")


@pytest.fixture
def flags():
    def feature(value):
        def check(key):
            assert key == "value"
            return value
        return check

    env = {"key": "value"}
    return Flags({
        "foo": feature(True),
        "bar": feature(False),
        "qux": feature(True),
    }, env)


def test_capability_repr(flags):
    flags_str = str(flags)
    assert "-bar" in flags_str
    assert "+foo" in flags_str
    assert "+qux" in flags_str


def test_flag_checks(flags):
    assert "bar" not in flags
    assert "foo" in flags
    assert "qux" in flags
    assert "foo,qux" in flags
    assert "qux,foo" in flags
    assert "foo,bar,qux" not in flags


def test_flag_signed_checks(flags):
    assert "-bar" in flags
    assert "+foo" in flags
    assert "+foo,-bar,+qux" in flags
    assert "+foo,bar,+qux" not in flags
    assert "-foo,-bar,+qux" not in flags


def test_env_flag():
    import os
    os.environ['Alpha'] = 'True'
    os.environ['Omega'] = '0'

    assert EnvFlag('Alpha', False)()
    assert not EnvFlag('Omega', True)()
    assert not EnvFlag('Beta', False)()
    assert EnvFlag('Beta', True)()

    assert EnvFlag('Alpha', Value('key', '???'))(key='value')
    assert not EnvFlag('Omega', Value('key', 'value'))(key='value')
    assert not EnvFlag('Beta', Value('key', '???'))(key='value')
    assert EnvFlag('Beta', Value('key', 'value'))(key='value')


def test_kernel_flag():
    def loader(success):
        def load():
            return success if success else None
        return load
    assert Kernel(loader(True))()
    assert not Kernel(loader(False))()


def test_flag_combinators():
    def create_dummy_feature(value):
        dummy = FeatureTest()
        dummy.check = MagicMock(return_value=value)
        return dummy
    a = create_dummy_feature(True)
    b = create_dummy_feature(False)
    assert (a & a)()
    assert not (Not(a) & a)()
    assert not (a & b)()
    assert (a & Not(b))()
