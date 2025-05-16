###############################################################################
# Copyright (C) 2024-2025 Intel Corporation
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.
###############################################################################

import os
import pytest
from vllm_hpu_extension.config import VersionRange, Config, Kernel, Flag, boolean, All, Not, Eq, Active, FirstActive, choice


def with_cfg(fn):
    def sub_fn(**kwargs):
        return fn(Config(kwargs))
    return sub_fn


def Cfg(constructor):
    def sub_constructor(*args, **kwargs):
        obj = constructor(*args, **kwargs)
        return with_cfg(obj)
    return sub_constructor


CfgVersionRange = Cfg(VersionRange)


def test_operators():
    assert CfgVersionRange("<1.19.0.100")(build="1.19.0.99")
    assert not CfgVersionRange("<1.19.0.100")(build="1.19.0.100")
    assert not CfgVersionRange("<1.19.0.100")(build="1.19.0.101")

    assert CfgVersionRange("<=1.19.0.100")(build="1.19.0.99")
    assert CfgVersionRange("<=1.19.0.100")(build="1.19.0.100")
    assert not CfgVersionRange("<=1.19.0.100")(build="1.19.0.101")

    assert not CfgVersionRange("==1.19.0.100")(build="1.19.0.99")
    assert CfgVersionRange("==1.19.0.100")(build="1.19.0.100")
    assert not CfgVersionRange("==1.19.0.100")(build="1.19.0.101")

    assert not CfgVersionRange(">1.19.0.100")(build="1.19.0.99")
    assert not CfgVersionRange(">1.19.0.100")(build="1.19.0.100")
    assert CfgVersionRange(">1.19.0.100")(build="1.19.0.101")

    assert not CfgVersionRange(">=1.19.0.100")(build="1.19.0.99")
    assert CfgVersionRange(">=1.19.0.100")(build="1.19.0.100")
    assert CfgVersionRange(">=1.19.0.100")(build="1.19.0.101")


def test_two_ranges_same_release():
    assert not CfgVersionRange(">1.19.0.100,<1.19.0.200")(build="1.19.0.50")
    assert CfgVersionRange(">1.19.0.100,<1.19.0.200")(build="1.19.0.150")
    assert not CfgVersionRange(">1.19.0.100,<1.19.0.200")(build="1.19.0.250")


def test_multiple_ranges_same_release():
    ver_check = CfgVersionRange(">=1.19.0.100,<1.19.0.200", ">=1.19.0.300,<1.19.0.400")
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
    ver_check = CfgVersionRange(">=1.19.0.100")
    assert ver_check(build="1.19.1.50")
    assert ver_check(build="1.20.0.50")

    ver_check = CfgVersionRange("<1.19.0.100")
    assert not ver_check(build="1.19.1.50")
    assert not ver_check(build="1.20.0.50")

    ver_check = CfgVersionRange(">=1.19.0.100,<1.20.0")
    assert ver_check(build="1.19.1.50")
    assert not ver_check(build="1.20.0.50")


def test_env_flag():
    os.environ['A'] = 'True'
    os.environ['B'] = 'true'
    os.environ['C'] = 't'
    os.environ['D'] = '1'
    os.environ['E'] = 'False'
    os.environ['F'] = 'false'
    os.environ['G'] = 'f'
    os.environ['H'] = '0'

    assert Flag('A', boolean)(None) is True
    assert Flag('B', boolean)(None) is True
    assert Flag('C', boolean)(None) is True
    assert Flag('D', boolean)(None) is True

    assert Flag('E', boolean)(None) is False
    assert Flag('F', boolean)(None) is False
    assert Flag('G', boolean)(None) is False
    assert Flag('H', boolean)(None) is False

    assert Flag('I', boolean)(None) is None

    assert Flag('A', str)(None) == 'True'
    assert Flag('D', str)(None) == '1'
    assert Flag('D', int)(None) == 1


def test_kernel():
    def loader(success):
        def load():
            return success if success else None
        return load
    assert Kernel(loader(True))(None) is True
    assert Kernel(loader(False))(None) is False


def false(_):
    return False


def true(_):
    return True


def none(_):
    return None


def test_combinators__all():
    assert All(none)(None) is False
    assert All(false)(None) is False
    assert All(true)(None) is True
    assert All(false, false)(None) is False
    assert All(false, true)(None) is False
    assert All(true, false)(None) is False
    assert All(true, true)(None) is True
    assert All(true, true, none)(None) is False


def test_combinators__not():
    assert Not(none)(None) is True
    assert Not(false)(None) is True
    assert Not(true)(None) is False


def test_combinators__eq():
    assert Eq('foo', 'bar')({'foo': 'bar'}) is True
    assert Eq('foo', 'bar')({'foo': 'dingo'}) is False
    assert Eq('foo', 'bar')({'dingo': 'bar'}) is False


def test_combinators__active():
    assert Active('foo')({}) is False
    assert Active('foo')({'foo': False}) is False
    assert Active('foo')({'foo': True}) is True


def test_combinators__first_active():
    assert FirstActive('foo', 'bar')({}) is None
    assert FirstActive('foo', 'bar')({'foo': False, 'bar': False}) is None
    assert FirstActive('foo', 'bar')({'foo': False, 'bar': True}) == 'bar'
    assert FirstActive('foo', 'bar')({'foo': True, 'bar': False}) == 'foo'
    assert FirstActive('foo', 'bar')({'foo': True, 'bar': True}) == 'foo'


def test_choice():
    assert choice('a', 'b')('a') == 'a'
    assert choice('a', 'b')('b') == 'b'
    with pytest.raises(AssertionError):
        assert choice('a', 'b')('c')
