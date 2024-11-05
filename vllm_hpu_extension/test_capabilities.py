###############################################################################
# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.
###############################################################################

import pytest

from vllm_hpu_extension.capabilities import VersionRange, Capabilities


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
def capabilities():
    def feature(value):
        def check(key):
            assert key == "value"
            return value
        return check

    env = {"key": "value"}
    return Capabilities({
        "foo": feature(True),
        "bar": feature(False),
        "qux": feature(True),
    }, env)


def test_capability_repr(capabilities):
    capabilities_str = str(capabilities)
    assert "-bar" in capabilities_str
    assert "+foo" in capabilities_str
    assert "+qux" in capabilities_str


def test_capability_checks(capabilities):
    assert "bar" not in capabilities
    assert "foo" in capabilities
    assert "qux" in capabilities
    assert "foo,qux" in capabilities
    assert "qux,foo" in capabilities
    assert "foo,bar,qux" not in capabilities


def test_capability_signed_checks(capabilities):
    assert "-bar" in capabilities
    assert "+foo" in capabilities
    assert "+foo,-bar,+qux" in capabilities
    assert "+foo,bar,+qux" not in capabilities
    assert "-foo,-bar,+qux" not in capabilities
