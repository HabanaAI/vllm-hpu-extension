###############################################################################
# Copyright (C) 2024-2025 Intel Corporation
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.
###############################################################################

from .utils import logger
from .config import Parameter, choice


_RUNTIME_PARAMS = {}


def _get_hw(_):
    import habana_frameworks.torch.utils.experimental as htexp
    device_type = htexp._get_device_type()
    match device_type:
        case htexp.synDeviceType.synDeviceGaudi2:
            return "gaudi2"
        case htexp.synDeviceType.synDeviceGaudi3:
            return "gaudi3"
    from vllm_hpu_extension.utils import is_fake_hpu
    if is_fake_hpu():
        return "cpu"
    logger().warning(f'Unknown device type: {device_type}')
    return None


def _get_build(_):
    import re
    import subprocess
    output = subprocess.run("pip show habana-torch-plugin",
                            shell=True,
                            text=True,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    version_re = re.compile(r'Version:\s*(?P<version>.*)')
    match = version_re.search(output.stdout)
    if output.returncode == 0 and match:
        return match.group('version')
    # In cpu-test environment we don't have access to habana-torch-plugin
    from vllm_hpu_extension.utils import is_fake_hpu
    result = '0.0.0.0' if is_fake_hpu() else None
    logger().warning(f"Unable to detect habana-torch-plugin version! Returning: {result}")
    return result


def _get_model_type(_):
    return _RUNTIME_PARAMS.get('model_type', None)


def set_model_config(cfg):
    global _RUNTIME_PARAMS
    if hasattr(cfg, 'hf_config'):
        _RUNTIME_PARAMS['model_type'] = cfg.hf_config.model_type
    else:
        _RUNTIME_PARAMS['model_type'] = cfg.model_type


def _get_vllm_engine(_):
    import vllm.envs as envs
    return 'v1' if envs.VLLM_USE_V1 else 'v0'


def _get_pt_bridge_mode(_):
    import habana_frameworks.torch as htorch
    return 'lazy' if htorch.utils.internal.is_lazy() else 'eager'


def get_environment():
    return [
        Parameter('hw', _get_hw, env_var_type=choice('cpu', 'gaudi', 'gaudi2', 'gaudi3')),
        Parameter('model_type', _get_model_type, env_var_type=str),
        Parameter('build', _get_build, env_var_type=str),
        Parameter('engine', _get_vllm_engine, env_var_type=str),
        Parameter('mode', _get_pt_bridge_mode, env_var_type=choice('eager', 'lazy')),
    ]
