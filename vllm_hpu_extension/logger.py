###############################################################################
# Copyright (C) 2025 Intel Corporation
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.
###############################################################################

from functools import cache


@cache
def logger():
    try:
        from vllm.logger import init_logger
        return init_logger("vllm")
    except ImportError:
        import logging
        return logging.getLogger("vllm")

    
def init_debug_logger(area):
    from vllm_hpu_extension.runtime import get_config
    if area in get_config().VLLM_DEBUG:
        def enabled_dbg_logger(msg):
            logger().warning(f'[debug/{area}] {msg}')
        return enabled_dbg_logger
    else:
        def disabled_dbg_logger(_):
            pass
        return disabled_dbg_logger
