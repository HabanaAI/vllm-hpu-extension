###############################################################################
# Copyright (C) 2024-2025 Intel Corporation
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.
###############################################################################


from vllm_hpu_extension.environment import get_environment
from vllm_hpu_extension.features import get_features, get_user_flags, get_experimental_flags
from vllm_hpu_extension.config import Config

import torch

DETECTED = None


def to_map(collection):
    return {p.name: p for p in collection}


def flags(params):
    return [p.to_flag() for p in params]


def filter_defined(config, keys):
    return {k: v for k, v in config.get_all(keys).items() if v is not None}


def dump(prefix, values):
    if values:
        print(f'{prefix}:')
        for key, value in values.items():
            print(f'  {key}: {value}')


@torch._dynamo.allow_in_graph
def get_config():

    global DETECTED

    if DETECTED:
        return DETECTED

    user_flags = get_user_flags()
    experimental_flags = get_experimental_flags()
    environment = get_environment()
    features = get_features()

    user_flags = to_map(user_flags)
    experimental_flags = to_map(experimental_flags) | to_map(flags(environment)) | to_map(flags(features))
    environment = to_map(environment)
    features = to_map(features)

    detected = Config(user_flags | experimental_flags | environment | features)
    detected.get_all()

    user_flags = filter_defined(detected, user_flags.keys())
    experimental_flags = filter_defined(detected, experimental_flags.keys())
    environment = filter_defined(detected, environment.keys())
    features = filter_defined(detected, features.keys())

    experimental_flag_names = experimental_flags.keys()
    if len(experimental_flag_names) > 0 and not detected.VLLM_ENABLE_EXPERIMENTAL_FLAGS:
        from .utils import logger
        asterisks = 48 * '*'
        header = f"{asterisks} Warning! {asterisks}"
        footer = '*' * len(header)
        logger().warning(header)
        logger().warning(f"Following environment variables are considered experimental: {', '.join(experimental_flag_names)}")
        logger().warning("In future releases using those flags without VLLM_ENABLE_EXPERIMENTAL_FLAGS will trigger a fatal error.")
        logger().warning(footer)

    dump('Environment', environment)
    dump('Features', features)
    dump('User flags', user_flags)
    dump('Experimental flags', experimental_flags)

    DETECTED = detected

    return detected
