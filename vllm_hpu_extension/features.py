###############################################################################
# Copyright (C) 2025 Intel Corporation
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.
###############################################################################

from vllm_hpu_extension.config import Not, Hardware, VersionRange, ModelType, Kernel, Lazy, FirstActive, All, Parameter, Flag, choice, boolean
from vllm_hpu_extension.kernels import fsdpa, block_softmax_adjustment


def get_user_flags():
    return [
        Flag('VLLM_ENABLE_EXPERIMENTAL_FLAGS', boolean),
        Flag('VLLM_PROMPT_BS_BUCKET_MIN', int),
        Flag('VLLM_PROMPT_BS_BUCKET_STEP', int),
        Flag('VLLM_PROMPT_BS_BUCKET_MAX', int),
        Flag('VLLM_PROMPT_SEQ_BUCKET_MIN', int),
        Flag('VLLM_PROMPT_SEQ_BUCKET_STEP', int),
        Flag('VLLM_PROMPT_SEQ_BUCKET_MAX', int),
        Flag('VLLM_DECODE_BS_BUCKET_MIN', int),
        Flag('VLLM_DECODE_BS_BUCKET_STEP', int),
        Flag('VLLM_DECODE_BS_BUCKET_MAX', int),
        Flag('VLLM_DECODE_BLOCK_BUCKET_MIN', int),
        Flag('VLLM_DECODE_BLOCK_BUCKET_STEP', int),
        Flag('VLLM_DECODE_BLOCK_BUCKET_MAX', int),
    ]


def get_experimental_flags():
    return [
        Flag('VLLM_PT_PROFILE', str),
        Flag('VLLM_PROFILE_PROMPT', str),
        Flag('VLLM_PROFILE_DECODE', str),
    ]


def get_features():
    supported_attn_impls = ['flex_impl', 'fsdpa_impl', 'naive_impl']
    return [
        Parameter('fp32_softmax', ModelType('qwen2')),
        Parameter('compile_one_hot', All(VersionRange(">=1.20.0.370"),
                                         Not(Lazy()))),
        Parameter('fused_block_softmax_adjustment', All(VersionRange(">=1.22.0.101"),
                                                        Hardware('gaudi3'),
                                                        Kernel(block_softmax_adjustment),
                                                        Not(ModelType('qwen2')))),
        Parameter('flex_impl', False, env_var='VLLM_PROMPT_USE_FLEX_ATTENTION'),
        Parameter('fsdpa_impl', All(Kernel(fsdpa),
                                    Not(ModelType('mllama'))), env_var='VLLM_PROMPT_USE_FUSEDSDPA'),
        Parameter('naive_impl', True),
        Parameter('prompt_attn_impl', FirstActive(*supported_attn_impls), env_var_type=choice(*supported_attn_impls)),
        Parameter('skip_warmup', False, env_var='VLLM_SKIP_WARMUP'),
        Parameter('contiguous_pa', True),
        Parameter('delayed_sampling', True),
    ]
