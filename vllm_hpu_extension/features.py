###############################################################################
# Copyright (C) 2025 Intel Corporation
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.
###############################################################################

from vllm_hpu_extension.config import Not, Hardware, VersionRange, ModelType, Kernel, FirstEnabled, All, Value, Env, Disabled, Engine, choice, boolean, to_dict, split_values_and_flags
from vllm_hpu_extension.kernels import fsdpa, block_softmax_adjustment


def get_user_flags():
    flags = [
        Env('VLLM_USE_V1', boolean),
        Env('VLLM_ENABLE_EXPERIMENTAL_FLAGS', boolean),
        Env('VLLM_EXPONENTIAL_BUCKETING', boolean),
        Env('VLLM_PROMPT_BS_BUCKET_MIN', int),
        Env('VLLM_PROMPT_BS_BUCKET_STEP', int),
        Env('VLLM_PROMPT_BS_BUCKET_MAX', int),
        Env('VLLM_PROMPT_BS_BUCKET_LIMIT', int),
        Env('VLLM_PROMPT_SEQ_BUCKET_MIN', int),
        Env('VLLM_PROMPT_SEQ_BUCKET_STEP', int),
        Env('VLLM_PROMPT_SEQ_BUCKET_MAX', int),
        Env('VLLM_PROMPT_SEQ_BUCKET_LIMIT', int),
        Env('VLLM_DECODE_BS_BUCKET_MIN', int),
        Env('VLLM_DECODE_BS_BUCKET_STEP', int),
        Env('VLLM_DECODE_BS_BUCKET_MAX', int),
        Env('VLLM_DECODE_BS_BUCKET_LIMIT', int),
        Env('VLLM_DECODE_BLOCK_BUCKET_MIN', int),
        Env('VLLM_DECODE_BLOCK_BUCKET_STEP', int),
        Env('VLLM_DECODE_BLOCK_BUCKET_MAX', int),
        Env('VLLM_DECODE_BLOCK_BUCKET_LIMIT', int),

        # Non-vllm flags that are also important to print
        Env('EXPERIMENTAL_WEIGHT_SHARING', str),
        Env('PT_HPU_WEIGHT_SHARING', str),
        
        # Sliding window flags
        Env('PT_HPU_SDPA_QKV_SLICE_MODE_FWD', boolean),
        Env('PT_HPU_SDPA_BC_FACTOR', int),
        Env('VLLM_FUSEDSDPA_SLIDE_THLD', int),
    ]
    return to_dict(flags)


def get_experimental_flags():
    flags = [
        Env('VLLM_PT_PROFILE', str),
        Env('VLLM_PROFILE_PROMPT', str),
        Env('VLLM_PROFILE_DECODE', str),
    ]
    return to_dict(flags)


def get_features():
    supported_attn_impls = ['flex_impl', 'fsdpa_impl', 'naive_impl']
    bucketing_strategies = ['exponential_bucketing', 'linear_bucketing']
    features = [
        Value('fp32_alibi_biases', True, env_var='VLLM_ALIBI_USE_FLOAT32_BIASES'),
        # May manually enable this flag for Qwen2 and Qwen2-MoE models to get better accuracy
        Value('fp32_softmax', False, env_var='VLLM_FP32_SOFTMAX'),
        Value('fused_block_softmax_adjustment', All(VersionRange(">=1.22.0.494"),
                                                    Hardware('gaudi3'),
                                                    Kernel(block_softmax_adjustment),
                                                    Not(ModelType('qwen2')))),
        Value('fused_block_softmax', False),
        Value('flex_impl', False, env_var='VLLM_PROMPT_USE_FLEX_ATTENTION'),
        Value('fsdpa_impl', All(Kernel(fsdpa),
                                Not(ModelType('mllama'))), env_var='VLLM_PROMPT_USE_FUSEDSDPA'),
        Value('naive_impl', True),
        Value('prompt_attn_impl', FirstEnabled(*supported_attn_impls), env_var_type=choice(*supported_attn_impls)),
        Value('skip_warmup', False),
        Value('merged_prefill', False),
        Value('use_contiguous_pa', Disabled('prefix_caching'), env_var='VLLM_CONTIGUOUS_PA'),
        Value('use_delayed_sampling', Engine('v0'), env_var='VLLM_DELAYED_SAMPLING'),
        Value('use_bucketing', True, env_var='VLLM_ENABLE_BUCKETING'),
        Value('exponential_bucketing', True, env_var='VLLM_EXPONENTIAL_BUCKETING'), 
        Value('linear_bucketing', True),
        Value('bucketing_strategy', FirstEnabled(*bucketing_strategies), env_var_type=choice(*bucketing_strategies)),
        Value('use_const_norm', False, env_var='VLLM_SOFTMAX_CONST_NORM'),
        Value('const_norm_value', 10.0, env_var='VLLM_SOFTMAX_CONST_NORM_VALUE'),
    ]
    return split_values_and_flags(features)
