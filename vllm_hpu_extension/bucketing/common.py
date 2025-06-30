import os
from typing import Dict
import inspect
from dataclasses import dataclass, field
from typing import List, Tuple

from vllm_hpu_extension.logger import logger as logger
from vllm_hpu_extension.runtime import get_config


class HPUBucketingManager():
    _instance = None
    prompt_buckets: List[Tuple[int, int, int]] = field(init=False)
    decode_buckets: List[Tuple[int, int, int]] = field(init=False)
    prompt_seq_cfg: Tuple[int, int, int]
    prompt_bs_cfg: Tuple[int, int, int]
    decode_ctx_cfg: Tuple[int, int, int]
    decode_bs_cfg: Tuple[int, int, int]

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(HPUBucketingManager, cls).__new__(cls)
        return cls._instance

    def __init__(self, max_num_seqs, max_num_prefill_seqs, block_size,
                 max_num_batched_tokens, max_model_len):
        self.max_num_seqs = max_num_seqs
        self.max_num_prefill_seqs = max_num_prefill_seqs
        self.block_size = block_size
        self.max_num_batched_tokens = max_num_batched_tokens
        self.num_hpu_blocks = None
        self.max_model_len = max_model_len

    def get_bucketing_strategy(self):
        strategy = None
        # TODO - we can use different startegies for decode and prompt
        use_exponential_bucketing = get_config().VLLM_EXPONENTIAL_BUCKETING
        
        if use_exponential_bucketing:
            from vllm_hpu_extension.bucketing.exponential import (
                ExponentialBucketingStrategy)
            strategy = ExponentialBucketingStrategy()
        else:
            from vllm_hpu_extension.bucketing.linear import LinearBucketingStrategy
            strategy = LinearBucketingStrategy()
        return strategy

    def generate_prompt_buckets(self):
        strategy = self.get_bucketing_strategy()

        self.prompt_buckets, self.prompt_seq_cfg, self.prompt_bs_cfg = \
                            strategy.get_prompt_buckets(
                            max_num_prefill_seqs = self.max_num_prefill_seqs,
                            block_size = self.block_size,
                            max_num_batched_tokens = self.max_num_batched_tokens,
                            max_model_len = self.max_model_len)
        self.log_generate_info(True)
        return

    def generate_decode_buckets(self, num_max_blocks):
        strategy = self.get_bucketing_strategy()

        self.decode_buckets, self.decode_ctx_cfg, self.decode_bs_cfg = \
                            strategy.get_decode_buckets(
                            max_num_seqs = self.max_num_seqs, 
                            block_size = self.block_size, 
                            max_num_batched_tokens = self.max_num_batched_tokens,
                            max_model_len = self.max_model_len, 
                            num_max_blocks = num_max_blocks)
        self.log_generate_info(False)
        return

    def log_generate_info(self, is_prompt):
        phase = 'prompt' if is_prompt else 'decode'
        buckets = self.prompt_buckets if is_prompt else self.decode_buckets
        bs_config = self.prompt_bs_cfg if is_prompt else self.decode_bs_cfg
        msg = (f"{phase.capitalize()} bucket config (min, step, max_warmup) "
               f"bs:{bs_config}, "
               f"{'seq' if is_prompt else 'blocks'}:"
               f"{self.prompt_seq_cfg if is_prompt else self.decode_ctx_cfg}")
        logger().info(msg)

        msg = (f"Generated {len(buckets)} "
               f"{phase} buckets [bs, seq]: "
               f"{list(buckets)}")
        logger().info(msg)

    def find_bucket(self, batch_size, seq_len, ctx_len, is_prompt):
        buckets = self.prompt_buckets if is_prompt else self.decode_buckets
        found_bucket = find_equal_or_closest_greater_config(buckets, (batch_size, seq_len, ctx_len))
        if found_bucket is None:
           logger().warning(f"Bucket for {batch_size, seq_len, ctx_len, \
                       'prompt' if is_prompt else 'decode'} was not previously warmed up")
           new_bucket = (batch_size, seq_len, ctx_len)
           self.prompt_buckets.append(new_bucket) if is_prompt else self.decode_buckets.append(new_bucket)
           return new_bucket
        return found_bucket

    def get_max_prompt_shape(self):
        return self.prompt_seq_cfg[-1]

    @classmethod
    def get_instance(cls):
        """
        Retrieve the singleton instance of the class.
        """
        return cls._instance


def get_bucketing_manager():
    instance = HPUBucketingManager.get_instance()
    return instance 


def find_equal_or_closest_greater_config(sorted_list, target_tuple):
    l, r = 0, len(sorted_list) - 1
    result = None

    while l <= r:
        m = (l + r) // 2
        if sorted_list[m] < target_tuple:
            l = m + 1
        else:
            result = sorted_list[m]
            r = m - 1

    return result
