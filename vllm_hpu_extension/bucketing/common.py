import os
import bisect
from typing import Dict
import inspect
from dataclasses import dataclass, field
from typing import List, Tuple

from vllm_hpu_extension.logger import logger as logger
from vllm_hpu_extension.runtime import get_config


class HPUBucketingManager():
    _instance = None
    prompt_buckets: List[Tuple[int, int, int]] = []
    decode_buckets: List[Tuple[int, int, int]] = []
    prompt_seq_cfg: Tuple[int, int, int]
    prompt_bs_cfg: Tuple[int, int, int]
    decode_ctx_cfg: Tuple[int, int, int]
    decode_bs_cfg: Tuple[int, int, int]
    initialized = False

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(HPUBucketingManager, cls).__new__(cls)
        return cls._instance

    def initialize(self, max_num_seqs, max_num_prefill_seqs, block_size,
                   max_num_batched_tokens, max_model_len):
        self.max_num_seqs = max_num_seqs
        self.max_num_prefill_seqs = max_num_prefill_seqs
        self.block_size = block_size
        self.max_num_batched_tokens = max_num_batched_tokens
        self.num_hpu_blocks = None
        self.max_model_len = max_model_len
        self.initialized = True

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
        if self.initialized:
            strategy = self.get_bucketing_strategy()

            self.prompt_buckets, self.prompt_seq_cfg, self.prompt_bs_cfg = \
                            strategy.get_prompt_buckets(
                            max_num_prefill_seqs = self.max_num_prefill_seqs,
                            block_size = self.block_size,
                            max_num_batched_tokens = self.max_num_batched_tokens,
                            max_model_len = self.max_model_len)
            self.log_generate_info(True)
        else:
            logger().info("Bucketing is off - skipping prompt buckets generation")
            self.prompt_buckets = []
        return

    def generate_decode_buckets(self, num_max_blocks):
        if self.initialized:
            strategy = self.get_bucketing_strategy()

            self.decode_buckets, self.decode_ctx_cfg, self.decode_bs_cfg = \
                            strategy.get_decode_buckets(
                            max_num_seqs = self.max_num_seqs,
                            block_size = self.block_size, 
                            max_num_batched_tokens = self.max_num_batched_tokens,
                            max_model_len = self.max_model_len, 
                            num_max_blocks = num_max_blocks)
            self.log_generate_info(False)
        else:
            logger().info("Bucketing is off - skipping decode buckets generation")
            self.decode_buckets = []
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
               f"{phase} buckets [bs, query, num_blocks]: "
               f"{list(buckets)}")
        logger().info(msg)

    def find_prompt_bucket(self, batch_size, seq_len, ctx=0):
        if self.initialized:
            found_bucket = find_equal_or_closest_greater_config(self.prompt_buckets, (batch_size, seq_len, ctx))
            if found_bucket is None:
                logger().warning(f"Prompt bucket for {batch_size, seq_len, ctx}"
                                 " was not previously warmed up")
                new_bucket = (batch_size, seq_len, ctx)
                self.prompt_buckets.append(new_bucket)
                return new_bucket
            return found_bucket
        return (batch_size, seq_len, ctx)

    def find_decode_bucket(self, batch_size, num_blocks):
        if self.initialized:
            found_bucket = find_equal_or_closest_greater_config(self.decode_buckets, (batch_size, 1, num_blocks))
            if found_bucket is None:
                logger().warning(f"Decode bucket for {batch_size, 1, num_blocks}"
                                 " was not previously warmed up")
                new_bucket = (batch_size, 1, num_blocks)
                self.decode_buckets.append(new_bucket)
                return new_bucket
            return found_bucket
        return (batch_size, 1, num_blocks)

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
    result = bisect.bisect_left(sorted_list, target_tuple)
    return sorted_list[result] if result < len(sorted_list) else None

