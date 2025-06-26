import os
from typing import Dict
import inspect
from dataclasses import dataclass, field
from typing import List, Tuple


class HPUBucketingManager():
    _instance = None
    prompt_buckets: List[Tuple[int, int, int]] = field(init=False)
    decode_buckets: List[Tuple[int, int, int]] = field(init=False)
    prompt_seq_cfgi: Tuple[int, int, int]
    max_prompt_config: Tuple[int, int, int]
    max_decode_config: Tuple[int, int, int]

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(HPUBucketingManager, cls).__new__(cls)
        return cls._instance

    def __init__(self, max_num_seqs, max_num_prefill_seqs, block_size,
                 max_num_batched_tokens, use_merged_prefill, prefix_caching,
                 max_model_len, max_prompt_seq=None, max_decode_seq=None):
        self.max_num_seqs = max_num_seqs
        self.max_num_prefill_seqs = max_num_prefill_seqs
        self.block_size = block_size
        self.max_num_batched_tokens = max_num_batched_tokens
        self.num_hpu_blocks = None
        self.max_model_len = max_model_len
        self.max_prompt_seq = max_prompt_seq
        self.max_decode_seq = max_decode_seq
        self.prefix_caching = prefix_caching

    def get_bucketing_strategy(self, prompt_strategy = None, decode_strategy = None):
        strategy = None
        # TODO check if strategy
        use_exponential_bucketing = os.environ.get(
            'VLLM_EXPONENTIAL_BUCKETING', 'true').lower() == 'true'
        if use_exponential_bucketing:
            from vllm_hpu_extension.bucketing.exponential import (
                ExponentialBucketingStrategy)
            strategy = ExponentialBucketingStrategy()
        else:
            from vllm_hpu_extension.bucketing.linear import LinearBucketingStrategy
            strategy = LinearBucketingStrategy()
        return strategy

    def generate_prompt_buckets(self, prompt_strategy = None):
        strategy = self.get_bucketing_strategy(prompt_strategy=prompt_strategy)

        self.prompt_buckets, self.prompt_seq_cfg = strategy.get_prompt_buckets(
                            max_num_prefill_seqs = self.max_num_prefill_seqs,
                            block_size = self.block_size,
                            max_num_batched_tokens = self.max_num_batched_tokens,
                            max_prompt_seq = self.max_prompt_seq,
                            max_model_len = self.max_model_len,
                            prefix_caching = self.prefix_caching)
        return

    def generate_decode_buckets(self, num_max_blocks, decode_strategy = None):
        strategy = self.get_bucketing_strategy(decode_strategy=decode_strategy)

        self.decode_buckets = sorted(strategy.get_decode_buckets(
                            max_num_seqs = self.max_num_seqs, 
                            block_size = self.block_size, 
                            max_num_batched_tokens = self.max_num_batched_tokens, 
                            max_decode_seq = self.max_decode_seq, 
                            max_model_len = self.max_model_len, 
                            num_max_blocks = num_max_blocks,
                            prefix_caching = self.prefix_caching))
        return

    def find_bucket(self, batch_size, seq_len, ctx_len, is_prompt):
        buckets = self.prompt_buckets if is_prompt else self.decode_buckets
        found_bucket = find_equal_or_closest_greater_config(buckets, (batch_size, seq_len, ctx_len))
        if found_bucket is None:
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

        Returns:
            The singleton instance of the class.

        Raises:
            AssertionError: If the class has not been initialized and no instance exists.
        """
        if cls._instance is None:
            cls._instance = cls()
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
