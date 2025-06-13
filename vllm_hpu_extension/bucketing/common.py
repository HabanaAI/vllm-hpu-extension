import os
from typing import Dict
import inspect
from dataclasses import dataclass, field
from typing import List, Tuple
from weakref import WeakValueDictionary

class WeakSingleton(type):
    """
    A metaclass that creates a WeakSingleton instance. This ensures that only one instance of the class exists.
    WeakSingleton doesn't hold a strong reference to the instance, allowing it to be garbage collected when no longer in use.

    Attributes:
        _instances (Dict[type, object]): A dictionary to store the single instance of each class.
        _instances_argspec (Dict[type, object]): A dictionary to store the argument specifications of each instance.

    Methods:
        __call__(cls, *args, **kwargs):
            Creates or returns the single instance of the class. If the instance already exists, it checks that the 
            arguments used to create the instance are the same as the ones provided. Raises an assertion error if 
            the arguments differ.
    """
    # NOTE(kzawora): The instances are stored in a weakref dictionary, 
    # which allows the instances to be garbage collected when they are 
    # no longer in use. This is important for tests, where model runner 
    # can get constructed and destroyed multiple times, and we don't 
    # want to reuse the bucketing context from the previous instance.
    _instances: WeakValueDictionary[type, object] = WeakValueDictionary()
    _instances_argspec: Dict[type, object] = {}

    def __call__(cls, *args, **kwargs):
        argspec = inspect.getcallargs(super().__call__, args, kwargs)
        if cls not in cls._instances:
            # NOTE(kzawora): *DO NOT* assign to self._instances[cls] here,
            # we need this variable to to force a strong reference.
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
            cls._instances_argspec[cls] = argspec
        assert cls._instances_argspec[cls] == argspec, "Singleton instance already initialized with different arguments"
        return cls._instances[cls]


class HPUBucketingManager():
    prompt_buckets: List[Tuple[int, int, int]] = field(init=False)
    decode_buckets: List[Tuple[int, int, int]] = field(init=False)
    max_prompt_config: Tuple[int, int, int]
    max_decode_config: Tuple[int, int, int]

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

    def hello(self):
        # sanity check
        print("hello from manager")

    def get_bucketing_strategy(self, prompt_strategy = None, decode_strategy = None):
        strategy = None
        # TODO check if strategy
        use_exponential_bucketing = os.environ.get(
            'VLLM_EXPONENTIAL_BUCKETING', 'true').lower() == 'true'
        if use_exponential_bucketing:
            from vllm_hpu_extension.bucketing.exponential import (
                HPUExponentialBucketingContext as HPUBucketingContext)
        else:
            from vllm_hpu_extension.bucketing.linear import LinearBucketingStrategy
            strategy = LinearBucketingStrategy()
        return strategy

    def generate_prompt_buckets(self, prompt_strategy = None):
        strategy = self.get_bucketing_strategy(prompt_strategy=prompt_strategy)

        self.prompt_buckets = sorted(strategy.get_prompt_buckets(
                            max_num_prefill_seqs = self.max_num_prefill_seqs,
                            block_size = self.block_size,
                            max_num_batched_tokens = self.max_num_batched_tokens,
                            max_prompt_seq = self.max_prompt_seq,
                            max_model_len = self.max_model_len,
                            prefix_caching = self.prefix_caching))
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
        print("looking for:", batch_size, seq_len, ctx_len, is_prompt)
        found_bucket = find_equal_or_closest_greater_config(buckets, (batch_size, seq_len, ctx_len))
        print("closest: ", found_bucket)
        return found_bucket


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
