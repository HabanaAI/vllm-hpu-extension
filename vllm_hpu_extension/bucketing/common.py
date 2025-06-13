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

        self.prompt_buckets = strategy.get_prompt_buckets(self.max_num_prefill_seqs, self.block_size,
                           self.max_num_batched_tokens, self.max_prompt_seq, self.max_model_len)
        return

    def generate_decode_buckets(self, num_max_blocks, decode_strategy = None):
        strategy = self.get_bucketing_strategy(decode_strategy=decode_strategy)

        self.decode_buckets = strategy.get_decode_buckets(max_num_seqs = self.max_num_seqs, 
                                                          block_size = self.block_size, 
                                                          max_num_batched_tokens = self.max_num_batched_tokens, 
                                                          max_decode_seq = self.max_decode_seq, 
                                                          max_model_len = self.max_model_len, 
                                                          num_max_blocks = num_max_blocks)
        return

    def find_bucket():
        pass



'''
    global_state = HPUBucketingGlobalState()

    @property
    def prompt_buckets(self):
        return self.global_state.prompt_buckets

    @property
    def decode_buckets(self):
        return self.global_state.decode_buckets

    def sorted_prompt_buckets(self):
        return self.global_state.prompt_buckets)

    def get_max_prompt_shape(self):
        return (self.global_state.prompt_bs_bucket_cfg[-1],
                self.global_state.prompt_seq_bucket_cfg[-1])

    def get_padded_prompt_batch_size(self, batch_size):
        return find_bucket(batch_size,
                           self.global_state.prompt_bs_bucket_cfg)

    def get_padded_decode_batch_size(self, batch_size):
        return find_bucket(batch_size,
                           self.global_state.decode_bs_bucket_cfg)

    def get_padded_prompt_seq_len(self, seq_len):
        return find_bucket(seq_len,
                           self.global_state.prompt_seq_bucket_cfg)

    def get_padded_decode_num_blocks(self, num_blocks):
        assert self.num_hpu_blocks is not None, "num_hpu_blocks is not set"
        bucket_size = find_bucket(num_blocks,
                                  self.global_state.decode_block_bucket_cfg)
        return min(bucket_size, self.num_hpu_blocks)

    def get_padded_batch_size(self, batch_size, is_prompt):
        if is_prompt:
            return self.get_padded_prompt_batch_size(batch_size)
        return self.get_padded_decode_batch_size(batch_size)

    def get_padded_seq_or_block(self, seq_or_block, is_prompt):
        if is_prompt:
            return self.get_padded_prompt_seq_len(seq_or_block)
        return self.get_padded_decode_num_blocks(seq_or_block)



def get_bucketing_context():
    use_exponential_bucketing = os.environ.get(
        'VLLM_EXPONENTIAL_BUCKETING', 'true').lower() == 'true'
    print("wybieramy bucketing")
    if use_exponential_bucketing:
        from vllm_hpu_extension.bucketing.exponential import (
            HPUExponentialBucketingContext as HPUBucketingContext)
    else:
        from vllm_hpu_extension.bucketing.linear import HPUBucketingContext

    return HPUBucketingContext

def next_pow2(value: int, base: int) -> int:
    res = base
    while value > res:
        res *= 2
    return res


def round_up(value: int, k: int) -> int:
    return (value + k - 1) // k * k


def find_bucket(value: int, config: Tuple[int, int, int]) -> int:
    bmin, bstep, _ = config
    if value <= bmin:
        return bmin
    else:
        next_step = round_up(value, bstep)
        next_pow = next_pow2(value, bmin)
        return next_pow if next_pow <= bstep else next_step
'''
