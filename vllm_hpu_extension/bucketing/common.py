import os
import bisect
import math
from typing import Dict
import inspect
from dataclasses import dataclass, field
from typing import List, Tuple

from vllm_hpu_extension.logger import logger as logger
from vllm_hpu_extension.runtime import get_config

def calc_fallback_value(n: int, base_step: int):
    """ Calculate next bucket for yet unbucketized value"""
    if n <= 1:
        return n
    power = 1/3
    # The basic idea is that we first estimate bucket size based
    # on exponent of the number, so higher numbers will generate
    # bigger gaps between individual buckets, but it's not as steep
    # as exponential bucketing. Additionally this has a nice
    # property that generated values are guaranteed to be divisible
    # by base_step
    #
    # examples:
    # n=31, base_step=32
    #   => bucket_size = ceil(31^1/3) * 32 = 4 * 32 = 128
    #   => next_value = round_up(31, 128) = 128
    # n=4001, base_step=32
    #   => bucket_size = ceil(4001^1/3) * 32 = 16 * 32 = 512
    #   => next_value = round_up(4001, 512) = 4096
    
    # Calculate bucket size
    power_result = math.pow(n, power)
    ceil_power = math.ceil(power_result)
    bucket_size = ceil_power * base_step
    result = math.ceil(n / bucket_size) * bucket_size
    
    # Log the calculation for debugging
    if base_step == 2:  # Only log for batch size calculations to avoid spam
        logger().debug(f"    calc_fallback_value({n}, {base_step}): "
                      f"{n}^{power:.3f} = {power_result:.3f} -> ceil = {ceil_power} -> "
                      f"bucket_size = {bucket_size} -> result = {result}")
    
    return result

class HPUBucketingManager():
    _instance = None
    prompt_buckets: List[Tuple[int, int, int]] = []
    decode_buckets: List[Tuple[int, int, int]] = []
    initialized = False

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(HPUBucketingManager, cls).__new__(cls)
        return cls._instance

    def initialize(self, max_num_seqs, max_num_prefill_seqs, block_size,
                   max_num_batched_tokens, max_model_len):
        logger().info(f"HPUBucketingManager.initialize() called with:")
        logger().info(f"  - max_num_seqs: {max_num_seqs}")
        logger().info(f"  - max_num_prefill_seqs: {max_num_prefill_seqs}")
        logger().info(f"  - block_size: {block_size}")
        logger().info(f"  - max_num_batched_tokens: {max_num_batched_tokens}")
        logger().info(f"  - max_model_len: {max_model_len}")
        
        self.max_num_seqs = max_num_seqs
        self.max_num_prefill_seqs = max_num_prefill_seqs
        self.block_size = block_size
        self.max_num_batched_tokens = max_num_batched_tokens
        self.num_hpu_blocks = None
        self.max_model_len = max_model_len
        self.initialized = True

        self.fallback_bs_base_step = 2
        self.fallback_seq_base_step = 32
        self.fallback_blocks_base_step = 32

        self.use_sliding_window = get_config().PT_HPU_SDPA_QKV_SLICE_MODE_FWD
        if self.use_sliding_window:
            self.slice_size = get_config().PT_HPU_SDPA_BC_FACTOR if \
                get_config().PT_HPU_SDPA_BC_FACTOR is not None else 1024
            self.slice_thld = get_config().VLLM_FUSEDSDPA_SLIDE_THLD if \
                get_config().VLLM_FUSEDSDPA_SLIDE_THLD is not None else 8192

    def get_bucketing_strategy(self):
        strategy = None
        # TODO - we can use different strategies for decode and prompt
        use_exponential_bucketing = True if \
                get_config().VLLM_EXPONENTIAL_BUCKETING == None else \
                get_config().VLLM_EXPONENTIAL_BUCKETING
        
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

            self.prompt_buckets = strategy.get_prompt_buckets(
                            max_num_prefill_seqs = self.max_num_prefill_seqs,
                            block_size = self.block_size,
                            max_num_batched_tokens = self.max_num_batched_tokens,
                            max_model_len = self.max_model_len)

            if self.use_sliding_window:
                self.prompt_buckets = [t for t in self.prompt_buckets
                    if t[1] < self.slice_thld or (
                    t[1] >= self.slice_thld and t[1] % self.slice_size == 0)]

            self.log_generate_info(True)
        else:
            logger().info("Bucketing is off - skipping prompt buckets generation")
            self.prompt_buckets = []
        return

    def generate_decode_buckets(self):
        if self.initialized:
            strategy = self.get_bucketing_strategy()

            self.decode_buckets = strategy.get_decode_buckets(
                            max_num_seqs = self.max_num_seqs,
                            block_size = self.block_size, 
                            max_num_batched_tokens = self.max_num_batched_tokens,
                            max_model_len = self.max_model_len, 
                            num_max_blocks = self.num_hpu_blocks)
            
            # Add extra bucket with full HPU blocks capacity if use_contiguous_pa is enabled
            if get_config().use_contiguous_pa and self.num_hpu_blocks is not None:
                logger().info(f"CONTIGUOUS PA DEBUG: use_contiguous_pa=True, num_hpu_blocks={self.num_hpu_blocks}")
                # Check if we already have a bucket with num_hpu_blocks
                has_max_blocks_bucket = any(bucket[2] == self.num_hpu_blocks for bucket in self.decode_buckets)
                if not has_max_blocks_bucket:
                    # Add bucket with maximum blocks capacity
                    max_bs = max(bucket[0] for bucket in self.decode_buckets) if self.decode_buckets else self.max_num_seqs
                    self.decode_buckets.append((max_bs, 1, self.num_hpu_blocks))
                    self.decode_buckets.sort()
                    logger().info(f"Added max capacity decode bucket: ({max_bs}, 1, {self.num_hpu_blocks})")
                else:
                    logger().info(f"Max capacity bucket with {self.num_hpu_blocks} blocks already exists")
            else:
                logger().info(f"CONTIGUOUS PA DEBUG: use_contiguous_pa={get_config().use_contiguous_pa}, num_hpu_blocks={self.num_hpu_blocks}")
            
            self.log_generate_info(False)
        else:
            logger().info("Bucketing is off - skipping decode buckets generation")
            self.decode_buckets = []
        return

    def log_generate_info(self, is_prompt):
        phase = 'prompt' if is_prompt else 'decode'
        buckets = self.prompt_buckets if is_prompt else self.decode_buckets
        msg = (f"Generated {len(buckets)} "
               f"{phase} buckets [bs, query, num_blocks]: "
               f"{list(buckets)}")
        logger().info(msg)

    def generate_fallback_bucket(self, batch_size, seq_len, ctx):
        assert self.max_num_batched_tokens is not None
        
        # Log the fallback calculation process
        logger().warning(f"FALLBACK CALCULATION:")
        logger().warning(f"  - Input: bs={batch_size}, seq_len={seq_len}, ctx={ctx}")
        
        original_batch_size = batch_size
        new_batch_size = calc_fallback_value(batch_size, self.fallback_bs_base_step)
        logger().warning(f"  - Batch size: {original_batch_size} -> {new_batch_size} (base_step={self.fallback_bs_base_step})")
        
        if self.use_sliding_window and seq_len >= self.slice_thld:
            new_seq_len = math.ceil(seq_len / self.slice_size) * self.slice_size
            logger().warning(f"  - Seq len (sliding window): {seq_len} -> {new_seq_len} (slice_size={self.slice_size})")
        else:
            fallback_seq = calc_fallback_value(seq_len, self.fallback_seq_base_step)
            new_seq_len = min(fallback_seq, self.max_num_batched_tokens)
            logger().warning(f"  - Seq len: {seq_len} -> {fallback_seq} -> {new_seq_len} (base_step={self.fallback_seq_base_step}, max={self.max_num_batched_tokens})")

        if self.num_hpu_blocks is None:
            new_ctx = 0
            logger().warning(f"  - Context: {ctx} -> 0 (num_hpu_blocks is None)")
        else:
            if ctx > self.num_hpu_blocks:
                logger().warning(f"  - WARNING: Requested {ctx} blocks exceeds available {self.num_hpu_blocks} blocks!")
                logger().warning(f"    This suggests a memory calculation issue upstream.")
            
            fallback_ctx = calc_fallback_value(ctx, self.fallback_blocks_base_step)
            new_ctx = min(fallback_ctx, self.num_hpu_blocks)
            logger().warning(f"  - Context: {ctx} -> {fallback_ctx} -> {new_ctx} (base_step={self.fallback_blocks_base_step}, max={self.num_hpu_blocks})")
        
        final_bucket = (new_batch_size, new_seq_len, new_ctx)
        logger().warning(f"  - Final bucket: {final_bucket}")
        return final_bucket

    def find_prompt_bucket(self, batch_size, seq_len, ctx=0, use_fallback=True):
        if self.initialized:
            found_bucket = find_equal_or_closest_greater_config(self.prompt_buckets, (batch_size, seq_len, ctx))
            if found_bucket is None:
                if use_fallback:
                    new_bucket = self.generate_fallback_bucket(batch_size, seq_len, ctx)
                    logger().warning(f"Prompt bucket for {batch_size, seq_len, ctx}"
                                     f" was not prepared. Adding new bucket: {new_bucket}")
                    self.prompt_buckets.append(new_bucket)
                    self.prompt_buckets.sort()
                    return new_bucket
                else:
                    return (None, None, None)
            return found_bucket
        return (batch_size, seq_len, ctx)

    def find_decode_bucket(self, batch_size, num_blocks):
        if self.initialized:
            # Always log bucket lookup attempts for debugging
            logger().info(f"BUCKET LOOKUP: Searching for decode bucket ({batch_size}, 1, {num_blocks})")
            logger().info(f"  - Available buckets: {len(self.decode_buckets)} total")
            logger().info(f"  - Bucket list: {self.decode_buckets}")
            
            found_bucket = find_equal_or_closest_greater_config(self.decode_buckets, (batch_size, 1, num_blocks))
            
            if found_bucket is None:
                # Log detailed information about why this bucket wasn't prepared
                max_available_blocks = max(bucket[2] for bucket in self.decode_buckets) if self.decode_buckets else 0
                logger().warning(f"BUCKET ANALYSIS: Decode bucket for ({batch_size}, 1, {num_blocks}) was not prepared.")
                logger().warning(f"  - Requested blocks: {num_blocks}")
                logger().warning(f"  - Max available in buckets: {max_available_blocks}")
                logger().warning(f"  - Total HPU blocks: {self.num_hpu_blocks}")
                logger().warning(f"  - Ratio requested/available: {num_blocks/self.num_hpu_blocks:.2f}x" if self.num_hpu_blocks else "N/A")
                
                # Add stack trace to understand where this request comes from
                import traceback
                logger().warning(f"  - Call stack:\n{traceback.format_stack()[-3]}")
                
                new_bucket = self.generate_fallback_bucket(batch_size, 1, num_blocks)
                logger().warning(f"  - Generated fallback bucket: {new_bucket}")
                self.decode_buckets.append(new_bucket)
                self.decode_buckets.sort()
                return new_bucket
            else:
                # Log successful bucket matches too
                logger().info(f"BUCKET MATCH: Found existing bucket {found_bucket} for request ({batch_size}, 1, {num_blocks})")
                return found_bucket
        return (batch_size, 1, num_blocks)

    def get_max_prompt_shape(self):
        return max(b[1] for b in self.prompt_buckets) \
               if len(self.prompt_buckets) > 0 else self.max_model_len

    @classmethod
    def get_instance(cls):
        """
        Retrieve the singleton instance of the class.
        """
        return cls._instance

    @property
    def num_hpu_blocks(self):
        return self._num_hpu_blocks
    
    @num_hpu_blocks.setter
    def num_hpu_blocks(self, value):
        if hasattr(self, '_num_hpu_blocks') and self._num_hpu_blocks != value:
            import traceback
            logger().info(f"HPU_BLOCKS_CHANGE: num_hpu_blocks changed from {self._num_hpu_blocks} to {value}")
            logger().info(f"  - Call stack:\n{''.join(traceback.format_stack()[-3:-1])}")
        elif not hasattr(self, '_num_hpu_blocks'):
            logger().info(f"HPU_BLOCKS_INIT: num_hpu_blocks set to {value}")
        self._num_hpu_blocks = value


def get_bucketing_manager():
    instance = HPUBucketingManager.get_instance()
    return instance 


def is_greater_or_equal(tuple1, tuple2):
    return tuple1[0] >= tuple2[0] and tuple1[1] >= tuple2[1] \
           and tuple1[2] >= tuple2[2]


def find_equal_or_closest_greater_config(sorted_list, target_tuple):
    idx = bisect.bisect_left(sorted_list, target_tuple)
    for i in range(idx, len(sorted_list)):
        if is_greater_or_equal(sorted_list[i], target_tuple):
            return sorted_list[i]
    return None

