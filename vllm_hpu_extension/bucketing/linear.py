import itertools
import logging
import operator
import os
from dataclasses import dataclass, field
from typing import List, Tuple

from vllm_hpu_extension.runtime import get_config

logger = logging.getLogger(__name__)


class LinearBucketingStrategy:
    def get_prompt_buckets(self, max_num_prefill_seqs, block_size, 
                           max_num_batched_tokens, max_prompt_seq, max_model_len, prefix_caching):
        default_max_prompt_seq = 1024
        if max_model_len is None and max_prompt_seq is None:
            logger.warning(f"max_model_len and max_prompt_seq are not set. Using default value max_prompt_seq={default_max_prompt_seq}. This may cause issues.")

        max_prompt_seq = next((item for item in [max_prompt_seq, max_model_len] if item is not None), default_max_prompt_seq)

        prompt_bs_bucket_cfg = read_bucket_settings(
            'prompt', 'bs', min=1, step=32,
            max=max_num_prefill_seqs)
        prompt_seq_bucket_cfg = read_bucket_settings(
            'prompt', 'seq', min=block_size,
            step=block_size, max=max_prompt_seq)

        if self.use_merged_prefill:
            prev_prompt_bs_bucket_cfg = tuple(prompt_bs_bucket_cfg)
            prev_prompt_seq_bucket_cfg = tuple(prompt_seq_bucket_cfg)
            seq_min, seq_step, seq_max = prev_prompt_seq_bucket_cfg
            max_bs = prompt_bs_bucket_cfg[2]
            prompt_bs_bucket_cfg = (1, 1, 1)
            prompt_seq_bucket_cfg = (seq_min, seq_step, min(max_bs * seq_max, max_num_batched_tokens))
            new_prompt_bs_bucket_cfg = prompt_bs_bucket_cfg
            new_prompt_seq_bucket_cfg = prompt_seq_bucket_cfg
            print('Merged prefill is enabled!\n'
                  'Overriding prompt bucketing settings!\n'
                  f'prompt bs cfg: {prev_prompt_bs_bucket_cfg} -> {new_prompt_bs_bucket_cfg}\n'
                  f'prompt seq cfg: {prev_prompt_seq_bucket_cfg} -> {new_prompt_seq_bucket_cfg}\n')

        msg = ("Prompt bucket config (min, step, max_warmup) "
               f"bs:{prompt_bs_bucket_cfg}, "
               f"seq:{prompt_seq_bucket_cfg}")
        logger.info(msg)

        prompt_buckets, prompt_omitted_buckets = \
            generate_prompt_buckets(
            prompt_bs_bucket_cfg,
            prompt_seq_bucket_cfg,
            block_size,
            prefix_caching,
            max_num_batched_tokens)

        msg = (f"Generated {len(prompt_buckets)} "
               f"prompt buckets [bs, seq]: "
               f"{list(sorted(prompt_buckets))}")
        logger.info(msg)

        msg = (f"Omitted {len(prompt_omitted_buckets)} "
               "prompt buckets due to exceeded token budget "
               f"(max_num_batched_tokens={max_num_batched_tokens})")
        logger.info(msg)
        
        return sorted(prompt_buckets), prompt_seq_bucket_cfg

    def get_decode_buckets(self, max_num_seqs, block_size, max_num_batched_tokens, max_decode_seq, max_model_len, num_max_blocks, prefix_caching):
        default_max_decode_seq = 2048
        if max_model_len is None and max_decode_seq is None:
            logger.warning(f"max_model_len and max_decode_seq are not set. Using default value max_decode_seq={default_max_decode_seq}. This may cause issues.")

        max_decode_seq = next((item for item in [max_decode_seq, max_model_len] if item is not None), default_max_decode_seq)
        max_blocks = max(
            block_size,
            max_num_seqs * max_decode_seq // block_size)

        decode_bs_bucket_cfg = read_bucket_settings(
            'decode', 'bs', min=1, step=32,
            max=max_num_seqs)
        decode_block_bucket_cfg = read_bucket_settings(
            'decode', 'block', min=block_size,
            step=block_size, max=max_blocks)

        msg = ("Decode bucket config (min, step, max_warmup) "
               f"bs:{decode_bs_bucket_cfg}, "
               f"block:{decode_block_bucket_cfg}")
        logger.info(msg)

        decode_buckets = generate_decode_buckets(
            decode_bs_bucket_cfg,
            decode_block_bucket_cfg, num_max_blocks)
        logger.info(f"Generated {len(decode_buckets)} "
              f"decode buckets [bs, total_blocks]: "
              f"{list(sorted(decode_buckets))}")

        return decode_buckets

    @classmethod
    def get_instance(cls):
        """
        Retrieve the singleton instance of the class.

        Returns:
            The singleton instance of the class.

        Raises:
            AssertionError: If the class has not been initialized and no instance exists.
        """
        assert cls in cls._instances, "Singleton instance not initialized"
        return type(cls)._instances[cls]

def read_bucket_settings(phase: str, dim: str, **defaults):
    """Read bucketing configuration from env variables.

    phase is either 'prompt' or 'decode'
    dim is either 'bs', 'seq' or 'block'
    param is either 'min', 'step' or 'max'
    example env variable: VLLM_DECODE_BS_BUCKET_STEP=128
    """
    params = ['min', 'step', 'max']
    env_vars = [f'VLLM_{phase}_{dim}_BUCKET_{p}'.upper() for p in params]
    default_values = [defaults[p] for p in params]
    values = [
        int(os.environ.get(e, d)) for e, d in zip(env_vars, default_values)
    ]
    for e, v, d in zip(env_vars, values, default_values):
        logger.info(f'{e}={v} (default:{d})')
    return values


def warmup_range(config: Tuple[int, int, int]):
    """Generate a warmup range.

    Start from bmin and multiply by 2 until you reach bstep.
    Then, increase the values in the range by the value of bstep until you 
    reach bmax.

    Example:
    bmin = 2, bstep = 32, bmax = 64
    => ramp_up = (2, 4, 8, 16)
    => stable = (32, 64)
    => return ramp_up + stable => (2, 4, 8, 16, 32, 64)
    """
    bmin, bstep, bmax = config
    assert bmin <= bmax, ("Min. batch size cannot be greater than max. "
                          "batch size. If you want to skip warmup, "
                          "set VLLM_SKIP_WARMUP=true")
    base = itertools.repeat(2)
    ramp_up_acc = itertools.accumulate(base, func=operator.mul, initial=bmin)
    ramp_up_tw = itertools.takewhile(lambda x: x < bstep and x <= bmax, \
        ramp_up_acc)
    stable = range(bstep, bmax + 1, bstep)
    buckets = list(ramp_up_tw) + list(stable)
    return list(filter(lambda bucket: bucket >= bmin, buckets))


def generate_prompt_buckets(bs_bucket_config,
                            seq_bucket_config,
                            block_size,
                            prefix_caching,
                            max_num_batched_tokens=None):
    _, _, bmax = seq_bucket_config
    batch_size_buckets = warmup_range(bs_bucket_config)
    seq_bucket_config = warmup_range(seq_bucket_config)

    if prefix_caching:
        buckets_3d = []
        for bs in batch_size_buckets:
            for b in seq_bucket_config:
                max_blocks_range = (bmax - b) // block_size
                for i in range(0, max_blocks_range + 1):
                    buckets_3d.append((bs, b, i))
        buckets = buckets_3d
    else:
        buckets = list(
                itertools.product(batch_size_buckets,
                                seq_bucket_config, [0]))

    if len(buckets) == 0:
        msg = ("No buckets could be captured with following config "
               f"(min, step, max_warmup): "
               f"bs:{bs_bucket_config}, "
               f"seq:{seq_bucket_config}")
        raise ValueError(msg)

    filtered_buckets = buckets
    if max_num_batched_tokens is not None:
        # Remove buckets exceeding batch token budget
        filtered_buckets = list(
            filter(
                lambda bucket: bucket[0] * (bucket[1] +  bucket[2] * block_size) <= max_num_batched_tokens,
                buckets))

        if len(filtered_buckets) == 0:
            # we can handle this if we ignore max_num_batched_tokens
            min_bucket_bs, min_bucket_seq, min_bucket_ctx = min(buckets,
                                                key=lambda b: (b[0] * b[1]))
            min_reqd_budget = min_bucket_bs * (min_bucket_seq + min_bucket_ctx * block_size)
            msg = (
                "The current bucketing configuration "
                f"(min, step, max_warmup): "
                f"bs:{bs_bucket_config}, "
                f"seq:{seq_bucket_config} cannot be used with specified "
                f"max_num_batched_tokens ({max_num_batched_tokens}), as the "
                f"smallest bucket ({min_reqd_budget}) would exceed token "
                "budget. Please increase max_num_batched_tokens or decrease "
                "bucket minimum. Ignoring max_num_batched_tokens at risk of "
                "out-of-memory errors.")
            logger.info(msg)
            return list(
                sorted(buckets, key=lambda b: (b[0] * b[1], b[1], b[0]))), []

    captured_buckets = list(
        sorted(filtered_buckets, key=lambda b: (b[0] * b[1], b[1], b[0])))

    omitted_buckets = list(
        sorted([x for x in buckets if x not in filtered_buckets]))
    return captured_buckets, omitted_buckets


def generate_decode_buckets(bs_bucket_config, blocks_bucket_config,
                            max_blocks):
    buckets = []
    bs_buckets = warmup_range(bs_bucket_config)
    use_contiguous_pa = get_config().use_contiguous_pa
    if os.environ.get('VLLM_DECODE_BLOCK_BUCKET_MAX') is None\
       and use_contiguous_pa:
        blocks_bucket_config[2] = max_blocks
    block_buckets = warmup_range(blocks_bucket_config)
    if os.environ.get('VLLM_DECODE_BLOCK_BUCKET_MAX') is None\
       and max_blocks not in block_buckets and use_contiguous_pa:
        block_buckets.append(max_blocks)
    last_bucket = max_blocks
    for bs in bs_buckets:
        for blocks in block_buckets:
            if bs > blocks:
                # Skip a dummy case when bs > blocks, which cannot occur in real execution
                continue
            if blocks >= last_bucket:
                buckets.append((bs, 1, last_bucket))
                break
            buckets.append((bs, 1, blocks))
    return list(sorted(buckets, key=lambda b: (b[0] * b[1], b[1], b[0])))
