import itertools
import operator
import os
import copy
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


class Singleton(type):
    _instances: Dict[type, object] = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


@dataclass
class HPUBucketingGlobalState(metaclass=Singleton):
    prompt_bs_bucket_cfg: Tuple[int, int, int, float] = field(init=False)
    decode_bs_bucket_cfg: Tuple[int, int, int, float] = field(init=False)
    prompt_seq_bucket_cfg: Tuple[int, int, int, float] = field(init=False)
    decode_block_bucket_cfg: Tuple[int, int, int, float] = field(init=False)
    prompt_buckets: List[Tuple[int, int]] = field(init=False)
    decode_buckets: List[Tuple[int, int]] = field(init=False)


class HPUBucketingContext(metaclass=Singleton):
    global_state = HPUBucketingGlobalState()

    def __init__(self, max_num_seqs, max_num_prefill_seqs, block_size,
                 max_num_batched_tokens):
        self.max_num_seqs = max_num_seqs
        self.max_num_prefill_seqs = max_num_prefill_seqs
        self.block_size = block_size
        self.max_num_batched_tokens = max_num_batched_tokens
        self._setup_buckets()
        self.num_hpu_blocks = None

    def _setup_buckets(self) -> None:
        # FIXME: The default values should be max_model_len
        max_prompt_seq = 1024
        max_decode_seq = 2048
        max_blocks = max(
            self.block_size,
            self.max_num_seqs * max_decode_seq // self.block_size)

        self.global_state.prompt_bs_bucket_cfg = read_bucket_settings(
            'prompt', 'bs', min=1, step=32,
            max=self.max_num_prefill_seqs, limit=0.25)
        self.global_state.decode_bs_bucket_cfg = read_bucket_settings(
            'decode', 'bs', min=1, step=32,
            max=self.max_num_seqs, limit=0.25)
        self.global_state.prompt_seq_bucket_cfg = read_bucket_settings(
            'prompt', 'seq', min=self.block_size,
            step=self.block_size, max=max_prompt_seq, limit=0.25)
        self.global_state.decode_block_bucket_cfg = read_bucket_settings(
            'decode', 'block', min=self.block_size,
            step=self.block_size, max=max_blocks, limit=0.25)

        msg = ("Prompt bucket config (min, step, max_warmup) "
               f"bs:{self.global_state.prompt_bs_bucket_cfg}, "
               f"seq:{self.global_state.prompt_seq_bucket_cfg}")
        print(msg)

        msg = ("Decode bucket config (min, step, max_warmup) "
               f"bs:{self.global_state.decode_bs_bucket_cfg}, "
               f"block:{self.global_state.decode_block_bucket_cfg}")
        print(msg)

    def generate_prompt_buckets(self):
        self.global_state.prompt_buckets, prompt_omitted_buckets = \
            generate_prompt_buckets(
            self.global_state.prompt_bs_bucket_cfg,
            self.global_state.prompt_seq_bucket_cfg,
            self.max_num_batched_tokens)

        msg = (f"Generated {len(self.global_state.prompt_buckets)} "
               f"prompt buckets [bs, seq]: "
               f"{list(sorted(self.global_state.prompt_buckets))}")
        print(msg)

        msg = (f"Omitted {len(prompt_omitted_buckets)} "
               "prompt buckets due to exceeded token budget "
               f"(max_num_batched_tokens={self.max_num_batched_tokens})")
        print(msg)

        msg = f"Omitted prompt buckets: {list(sorted(prompt_omitted_buckets))}"
        print(msg)

    def generate_decode_buckets(self, max_blocks, num_speculative_tokens = 0):
        decode_bs_bucket_cfg = copy.deepcopy(self.global_state.decode_bs_bucket_cfg)
        decode_bs_bucket_cfg[2] = self.global_state.decode_bs_bucket_cfg[2] * (num_speculative_tokens + 1)
        self.global_state.decode_buckets = generate_decode_buckets(
            decode_bs_bucket_cfg,
            self.global_state.decode_block_bucket_cfg, max_blocks)
        print(f"Generated {len(self.global_state.decode_buckets)} "
              f"decode buckets [bs, total_blocks]: "
              f"{list(sorted(self.global_state.decode_buckets))}")

    def get_max_prompt_shape(self):
        return (self.global_state.prompt_bs_bucket_cfg[2],
                self.global_state.prompt_seq_bucket_cfg[2])

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

    @property
    def prompt_buckets(self):
        return self.global_state.prompt_buckets

    @property
    def decode_buckets(self):
        return self.global_state.decode_buckets


def read_bucket_settings(phase: str, dim: str, **defaults):
    """Read bucketing configuration from env variables.

    phase is either 'prompt' or 'decode'
    dim is either 'bs', 'seq' or 'block'
    param is either 'min', 'step' or 'max'
    example env variable: VLLM_DECODE_BS_BUCKET_STEP=128
    """
    params = ['min', 'step', 'max', 'limit']
    env_vars = [f'VLLM_{phase}_{dim}_BUCKET_{p}'.upper() for p in params]
    default_values = [defaults[p] for p in params]
    values = []
    for e, d in zip(env_vars, default_values):
        value = str(os.environ.get(e, d))
        if value.isdigit():
            value = int(value)
        else:
            value = float(value)
        values.append(value)
    for e, v, d in zip(env_vars, values, default_values):
        print(f'{e}={v} (default:{d})')
    return values


def warmup_range_with_limit(config: Tuple[int, int, int, float]):
    """Generate a warmup range.

    Start from bmin and multiply by 2 until you reach bstep.
    Then, increase the values in the range by the value of bstep until you
    reach bmax if the padding ratio is within the limit.

    Example:
    bmin = 2, bstep = 32, bmax = 64
    => ramp_up = (2, 4, 8, 16)
    => stable = (32, 64)
    => return ramp_up + stable => (2, 4, 8, 16, 32, 64)
    """
    bucket_min, bucket_step, bucket_max, limit = config
    assert bucket_min <= bucket_max, ("bucket_min cannot be greater than bucket_max. "
                          "If you want to skip warmup, set VLLM_SKIP_WARMUP=true")
    buckets = [bucket_min]
    current_bucket = bucket_min
    while current_bucket <= bucket_max:
        last_bucket = buckets[-1]
        if current_bucket <= bucket_step:
            next_bucket = last_bucket * 2
            if next_bucket <= bucket_max:
                buckets.append(next_bucket)
        else:
            next_bucket = current_bucket + bucket_step
            max_padding_ratio = 1 - (last_bucket / (next_bucket  - 1))
            if max_padding_ratio > limit and current_bucket != last_bucket:
                buckets.append(current_bucket)
        current_bucket = next_bucket
    if buckets[-1] != bucket_max:
        buckets.append(bucket_max)
    return buckets


def generate_prompt_buckets(bs_bucket_config,
                            seq_bucket_config,
                            max_num_batched_tokens=None):
    buckets = list(
        itertools.product(warmup_range_with_limit(bs_bucket_config),
                          warmup_range_with_limit(seq_bucket_config)))
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
                lambda bucket: bucket[0] * bucket[1] <= max_num_batched_tokens,
                buckets))

        if len(filtered_buckets) == 0:
            # we can handle this if we ignore max_num_batched_tokens
            min_bucket_bs, min_bucket_seq = min(buckets,
                                                key=lambda b: (b[0] * b[1]))
            min_reqd_budget = min_bucket_bs * min_bucket_seq
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
            print(msg)
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
    bs_buckets = warmup_range_with_limit(bs_bucket_config)
    if os.environ.get(
            'VLLM_DECODE_BLOCK_BUCKET_MAX') is None and os.environ.get(
                'VLLM_CONTIGUOUS_PA', 'true').lower() == 'true':
        blocks_bucket_config[2] = max_blocks
    block_buckets = warmup_range_with_limit(blocks_bucket_config)
    if os.environ.get('VLLM_CONTIGUOUS_PA',
                          'true').lower() == 'true' and os.environ.get(
                              'VLLM_DECODE_BLOCK_BUCKET_MAX'
                          ) is None and max_blocks not in block_buckets:
        block_buckets.append(max_blocks)
    last_bucket = max_blocks
    for bs in bs_buckets:
        for blocks in block_buckets:
            if bs > blocks:  # Skip invalid buckets
                continue
            if blocks >= last_bucket:
                buckets.append((bs, last_bucket))
                break
            buckets.append((bs, blocks))
    return list(sorted(buckets, key=lambda b: (b[0] * b[1], b[1], b[0])))


def next_pow2(value: int, base: int) -> int:
    res = base
    while value > res:
        res *= 2
    return res


def round_up(value: int, k: int) -> int:
    return (value + k - 1) // k * k


def find_bucket(value: int, config: Tuple[int, int, int, float]) -> int:
    buckets = warmup_range_with_limit(config)
    for b in buckets:
        if b >= value:
            return b
    return value

