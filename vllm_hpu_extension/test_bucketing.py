import pytest
import os

from vllm_hpu_extension.bucketing import (
    HPUBucketingContext,
    read_bucket_settings,
    warmup_range,
    generate_prompt_buckets,
    generate_decode_buckets,
    next_pow2,
    round_up,
    find_bucket,
)


@pytest.fixture
def hpu_bucketing_context():
    return HPUBucketingContext(
        max_num_seqs=128,
        max_num_prefill_seqs=16,
        block_size=128,
        max_num_batched_tokens=4096,
    )


def test_singleton():
    context1 = HPUBucketingContext(
        max_num_seqs=128,
        max_num_prefill_seqs=16,
        block_size=128,
        max_num_batched_tokens=4096,
    )
    context2 = HPUBucketingContext(
        max_num_seqs=128,
        max_num_prefill_seqs=16,
        block_size=128,
        max_num_batched_tokens=4096,
    )
    assert context1 is context2


def test_read_bucket_settings(monkeypatch):
    monkeypatch.setenv("VLLM_PROMPT_BS_BUCKET_MIN", "1")
    monkeypatch.setenv("VLLM_PROMPT_BS_BUCKET_STEP", "16")
    monkeypatch.setenv("VLLM_PROMPT_BS_BUCKET_MAX", "64")
    config = read_bucket_settings("prompt", "bs", min=1, step=32, max=128)
    assert config == [1, 16, 64]


def test_read_bucket_settings_empty_flags():
    config = read_bucket_settings("prompt", "bs", min=1, step=32, max=128)
    assert config == [1, 32, 128]


def test_warmup_range():
    config = (2, 64, 128)
    result = warmup_range(config)
    assert result == [2, 4, 8, 16, 32, 64, 128]


def test_generate_prompt_buckets():
    bs_bucket_config = (1, 4, 16)
    seq_bucket_config = (512, 512, 1024)
    max_num_batched_tokens = 2048
    buckets, omitted_buckets = generate_prompt_buckets(
        bs_bucket_config, seq_bucket_config, max_num_batched_tokens
    )
    assert len(buckets) == 5
    assert len(omitted_buckets) == 7
    assert all(bs * seq <= max_num_batched_tokens for bs, seq in buckets)


def test_generate_decode_buckets():
    bs_bucket_config = (1, 32, 128)
    blocks_bucket_config = (128, 128, 2048)
    max_blocks = 1024
    buckets = generate_decode_buckets(
        bs_bucket_config, blocks_bucket_config, max_blocks
    )
    assert len(buckets) == 72
    assert all(blocks <= max_blocks for _, blocks in buckets)


def test_next_pow2():
    assert next_pow2(15, 1) == 16
    assert next_pow2(16, 1) == 16
    assert next_pow2(17, 1) == 32
    assert next_pow2(129, 1) == 256


def test_round_up():
    assert round_up(5, 4) == 8
    assert round_up(12, 5) == 15


def test_find_bucket():
    config = (1, 32, 128)
    assert find_bucket(5, config) == 8
    assert find_bucket(9, config) == 16
    assert find_bucket(64, config) == 64
    assert find_bucket(65, config) == 96


def test_generate_prompt_buckets_method(hpu_bucketing_context):
    hpu_bucketing_context.generate_prompt_buckets()
    assert len(hpu_bucketing_context.prompt_buckets) > 0


def test_generate_decode_buckets_method(hpu_bucketing_context):
    hpu_bucketing_context.generate_decode_buckets(256)
    assert len(hpu_bucketing_context.decode_buckets) > 0


def test_get_max_prompt_shape(hpu_bucketing_context):
    max_shape = hpu_bucketing_context.get_max_prompt_shape()
    assert max_shape == (16, 1024)


def test_get_padded_prompt_batch_size(hpu_bucketing_context):
    padded_size = hpu_bucketing_context.get_padded_prompt_batch_size(5)
    assert padded_size == 8


def test_get_padded_decode_batch_size(hpu_bucketing_context):
    padded_size = hpu_bucketing_context.get_padded_decode_batch_size(5)
    assert padded_size == 8


def test_get_padded_prompt_seq_len(hpu_bucketing_context):
    padded_len = hpu_bucketing_context.get_padded_prompt_seq_len(100)
    assert padded_len == 128


def test_get_padded_decode_num_blocks(hpu_bucketing_context):
    padded_blocks = hpu_bucketing_context.get_padded_decode_num_blocks(100)
    assert padded_blocks == 128


def test_get_padded_batch_size(hpu_bucketing_context):
    padded_size = hpu_bucketing_context.get_padded_batch_size(5, is_prompt=True)
    assert padded_size == 8


def test_get_padded_seq_or_block(hpu_bucketing_context):
    padded_value = hpu_bucketing_context.get_padded_seq_or_block(100, is_prompt=True)
    assert padded_value == 128
