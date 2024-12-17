import pytest
import os

from vllm_hpu_extension.bucketing.exponential import (
    HPUExponentialBucketingContext,
    read_bucket_settings,
    warmup_range_with_limit,
    generate_prompt_buckets,
    generate_decode_buckets,
)


@pytest.fixture
def hpu_bucketing_context():
    ctx = HPUExponentialBucketingContext(
        max_num_seqs=128,
        max_num_prefill_seqs=16,
        block_size=128,
        max_num_batched_tokens=4096,
    )
    ctx.num_hpu_blocks = 1024
    return ctx


def test_singleton():
    context1 = HPUExponentialBucketingContext(
        max_num_seqs=128,
        max_num_prefill_seqs=16,
        block_size=128,
        max_num_batched_tokens=4096,
    )
    context2 = HPUExponentialBucketingContext(
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
    monkeypatch.setenv("VLLM_PROMPT_BS_BUCKET_LIMIT", "10")
    config = read_bucket_settings("prompt", "bs", min=1, step=32, max=128, limit=10)
    assert config == [1, 16, 64, 10]


def test_read_bucket_settings_empty_flags():
    config = read_bucket_settings("prompt", "bs", min=1, step=32, max=128, limit=10)
    assert config == [1, 32, 128, 10]


def test_warmup_range():
    config = (2, 1, 128, 10)
    result = warmup_range_with_limit(config)
    assert result == [2, 4, 6, 8, 13, 21, 32, 51, 81, 128]


def test_generate_prompt_buckets():
    bs_bucket_config = (1, 1, 16, 3)
    seq_bucket_config = (512, 1, 1024, 3)
    max_num_batched_tokens = 2048
    buckets, omitted_buckets = generate_prompt_buckets(
        bs_bucket_config, seq_bucket_config, max_num_batched_tokens
    )
    assert len(buckets) == 4
    assert len(omitted_buckets) == 5 
    assert all(bs * seq <= max_num_batched_tokens for bs, seq in buckets)


def test_generate_decode_buckets():
    bs_bucket_config = (1, 1, 128, 8)
    blocks_bucket_config = (128, 1, 2048, 4)
    max_blocks = 1024
    buckets = generate_decode_buckets(
        bs_bucket_config, blocks_bucket_config, max_blocks
    )
    assert len(buckets) == 8*4
    assert all(blocks <= max_blocks for _, blocks in buckets)


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
    assert padded_size == 5


def test_get_padded_decode_batch_size(hpu_bucketing_context):
    padded_size = hpu_bucketing_context.get_padded_decode_batch_size(5)
    assert padded_size == 6


def test_get_padded_prompt_seq_len(hpu_bucketing_context):
    padded_len = hpu_bucketing_context.get_padded_prompt_seq_len(100)
    assert padded_len == 128


def test_get_padded_decode_num_blocks(hpu_bucketing_context):
    padded_blocks = hpu_bucketing_context.get_padded_decode_num_blocks(100)
    assert padded_blocks == 128


def test_get_padded_batch_size(hpu_bucketing_context):
    padded_size = hpu_bucketing_context.get_padded_batch_size(5, is_prompt=True)
    assert padded_size == 5


def test_get_padded_seq_or_block(hpu_bucketing_context):
    padded_value = hpu_bucketing_context.get_padded_seq_or_block(100, is_prompt=True)
    assert padded_value == 128
