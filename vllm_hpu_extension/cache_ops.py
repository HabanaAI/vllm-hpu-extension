###############################################################################
# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.
###############################################################################

import math

import habana_frameworks.torch as htorch
import torch


def insert_or_update_cache(input, cache, block_indices, block_offsets):
    if block_offsets is None:
        cache.index_copy_(0, block_indices, input)
    else:
        cache.index_put_((block_indices, block_offsets), input)

def insert_or_update_cache_chunked(input, cache, block_indices, block_offsets):
    if block_offsets is None:
        cache.index_copy_(0, block_indices, input)
    else:
        if block_offsets.numel() == block_indices.numel():
            cache.index_put_((block_indices, block_offsets), input)
        else:
            offsets = None
            block_size = cache.shape[1]
            for i in range(block_indices.shape[0]):
                offsets = block_offsets[i * block_size:(i + 1) * block_size - 1]
                offset_indices = (offsets == -1).nonzero(as_tuple=True)
                start_index = offsets[0].item()
                end_index = offsets[offset_indices[0][0].item() - 1].item() + 1
                print(input[i][start_index:offset_indices[0][0]].shape)
                cache[block_indices[i], start_index:end_index] = input[i][:offset_indices[0][0]]

def swap_blocks(src, dst, block_mapping):
    if block_mapping.numel() == 0:
        return

    block_mapping = block_mapping.transpose(0, 1)
    src_indices = block_mapping[0]
    dst_indices = block_mapping[1]

    dst.index_put_(dst_indices, src.index_select(0, src_indices))

    htorch.core.mark_step()
    torch.hpu.synchronize()


def copy_blocks(key_caches, value_caches, block_mapping):
    if block_mapping.numel() == 0:
        return

    block_mapping = block_mapping.transpose(0, 1)
    src = block_mapping[0]
    dst = block_mapping[1]

    for key_cache, value_cache in zip(key_caches, value_caches):
        key_cache.index_copy_(0, dst, key_cache.index_select(0, src))
        value_cache.index_copy_(0, dst, value_cache.index_select(0, src))

    if key_caches[0].device.type == 'hpu':
        htorch.core.mark_step()
