###############################################################################
# Copyright (C) 2024-2025 Habana Labs, Ltd. an Intel Company
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.
###############################################################################

import habana_frameworks.torch as htorch
import torch


def insert_or_update_cache(input, cache, block_indices, flat_indices_with_offsets):
    if flat_indices_with_offsets is None:
        cache.index_copy_(0, block_indices, input)
    else:
        # Run index_copy on a 1D cache tensor
        # to avoid redundant memcopies due to small FCD
        num_blocks = cache.shape[0]
        block_size = cache.shape[1]
        cache_flat = cache.view(num_blocks * block_size, *cache.shape[2:])

        cache_flat.index_copy_(0, flat_indices_with_offsets, input)
        cache = cache_flat.view(num_blocks, block_size, *cache.shape[2:])

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
