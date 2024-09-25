###############################################################################
# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.
###############################################################################

import math
import torch
import torch.nn as nn

from vllm.logger import init_logger

logger = init_logger(__name__)

try:
    from habana_frameworks.torch.hpex.kernels import (
        RotaryPosEmbeddingHelperV1 as FusedRoPE)
except ImportError:
    logger.warning("Could not import HPU FusedRoPE kernel. "
                    "vLLM will use forward_native implementation of RoPE.")
    FusedRoPE = None

class HpuRotaryEmbedding(nn.Module):

    def __init__(self,
                 head_size,
                 rotary_dim,
                 max_position_embeddings=2048,
                 base=10000,
                 is_neox_style=None,
                 device='hpu',
                 RoPEFallback=None,
                 skip_fallback=False):
        super().__init__()

        self.head_size = head_size
        self.dim = rotary_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.device = device

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(seq_len=max_position_embeddings,
                                device=device,
                                dtype=torch.get_default_dtype())
        if FusedRoPE is None and not skip_fallback:
            assert RoPEFallback is not None, (
                "HPU FusedRoPE kernel could not be imported, and "
                "fallback RoPE implementation was not provided!")
            self.fallback_impl = RoPEFallback(head_size,
                                              rotary_dim,
                                              max_position_embeddings,
                                              base,
                                              is_neox_style,
                                              dtype=torch.get_default_dtype())

    def _compute_inv_freq(self) -> torch.Tensor:
        inv_freq = 1.0 / (self.base**(
            torch.arange(0, self.dim, 2).float().to(self.device) / self.dim))
        return inv_freq

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len

        inv_freq = self._compute_inv_freq()
        t = torch.arange(self.max_seq_len_cached,
                         device=device,
                         dtype=inv_freq.dtype)

        freqs = torch.einsum("i,j->ij", t, inv_freq)
        # Different from paper, but it uses a different permutation in order
        # to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached",
                             emb.cos().to(dtype),
                             persistent=False)
        self.register_buffer("sin_cached",
                             emb.sin().to(dtype),
                             persistent=False)

    def forward(self, positions: torch.Tensor, query: torch.Tensor,
                key: torch.Tensor):
            if FusedRoPE is None:
                return self.fallback_impl(positions, query, key)
            if query.dim() == 2:
                query = query.unsqueeze(0)
            if key.dim() == 2:
                key = key.unsqueeze(0)
            if positions.dim() == 1:
                positions = positions.unsqueeze(0)

            query = query.reshape(
                (query.shape[0], query.shape[1], query.shape[2] // self.head_size,
                self.head_size))
            key = key.reshape((key.shape[0], key.shape[1],
                            key.shape[2] // self.head_size, self.head_size))

            cos = self.cos_cached[positions].unsqueeze(2).to(dtype=query.dtype)
            sin = self.sin_cached[positions].unsqueeze(2).to(dtype=query.dtype)
            query = FusedRoPE.apply(query, cos, sin, 0)
            key = FusedRoPE.apply(key, cos, sin, 0)
            return query.reshape(
                (query.shape[0], query.shape[1],
                query.shape[2] * query.shape[3])), key.reshape(
                    (key.shape[0], key.shape[1], key.shape[2] * key.shape[3]))


class HpuLlama3RotaryEmbedding(HpuRotaryEmbedding):

    def __init__(self,
                 head_size: int,
                 rotary_dim: int,
                 max_position_embeddings: int,
                 base: int,
                 is_neox_style: bool,
                 scaling_factor: float,
                 low_freq_factor: float,
                 high_freq_factor: float,
                 orig_max_position: int,
                 device="hpu",
                 RoPEFallback=None):

        self.scaling_factor = scaling_factor
        self.low_freq_factor = low_freq_factor
        self.high_freq_factor = high_freq_factor
        self.orig_max_position = orig_max_position
        super().__init__(head_size, rotary_dim, max_position_embeddings,
                         base, is_neox_style, device, RoPEFallback, skip_fallback=True)

        if FusedRoPE is None:
            assert RoPEFallback is not None, (
                "HPU FusedRoPE kernel could not be imported, and "
                "fallback RoPE implementation was not provided!")
            self.fallback_impl = RoPEFallback(head_size,
                                              rotary_dim,
                                              max_position_embeddings,
                                              base,
                                              is_neox_style,
                                              torch.get_default_dtype(),
                                              scaling_factor,
                                              low_freq_factor,
                                              high_freq_factor,
                                              orig_max_position)

    def _compute_inv_freq(self) -> torch.Tensor:
        inv_freqs = super()._compute_inv_freq()
        low_freq_wavelen = self.orig_max_position / self.low_freq_factor
        high_freq_wavelen = self.orig_max_position / self.high_freq_factor

        wave_len = 2 * math.pi / inv_freqs
        if self.low_freq_factor != self.high_freq_factor:
            smooth = (self.orig_max_position / wave_len - self.low_freq_factor
                      ) / (self.high_freq_factor - self.low_freq_factor)
        else:
            smooth = 0
        new_freqs = torch.where(
            wave_len < high_freq_wavelen,
            inv_freqs,
            torch.where(
                wave_len > low_freq_wavelen,
                inv_freqs / self.scaling_factor,
                (1 - smooth) * inv_freqs / self.scaling_factor +
                smooth * inv_freqs,
            ),
        )
        return new_freqs