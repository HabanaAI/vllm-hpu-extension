###############################################################################
# Copyright (C) 2024-2025 Habana Labs, Ltd. an Intel Company
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.
###############################################################################

import os
from functools import lru_cache, wraps

import habana_frameworks.torch as htorch
import torch
from habana_frameworks.torch.hpex.kernels.FusedSDPA import (
    gqa_input_reshape_fwd, gqa_output_reshape, is_gqa)

from vllm_hpu_extension.runtime import get_config


@lru_cache(maxsize=None)
def is_fake_hpu() -> bool:
    return os.environ.get('VLLM_USE_FAKE_HPU', '0') != '0'


def with_mark_steps(fn):

    @wraps(fn)
    def wrapped(*args, **kwargs):
        htorch.core.mark_step()
        result = fn(*args, **kwargs)
        del args
        del kwargs
        htorch.core.mark_step()
        return result

    return wrapped


class Matmul(torch.nn.Module):

    def __init__(self):
        super(Matmul, self).__init__()

    def forward(self, x, y, **kwargs):
        return torch.matmul(x, y, **kwargs)


class Softmax(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x, dim=None, inv_head=None):
        return torch.softmax(x, dim)


class VLLMKVCache(torch.nn.Module):

    def __init__(self):
        super(VLLMKVCache, self).__init__()
        self.use_contiguous_pa = get_config().use_contiguous_pa

    def forward(self, input, cache, slot_mapping):
        # In cross-attention kv cache forward inputs are None in decode
        # We don't want to store them in the cache in such case
        if input is not None:
            cache.index_copy_(0, slot_mapping, input)
        return cache

    def fetch_from_cache(self, cache, blocks):
        if self.use_contiguous_pa:
            return cache[:blocks.size(0)]
        else:
            return cache.index_select(0, blocks)


class VLLMFP8KVCache(VLLMKVCache):

    def __init__(self, input_scale=1.0):
        super(VLLMKVCache, self).__init__()
        self.use_contiguous_pa = get_config().use_contiguous_pa
        self.input_scale = input_scale
        self.output_scale = 1.0 / self.input_scale

    def quant_input(self, input):
        return torch.ops.hpu.cast_to_fp8_v2(input, self.input_scale, False, False, torch.float8_e4m3fn)[0]
    
    def dequant_output(self, output):
        return torch.ops.hpu.cast_from_fp8(output, self.output_scale, torch.bfloat16)

    def forward(self, input, *args, **kwargs):
        qinput = self.quant_input(input)
        return super().forward(qinput, *args, **kwargs)

    def fetch_from_cache(self, quant_cache, blocks, permutations=None):
        if permutations:
            output_cache = super().fetch_from_cache(quant_cache, blocks,
                                                        permutations)
            for i in range(len(output_cache)):
                output_cache[i] = self.dequant_output(output_cache[i])
            return output_cache
        output_cache = super().fetch_from_cache(quant_cache, blocks)
        return self.dequant_output(output_cache)


class FP8Matmul(torch.nn.Module):

    def __init__(self, scale_input=1.0, scale_other=1.0,):
        super().__init__()
        self.scale_input = scale_input
        self.scale_other = scale_other

    def quant_input(self, x, scale):
        return torch.ops.hpu.cast_to_fp8_v2(
            x, scale, False, False, torch.float8_e4m3fn
        )[0]

    def matmul_fp8(
        self, x, other, out_dtype, scale_input_inv=None, scale_other_inv=None
    ):
        return torch.ops.hpu.fp8_gemm_v2(
            A=x,
            trans_A=False,
            B=other,
            trans_B=False,
            D=None,
            out_dtype=out_dtype,
            A_scale_inv=scale_input_inv,
            B_scale_inv=scale_other_inv,
            bias=None,
            accumulate=False,
        )

    def forward(self, input, other):
        qinput = self.quant_input(input, self.scale_input)
        qother = self.quant_input(other, self.scale_other)
        output = self.matmul_fp8(
            qinput,
            qother,
            out_dtype=torch.bfloat16,
            scale_input_inv=1.0 / self.scale_input,
            scale_other_inv=1.0 / self.scale_other,
        )
        return output


class ModuleFusedSDPA(torch.nn.Module):
    def __init__(self, fusedSDPA):
        super().__init__()
        assert fusedSDPA is not None, f'fusedSDPA kernel is None'
        self._hpu_kernel_fsdpa = fusedSDPA

        impl_mapping = {
            'split_kv': SplitPrefixCausalSDPA,
            'slice_causal': CausalSliceSDPA,
            'slice_qkv': QKVSliceSDPA,
        }

        sdpa_slice_impl_option = get_config().VLLM_HPU_FSDPA_SLICE_IMPL
        sdpa_slice_impl_option = sdpa_slice_impl_option if sdpa_slice_impl_option is not None else 'slice_qkv'
        assert sdpa_slice_impl_option in impl_mapping, f'Unsupported sdpa slice impl: {sdpa_slice_impl_option}'

        self.sdpa_slice_impl = impl_mapping[sdpa_slice_impl_option]
        self.slice_causal = get_config().VLLM_HPU_FSDPA_SLICE_CAUSAL

        qkv_slice_thld = get_config().VLLM_HPU_FSDPA_SLICE_SEQ_LEN_THLD
        self.qkv_slice_thld = qkv_slice_thld if qkv_slice_thld is not None else 0

        q_chunk_size = get_config().VLLM_HPU_FSDPA_SLICE_CHUNK_SIZE
        self.q_chunk_size = q_chunk_size if q_chunk_size is not None else 4096

        kv_chunk_size = get_config().VLLM_HPU_FSDPA_SLICE_CHUNK_SIZE
        self.kv_chunk_size = kv_chunk_size if kv_chunk_size is not None else 4096

        causal_chunk_size = get_config().VLLM_HPU_FSDPA_SLICE_CHUNK_SIZE
        self.causal_chunk_size = causal_chunk_size if causal_chunk_size is not None else 4096

        self.with_mark_step = os.getenv("VLLM_HPU_FSDPA_SLICE_WITH_MARK_STEP", "0") in ("1", "true")

    def forward(
        self,
        query,
        key,
        value,
        attn_mask,
        dropout_p,
        is_causal,
        scale,
        softmax_mode,
        recompute_mode,
        valid_sequence_lengths,
        padding_side="left",
        window_size=None,
    ):

        bs = query.size(0)
        query_len = query.size(-2)
        kv_len = key.size(-2)

        # prefill with prefix caching
        if (bs == 1 and self.qkv_slice_thld > 0 and query_len != 1
                and query_len != kv_len and kv_len >= self.qkv_slice_thld):
            prefix_len = kv_len - query_len

            key_prefix = key[..., :prefix_len, :]
            value_prefix = value[..., :prefix_len, :]

            key_causal = key[..., prefix_len:, :]
            value_causal = value[..., prefix_len:, :]

            assert attn_mask is not None, "attn_mask should not be None for prefix caching prefill"
            causal_attn_mask = attn_mask[..., prefix_len:]

            attn_weights = self.sdpa_slice_impl.apply(
                query,
                key_prefix,
                value_prefix,
                key_causal,
                value_causal,
                causal_attn_mask,
                self.q_chunk_size,
                self.kv_chunk_size,
                self.causal_chunk_size,
                scale,
                softmax_mode,
                None,  #valid_seq_len
                padding_side,
                self.with_mark_step,
            )

            return attn_weights

        elif (is_causal and window_size is None and self.slice_causal
              and self.qkv_slice_thld > 0
              and query_len >= self.qkv_slice_thld):
            attn_results = QKVSliceCausalSDPA.apply(
                query,
                key,
                value,
                self.causal_chunk_size,
                attn_mask,
                scale,
                softmax_mode,
                None,  #valid_seq_len
                padding_side,
                True,  # output_original_dtype
                self.with_mark_step,
            )
            attn_weights = attn_results[0]

            return attn_weights

        else:
            if window_size:
                return self._hpu_kernel_fsdpa.apply(
                    query,
                    key,
                    value,
                    attn_mask,
                    dropout_p,
                    is_causal,
                    scale,
                    softmax_mode,
                    recompute_mode,
                    valid_sequence_lengths,
                    padding_side,
                    False,
                    False,
                    window_size)
            else:
                return self._hpu_kernel_fsdpa.apply(
                query,
                key,
                value,
                attn_mask,
                dropout_p,
                is_causal,
                scale,
                softmax_mode,
                recompute_mode,
                valid_sequence_lengths,
                padding_side,
            )


class QKVSliceSDPA(torch.autograd.Function):
    '''
    APC: split Prefix and Causal SDPA and do QKV slice on both parts.
    '''

    @staticmethod
    def forward(ctx,
                query,
                key_prefix,
                value_prefix,
                key,
                value,
                causal_attn_mask,
                q_chunk_size,
                kv_chunk_size,
                causal_chunk_size,
                scale,
                softmax_mode,
                valid_seq_len=None,
                padding_mode="right",
                with_mark_step=False,
            ):

        prefix_out, prefix_m, prefix_linv = QKVSlicePrefixSDPA.apply(
            query,
            key_prefix,
            value_prefix,
            q_chunk_size,
            kv_chunk_size,
            scale,
            softmax_mode,
            valid_seq_len,
            padding_mode,
            False,  #output_original_dtype
            with_mark_step,
        )

        text_out, text_m, text_linv = QKVSliceCausalSDPA.apply(
            query,
            key,
            value,
            causal_chunk_size,
            causal_attn_mask,
            scale,
            softmax_mode,
            valid_seq_len,
            padding_mode,
            False,  #output_original_dtype
            with_mark_step,
        )

        new_m = torch.maximum(prefix_m, text_m)
        prefix_linv_rescaled = (1.0 / prefix_linv) * torch.exp(prefix_m -
                                                               new_m)
        text_linv_rescaled = (1.0 / text_linv) * torch.exp(text_m - new_m)
        new_linv = 1.0 / (prefix_linv_rescaled + text_linv_rescaled)
        output = (prefix_linv_rescaled * new_linv) * prefix_out + (
            text_linv_rescaled * new_linv) * text_out
        return output.to(query.dtype)


class SplitPrefixCausalSDPA(torch.autograd.Function):
    '''
    APC: split Prefix and Causal SDPA parts.
    '''

    @staticmethod
    def forward(ctx,
                query,
                key_prefix,
                value_prefix,
                key,
                value,
                causal_attn_mask,
                q_chunk_size,
                kv_chunk_size,
                causal_chunk_size,
                scale,
                softmax_mode,
                valid_seq_len=None,
                padding_mode="right",
                with_mark_step=False):

        if with_mark_step:
            htorch.core.mark_step()
        prefix_out, prefix_m, prefix_linv = PrefixSDPA.apply(
            query,
            key_prefix,
            value_prefix,
            scale,
            softmax_mode,
            valid_seq_len,
            padding_mode,
            False,  #output_original_dtype
        )
        
        if with_mark_step:
            htorch.core.mark_step()

        text_out, text_m, text_linv = CausalSDPA.apply(
            query,
            key,
            value,
            causal_attn_mask,
            scale,
            softmax_mode,
            valid_seq_len,
            padding_mode,
            False,  #output_original_dtype
        )

        new_m = torch.maximum(prefix_m, text_m)
        prefix_linv_rescaled = (1.0 / prefix_linv) * torch.exp(prefix_m -
                                                               new_m)
        text_linv_rescaled = (1.0 / text_linv) * torch.exp(text_m - new_m)
        new_linv = 1.0 / (prefix_linv_rescaled + text_linv_rescaled)
        output = (prefix_linv_rescaled * new_linv) * prefix_out + (
            text_linv_rescaled * new_linv) * text_out
        
        if with_mark_step:
            htorch.core.mark_step()

        return output.to(query.dtype)


class CausalSliceSDPA(torch.autograd.Function):
    '''
    APC: split Prefix and Causal SDPA and do QKV slice on Causal part only.
    '''

    @staticmethod
    def forward(ctx,
                query,
                key_prefix,
                value_prefix,
                key,
                value,
                causal_attn_mask,
                q_chunk_size,
                kv_chunk_size,
                causal_chunk_size,
                scale,
                softmax_mode,
                valid_seq_len=None,
                padding_mode="right",
                with_mark_step=False):

        if with_mark_step:
            htorch.core.mark_step()
        prefix_out, prefix_m, prefix_linv = PrefixSDPA.apply(
            query,
            key_prefix,
            value_prefix,
            scale,
            softmax_mode,
            valid_seq_len,
            padding_mode,
            False,  #output_original_dtype
        )

        if with_mark_step:
            htorch.core.mark_step()

        text_out, text_m, text_linv = QKVSliceCausalSDPA.apply(
            query,
            key,
            value,
            causal_chunk_size,
            causal_attn_mask,
            scale,
            softmax_mode,
            valid_seq_len,
            padding_mode,
            False,  #output_original_dtype
            with_mark_step,
        )

        new_m = torch.maximum(prefix_m, text_m)
        prefix_linv_rescaled = (1.0 / prefix_linv) * torch.exp(prefix_m -
                                                               new_m)
        text_linv_rescaled = (1.0 / text_linv) * torch.exp(text_m - new_m)
        new_linv = 1.0 / (prefix_linv_rescaled + text_linv_rescaled)
        output = (prefix_linv_rescaled * new_linv) * prefix_out + (
            text_linv_rescaled * new_linv) * text_out
        
        if with_mark_step:
            htorch.core.mark_step()

        return output.to(query.dtype)


class PrefixSDPA(torch.autograd.Function):
    '''
    APC: Prefix part.
    '''

    @staticmethod
    def forward(ctx,
                query,
                key_prefix,
                value_prefix,
                scale,
                softmax_mode,
                valid_seq_len=None,
                padding_mode="right",
                output_original_dtype=False):

        gqa = is_gqa(query, key_prefix)
        if gqa:
            query, key_prefix, value_prefix, _ = gqa_input_reshape_fwd(
                query, key_prefix, value_prefix, None)

        result = torch.ops.hpu.sdpa_recomp_fwd(
            query,
            key_prefix,
            value_prefix,
            None,  #attn_mask
            0.0,  #dropout
            scale,
            False,  #is_causal
            True,  #requires_backward
            softmax_mode,
            valid_seq_len,
            padding_mode)

        out, m, linv = (gqa_output_reshape(x) if gqa else x
                        for x in result[:3])

        if output_original_dtype:
            out = out.to(query.dtype)
        else:
            out = out.to(torch.float32)
        m = m.to(torch.float32)
        if softmax_mode == "fast":
            linv = linv.to(torch.float32) * 128.0
        else:
            linv = linv.to(torch.float32)

        return out, m, linv


class QKVSlicePrefixSDPA(torch.autograd.Function):
    '''
    APC: Prefix part with QKV slice.
    '''

    def forward(self,
                query,
                key_prefix,
                value_prefix,
                q_chunk_size,
                kv_chunk_size,
                scale,
                softmax_mode,
                valid_seq_len=None,
                padding_mode="right",
                output_original_dtype=False,
                with_mark_step=False,
            ):

        gqa = is_gqa(query, key_prefix)
        if gqa:
            query, key_prefix, value_prefix, _ = gqa_input_reshape_fwd(
                query, key_prefix, value_prefix, None)

        query_len = query.size(-2)
        num_query_chunk = (query_len + q_chunk_size - 1) // q_chunk_size

        key_len = key_prefix.size(-2)
        num_kv_chunk = (key_len + kv_chunk_size - 1) // kv_chunk_size

        final_hidden_list = []
        final_m_list = []
        final_linv_list = []

        for query_idx in range(num_query_chunk):

            query_start = query_idx * q_chunk_size
            query_end = min((query_idx + 1) * q_chunk_size, query_len)

            query_slice = query[..., query_start:query_end, :]

            out = None
            m = None
            linv = None

            for kv_idx in range(num_kv_chunk):

                kv_start = kv_idx * kv_chunk_size
                kv_end = min((kv_idx + 1) * kv_chunk_size, key_len)

                key_slice = key_prefix[..., kv_start:kv_end, :]
                value_slice = value_prefix[..., kv_start:kv_end, :]

                if with_mark_step:
                    query_slice = query_slice.clone()
                    key_slice = key_slice.clone()
                    value_slice = value_slice.clone()
                    htorch.core.mark_step()
                block_result = torch.ops.hpu.sdpa_recomp_fwd(
                    query_slice,
                    key_slice,
                    value_slice,
                    None,  #attn_mask
                    0.0,  #dropout
                    scale,
                    False,  #is_causal
                    True,  #requires_backward
                    softmax_mode,
                    valid_seq_len,
                    padding_mode)

                block_out, block_m, block_linv = (gqa_output_reshape(x)
                                                  if gqa else x
                                                  for x in block_result[:3])
                block_out = block_out.to(torch.float32)
                block_m = block_m.to(torch.float32)
                if softmax_mode == "fast":
                    block_linv = block_linv.to(torch.float32) * 128.0
                else:
                    block_linv = block_linv.to(torch.float32)

                if kv_idx == 0:
                    out = block_out
                    m = block_m
                    linv = block_linv
                else:
                    new_m = torch.maximum(m, block_m)
                    l_rescaled = (1.0 / linv) * torch.exp(m - new_m)
                    block_l_rescaled = (1.0 / block_linv) * torch.exp(block_m -
                                                                      new_m)
                    new_linv = 1.0 / (l_rescaled + block_l_rescaled)
                    out = (l_rescaled * new_linv) * out + (
                        block_l_rescaled * new_linv) * block_out
                    linv = new_linv
                    m = new_m
                
                if with_mark_step:
                    htorch.core.mark_step()

            if output_original_dtype:
                final_hidden_list.append(out.to(query.dtype))
            else:
                final_hidden_list.append(out)
            final_linv_list.append(linv)
            final_m_list.append(m)

        output = torch.cat(final_hidden_list, dim=-2)
        final_linv = torch.cat(final_linv_list, dim=-2)
        final_m = torch.cat(final_m_list, dim=-2)

        return output, final_m, final_linv


class CausalSDPA(torch.autograd.Function):
    '''
    APC: Causal part.
    '''

    @staticmethod
    def forward(ctx,
                query,
                key,
                value,
                attn_mask,
                scale,
                softmax_mode,
                valid_seq_len=None,
                padding_mode="right",
                output_original_dtype=False):

        query_len = query.size(-2)
        kv_len = key.size(-2)

        assert query_len == kv_len, f"For causal SDPA, {query_len=} should be equal to {kv_len=}"

        gqa = is_gqa(query, key)
        if gqa:
            query, key, value, attn_mask = gqa_input_reshape_fwd(
                query, key, value, attn_mask)

        # Kernel limitation, need to use not causal and pass mask to get correct m and linv
        if query_len % 1024 != 0:
            is_causal = False
            if attn_mask is None:
                bs = query.size(0)
                mask_shape = (bs, 1, 1, query_len,
                              query_len) if gqa else (bs, 1, query_len,
                                                      query_len)
                mask = (1 - torch.tril(
                    torch.ones(
                        mask_shape, dtype=query.dtype,
                        device=query.device))) * torch.finfo(query.dtype).min
            else:
                mask = attn_mask
        else:
            is_causal = True
            mask = None

        result = torch.ops.hpu.sdpa_recomp_fwd(
            query,
            key,
            value,
            mask,  #attn_mask
            0.0,  #dropout
            scale,
            is_causal,  #is_causal
            True,  #requires_backward
            softmax_mode,
            valid_seq_len,
            padding_mode)

        out, m, linv = (gqa_output_reshape(x) if gqa else x
                        for x in result[:3])

        if output_original_dtype:
            out = out.to(query.dtype)
        else:
            out = out.to(torch.float32)
        m = m.to(torch.float32)
        if softmax_mode == "fast":
            linv = linv.to(torch.float32) * 128.0
        else:
            linv = linv.to(torch.float32)

        return out, m, linv


class QKVSliceCausalSDPA(torch.autograd.Function):
    '''
    APC: Causal part with QKV slice.
    '''

    def forward(self,
                query,
                key,
                value,
                chunk_size,
                attn_mask,
                scale,
                softmax_mode,
                valid_seq_len=None,
                padding_mode="right",
                output_original_dtype=False,
                with_mark_step=False,
            ):

        query_len = query.size(-2)
        kv_len = key.size(-2)

        assert query_len == kv_len, f"For QKV slice causal SDPA, {query_len=} should be equal to {kv_len=}"

        gqa = is_gqa(query, key)
        if gqa:
            query, key, value, attn_mask = gqa_input_reshape_fwd(
                query, key, value, attn_mask)

        num_chunk = (query_len + chunk_size - 1) // chunk_size

        final_hidden_list = []
        final_m_list = []
        final_linv_list = []

        for query_idx in range(num_chunk):

            query_start = query_idx * chunk_size
            query_end = min((query_idx + 1) * chunk_size, query_len)

            # step 1: compute the diagonal casual blocks
            query_slice = query[..., query_start:query_end, :]
            key_slice = key[..., query_start:query_end, :]
            value_slice = value[..., query_start:query_end, :]

            bs = query_slice.size(0)
            q_slice_len = query_slice.size(-2)

            if with_mark_step:
                query_slice = query_slice.clone()
                key_slice = key_slice.clone()
                value_slice = value_slice.clone()
                htorch.core.mark_step()
            
            # Kernel limitation, need to use not causal and pass mask to get correct m and linv
            if q_slice_len % 1024 != 0 or q_slice_len < chunk_size:
                if attn_mask is not None:
                    mask = attn_mask[..., query_start:query_end,
                                     query_start:query_end]
                    if with_mark_step:
                        mask = mask.clone()
                        htorch.core.mark_step()
                else:
                    mask_shape = (bs, 1, 1, q_slice_len,
                                  q_slice_len) if gqa else (bs, 1, q_slice_len,
                                                            q_slice_len)
                    mask = (1 - torch.tril(
                        torch.ones(mask_shape,
                                   dtype=query.dtype,
                                   device=query.device))) * torch.finfo(
                                       query.dtype).min

                result = torch.ops.hpu.sdpa_recomp_fwd(
                    query_slice,
                    key_slice,
                    value_slice,
                    mask,
                    0.0,  #dropout
                    scale,
                    False,  #is_causal
                    True,  #requires_backward
                    softmax_mode,
                    None,  #valid_seq_len
                    padding_mode)
            else:
                # causal
                result = torch.ops.hpu.sdpa_recomp_fwd(
                    query_slice,
                    key_slice,
                    value_slice,
                    None,  #mask
                    0.0,  #dropout
                    scale,
                    True,  #is_causal
                    True,  #requires_backward
                    softmax_mode,
                    None,  #valid_seq_len
                    padding_mode)

            out, m, linv = (gqa_output_reshape(x) if gqa else x
                            for x in result[:3])

            out = out.to(torch.float32)
            m = m.to(torch.float32)
            if softmax_mode == "fast":
                linv = linv.to(torch.float32) * 128.0
            else:
                linv = linv.to(torch.float32)

            if num_chunk == 1:
                if output_original_dtype:
                    out = out.to(query.dtype)
                return out, m, linv

            # step 2: compute the full attn blocks
            for kv_idx in range(0, query_idx):
                kv_start = kv_idx * chunk_size
                kv_end = (kv_idx + 1) * chunk_size

                key_slice = key[..., kv_start:kv_end, :]
                value_slice = value[..., kv_start:kv_end, :]

                if with_mark_step:
                    key_slice = key_slice.clone()
                    value_slice = value_slice.clone()
                    htorch.core.mark_step()

                block_result = torch.ops.hpu.sdpa_recomp_fwd(
                    query_slice,
                    key_slice,
                    value_slice,
                    None,  #mask
                    0.0,  #dropout
                    scale,
                    False,  #is_causal
                    True,  #requires_backward
                    softmax_mode,
                    None,  #valid_seq_len
                    padding_mode)

                block_out, block_m, block_linv = (gqa_output_reshape(x)
                                                  if gqa else x
                                                  for x in block_result[:3])

                block_out = block_out.to(torch.float32)
                block_m = block_m.to(torch.float32)
                if softmax_mode == "fast":
                    block_linv = block_linv.to(torch.float32) * 128.0
                else:
                    block_linv = block_linv.to(torch.float32)

                new_m = torch.maximum(m, block_m)
                l_rescaled = (1.0 / linv) * torch.exp(m - new_m)
                block_l_rescaled = (1.0 / block_linv) * torch.exp(block_m -
                                                                  new_m)
                new_linv = 1.0 / (l_rescaled + block_l_rescaled)
                out = (l_rescaled * new_linv) * out + (block_l_rescaled *
                                                       new_linv) * block_out
                linv = new_linv
                m = new_m

                if with_mark_step:
                    htorch.core.mark_step()

            if output_original_dtype:
                final_hidden_list.append(out.to(query.dtype))
            else:
                final_hidden_list.append(out)
            final_linv_list.append(linv)
            final_m_list.append(m)

        output = torch.cat(final_hidden_list, dim=-2)
        final_linv = torch.cat(final_linv_list, dim=-2)
        final_m = torch.cat(final_m_list, dim=-2)

        return output, final_m, final_linv
