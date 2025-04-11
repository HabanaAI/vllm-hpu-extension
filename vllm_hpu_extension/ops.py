###############################################################################
# Copyright (C) 2024-2025 Habana Labs, Ltd. an Intel Company
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.
###############################################################################
from typing import Callable, Optional, Tuple, List

import habana_frameworks.torch as htorch
import torch
import torch.nn.functional as F
import math
import habana_frameworks.torch.core as htcore
import habana_frameworks.torch.utils.experimental as htexp
from vllm_hpu_extension.flags import enabled_flags

from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)


def is_hpu_gaudi2():
    return current_platform.is_hpu() and htexp._get_device_type(
    ) == htexp.synDeviceType.synDeviceGaudi2


def get_hpu_gaudi2_scale_factor():
    return (torch.finfo(torch.float8_e4m3fn).max /
            torch.finfo(torch.float8_e4m3fnuz).max)


def grouped_max(block_max, batch_size, block_groups):
    group_max = torch.full([batch_size + 1, *block_max.shape[1:]], -math.inf,
                           dtype=block_max.dtype, device=block_max.device)
    group_max = group_max.index_reduce_(0, block_groups, block_max, 'amax')
    group_max = group_max.index_select(0, block_groups)
    return group_max


def b2b_impl(tensor, block_mapping, matmul_op):
    shape = tuple(tensor.shape)
    return matmul_op(block_mapping, tensor.view(shape[0], -1)).view(-1, *shape[1:])


def batch2block(tensor, block_mapping, matmul_op=torch.matmul):
    return b2b_impl(tensor, block_mapping, matmul_op)


def block2batch(tensor, block_mapping, matmul_op=torch.matmul):
    return b2b_impl(tensor, block_mapping.t(), matmul_op)


def pipelined_pa(attn, value, block_groups, block_mapping, batch_size,
                 matmul_av_op, batch2block_matmul_op, block2batch_matmul_op):
    # When fp32_softmax is enabled attn is left in fp32 after Q@K
    # We can return to native dtype after we renormalize and calculate the adjustments

    # Normalize the attention scores and cast attn to native dtype
    block_max = attn.amax(dim=-1, keepdim=True)
    adjustment_target_shape = block_max.shape
    attn = attn.sub(block_max)
    attn = attn.exp()
    attn = attn.to(value.dtype)
    block_sums = attn.sum(dim=-1, keepdim=True)
    attn = matmul_av_op(attn, value)
    block_max = block_max.squeeze()
    block_sums = block_sums.squeeze()

    # Calculate maximum of blocks that belong to the same sequences
    # and cast adjustments to native dtype
    group_max = grouped_max(block_max, batch_size, block_groups)
    block_adjustment = (block_max - group_max).exp()
    block_adjustment = block_adjustment.to(value.dtype)
    sum_adjusted = block_sums.mul(block_adjustment)

    # Sum block's sums that belongs to the same sequences
    group_sum_adjusted = block2batch(sum_adjusted, block_mapping, block2batch_matmul_op)
    group_sum_adjusted = batch2block(group_sum_adjusted, block_mapping, batch2block_matmul_op)
    sum_adjusted = sum_adjusted.view(*adjustment_target_shape)
    group_sum_adjusted = group_sum_adjusted.view(*adjustment_target_shape)
    block_adjustment = block_adjustment.view(*adjustment_target_shape)

    # For stability in case some of the sums have been zeroed out during block aggretation
    group_sum_adjusted = torch.maximum(group_sum_adjusted, sum_adjusted)

    # Post processing for the attention scores
    rescale = block_adjustment.div(group_sum_adjusted)
    attn = attn.mul(rescale)
    return attn

def flat_pa_mla(query, key_cache, value_cache, block_list, block_mapping,
                block_bias, block_groups, scale, matmul_qk_op,
                matmul_av_op, batch2block_matmul_op, block2batch_matmul_op,
                keys_fetch_func, values_fetch_func, kv_lora_rank):
    batch_size = query.size(0)
    q_heads = query.size(1)
    kv_heads = key_cache.size(2)

    query = batch2block(scale * query, block_mapping,
                            batch2block_matmul_op).unsqueeze(-2)
    key = keys_fetch_func(key_cache, block_list)
    if value_cache is not None:
        value = values_fetch_func(value_cache, block_list)
        key = torch.concat((value, key), dim=-1)
    elif kv_lora_rank is not None:
        value = key[..., :kv_lora_rank]
    else:
        assert False, "value_cache is None and kv_lora_rank is None"

    key = key.transpose(1, 2)
    value = value.transpose(1, 2)
    block_bias = block_bias.view(key.size(0), 1, 1, -1)
    if kv_heads != q_heads:
        block_bias = block_bias.unsqueeze(1)
        query = query.unflatten(1, (kv_heads, -1))
        key = key.unflatten(1, (kv_heads, 1))
        value = value.unflatten(1, (kv_heads, 1))
        key = key.transpose(3, 4)
    else:
        key = key.transpose(2, 3)

    attn = matmul_qk_op(query, key)
    attn = attn + block_bias
    attn = pipelined_pa(attn,
                        value,
                        block_groups,
                        block_mapping,
                        batch_size=batch_size,
                        matmul_av_op=matmul_av_op,
                        batch2block_matmul_op=batch2block_matmul_op,
                        block2batch_matmul_op=block2batch_matmul_op)
    attn = block2batch(attn, block_mapping, block2batch_matmul_op)
    attn = attn.squeeze(-2)
    if kv_heads != q_heads:
        attn = attn.flatten(1, 2)
    return attn

def flat_pa(query, key_cache, value_cache, block_list, block_mapping,
            block_bias, block_groups, scale, matmul_qk_op,
            matmul_av_op, batch2block_matmul_op, block2batch_matmul_op,
            keys_fetch_func, values_fetch_func, **ignored_args):
    batch_size, _, hidden_size = query.shape
    _, _, kv_heads, head_size = key_cache.shape
    q_heads = hidden_size // head_size

    query_shape = (-1, q_heads, 1, head_size)
    query = batch2block(scale * query, block_mapping, batch2block_matmul_op).view(query_shape)
    key = keys_fetch_func(key_cache, block_list).transpose(1, 2)
    value = values_fetch_func(value_cache, block_list).transpose(1, 2)
    block_bias = block_bias.view(key.size(0), 1, 1, -1)
    if kv_heads != q_heads:
        block_bias = block_bias.unsqueeze(1)
        query = query.unflatten(1, (kv_heads, -1))
        key = key.unflatten(1, (kv_heads, 1))
        value = value.unflatten(1, (kv_heads, 1))
        key = key.transpose(3, 4)
    else:
        key = key.transpose(2, 3)

    attn = matmul_qk_op(query, key)
    if 'fp32_softmax' in enabled_flags():
        attn = attn.float()
        htcore.mark_step()
    attn = attn + block_bias
    attn = pipelined_pa(attn, value, block_groups, block_mapping,
                        batch_size=batch_size, matmul_av_op=matmul_av_op,
                        batch2block_matmul_op=batch2block_matmul_op, block2batch_matmul_op=block2batch_matmul_op)
    attn = block2batch(attn, block_mapping, block2batch_matmul_op)
    attn = attn.squeeze(-2)
    if kv_heads != q_heads:
        attn = attn.flatten(1, 2)
    return attn


def _flex_prompt_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    **ignored_args,
) -> torch.Tensor:
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)

    def _causal(
        score: torch.Tensor,
        batch: torch.Tensor,
        head: torch.Tensor,
        token_q: torch.Tensor,
        token_kv: torch.Tensor,
    ) -> torch.Tensor:
        return torch.where(token_q >= token_kv, score, float("-inf"))

    from torch.nn.attention.flex_attention import flex_attention

    attn_weights = flex_attention(
        query,
        key,
        value,
        score_mod=_causal,
        enable_gqa=True,
        return_lse=False,
        block_mask=None,
        scale=scale,
    )

    attn_weights = attn_weights.transpose(1, 2)
    return attn_weights


def _naive_prompt_attention(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        scale: float,
        attn_bias: Optional[torch.Tensor] = None,
        matmul_qk_op=torch.matmul,
        softmax_op=torch.softmax,
        matmul_av_op=torch.matmul,
        **ignored_args
) -> torch.Tensor:
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)
    query_heads = query.size(1)
    kv_heads = key.size(1)
    if query_heads != kv_heads:
        query = query.unflatten(1, (kv_heads, -1))
        key = key.unflatten(1, (kv_heads, 1))
        value = value.unflatten(1, (kv_heads, 1))
        if attn_bias is not None:
            attn_bias = attn_bias.unsqueeze(2)
    attn_weights = matmul_qk_op(query * scale, key.transpose(-1, -2))
    if 'fp32_softmax' in enabled_flags():
        softmax_op = torch.softmax
        attn_weights = attn_weights.float()
        htcore.mark_step()
    if attn_bias is not None:
        attn_weights = attn_weights.add(attn_bias)
    attn_weights = softmax_op(attn_weights, dim=-1)
    attn_weights = attn_weights.to(query.dtype)
    attn_weights = matmul_av_op(attn_weights, value)
    if query_heads != kv_heads:
        attn_weights = attn_weights.flatten(1, 2)
    attn_weights = attn_weights.transpose(1, 2)
    return attn_weights


def _fsdpa_prompt_attention(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        scale: float,
        fsdpa_op,
        is_causal: bool,
        attn_bias: Optional[torch.Tensor] = None,
        valid_seq_lengths: Optional[torch.Tensor] = None,
        **ignored_args
) -> torch.Tensor:
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)
    if 'fp32_softmax' in enabled_flags():
        softmax_mode = 'fp32'
    else:
        softmax_mode = 'fast'
    recompute_mode = True
    assert attn_bias is not None or valid_seq_lengths is not None, \
        'Either attn_bias or valid_seq_lengths must be != None'
    if is_causal and attn_bias is not None:
        # TODO: causal + attn_bias is not yet supported
        is_causal = False
        valid_seq_lengths = None
    attn_weights = fsdpa_op(query, key, value, attn_bias, 0.0, is_causal,
                            scale, softmax_mode, recompute_mode,
                            valid_seq_lengths, 'right')
    attn_weights = attn_weights.transpose(1, 2)
    return attn_weights


def prompt_attention(
        impl: str,
        **args,
) -> torch.Tensor:
    impl_mapping = {
        'naive': _naive_prompt_attention,
        'fsdpa': _fsdpa_prompt_attention,
        'flex': _flex_prompt_attention,
    }
    assert impl in impl_mapping, f'Unsupported implementation: {impl}'
    return impl_mapping[impl](**args)


def prompt_attention_with_context(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_list: torch.Tensor,
    attn_bias: torch.Tensor,
    scale: float,
    matmul_qk_op,
    matmul_av_op,
    softmax_op,
    keys_fetch_func,
    values_fetch_func,
    **ignored_args,
) -> torch.Tensor:
    htorch.core.mark_step()
    query.mul_(scale)

    batch_size, _, query_heads, _ = query.shape
    _, block_size, kv_heads, _ = key_cache.shape
    max_num_blocks = block_list.size(-1) // batch_size
    context_len = max_num_blocks * block_size

    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)

    past_keys = keys_fetch_func(key_cache, block_list)
    past_keys = past_keys.reshape(batch_size, context_len, kv_heads, -1)
    past_keys = past_keys.transpose(1, 2)
    key = torch.concat((past_keys, key), dim=-2)

    past_values = values_fetch_func(value_cache, block_list)
    past_values = past_values.reshape(batch_size, context_len, kv_heads, -1)
    past_values = past_values.transpose(1, 2)
    value = torch.concat((past_values, value), dim=-2)

    if query_heads != kv_heads:
        query = query.unflatten(1, (kv_heads, -1))
        key = key.unflatten(1, (kv_heads, 1))
        past_keys = past_keys.unflatten(1, (kv_heads, 1))
        value = value.unflatten(1, (kv_heads, 1))
        past_values = past_values.unflatten(1, (kv_heads, 1))
        if attn_bias is not None:
            attn_bias = attn_bias.unsqueeze(2)

    attn_weights = matmul_qk_op(query, key.transpose(-1, -2))
    if attn_bias is not None:
        attn_weights.add_(attn_bias)
    attn_weights = softmax_op(attn_weights, dim=-1)
    attn_weights = matmul_av_op(attn_weights, value)

    if query_heads != kv_heads:
        attn_weights = attn_weights.flatten(1, 2)

    attn_weights = attn_weights.transpose(1, 2)
    htorch.core.mark_step()
    return attn_weights


class LoraMask:
    lora_mask = None

    @staticmethod
    def setLoraMask(mask):
        LoraMask.lora_mask = mask

    @staticmethod
    def getLoraMask():
        return LoraMask.lora_mask


def dispatch_bgmv_linear(
    y: torch.Tensor,
    x: torch.Tensor,
    wa_t_all: torch.Tensor,
    wb_t_all: torch.Tensor,
    layer_idx: int,
    scale: float,
):
    """
    `wa_t_all` and `wb_t_all` contains all LoRA A and LoRA B weight matrices
    stacked at dimension 0 into single tensors, assuming same rank. `wa` is the
    reshaped and transposed version of `wa_t_all` of shape
    (h_in, max_loras * lora_rank) and `wb` is the transposed and reshaped
    version of `wb_t_all` of shape (max_loras * lora_rank, h_out).

    Matmul input `x` with `wa`. Multiply `x` with a mask to zero-out inputs of
    inactive LoRA indices. Matmul masked output with `wb` and scale it to get
    the final output.
    """

    assert layer_idx == 0, f'layer_idx should be 0, but got {layer_idx}'
    mask = LoraMask.getLoraMask()

    wa = wa_t_all[:, 0, :, :]
    wb = wb_t_all[:, 0, :, :].transpose(1, 2)
    wa = wa.reshape(wa.shape[0] * wa.shape[1], wa.shape[2]).transpose(0, 1)
    wb = wb.reshape(wb.shape[0] * wb.shape[1], wb.shape[2])

    out = x @ wa
    assert (out.shape == mask.shape)
    out = out * mask
    out = out @ wb
    y += out * scale


def dispatch_bgmv_embedding(
    y: torch.Tensor,
    x: torch.Tensor,
    wb_t_all: torch.Tensor,
    layer_idx: int,
):
    """
    `wb_t_all` contains all LoRA-B weight matrices stacked at dimension 0 into
    a single tensor, assuming same rank. `wb` is the transposed and reshaped
    version of `wb_t_all` of shape (num_loras * lora_rank, embedding_dim).

    Output of LoRA-A embedding (tensor x) is repeated max_loras times to match
    the shape of `wb`. Multiply `x` with a mask to zero-out inputs of inactive
    LoRA indices. Matmul masked output with `wb` and scale it to get the final
    output.
    """

    assert layer_idx == 0, f'layer_idx should be 0, but got {layer_idx}'
    max_loras = wb_t_all.size(0)

    x = x.repeat(1, max_loras)
    x = x * LoraMask.getLoraMask()
    wb = wb_t_all[:, 0, :, :].transpose(1, 2)
    wb = wb.reshape(wb.shape[0] * wb.shape[1], wb.shape[2])
    out = x @ wb
    y += out


class MoeMatmul(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def set_weight(self, w):
        self.weight = w

    def forward(self, state, expert_id, w):
        raise NotImplementedError()


class VllmMixtureOfExpertsOp(torch.nn.Module):

    def __init__(self, num_total_experts, experts_min: int = 0, experts_max: int = 8):
        super().__init__()
        self.w13_list = torch.nn.ModuleList(
            [MoeMatmul() for _ in range(num_total_experts)])
        self.w2_list = torch.nn.ModuleList(
            [MoeMatmul() for _ in range(num_total_experts)])
        self.num_experts = num_total_experts
        self.experts_min = experts_min
        self.experts_max = experts_max

        max_expert_per_slice = 32
        self.moe_n_slice = 1 if self.num_experts <= max_expert_per_slice \
                else self.num_experts // max_expert_per_slice
        self.num_expert_per_group = self.num_experts // self.moe_n_slice

    def forward(self,
                hidden_states,
                expert_routing_table,
                router_weights,
                permuted_weights=True,
                activation="silu"):
        # pre-processing for custom op inputs
        experts_range = range(self.num_experts)
        w1_list = [self.w13_list[i].weight.squeeze() for i in experts_range]
        w2_list = [self.w2_list[i].weight.squeeze() for i in experts_range]

        for i in range(self.moe_n_slice):
            w1_list_slice = w1_list[i * self.num_expert_per_group:(i + 1) * self.num_expert_per_group]
            w2_list_slice = w2_list[i * self.num_expert_per_group:(i + 1) * self.num_expert_per_group]
            min_expert = self.experts_min + i * self.num_expert_per_group
            max_expert = min_expert + self.num_expert_per_group - 1
            slice_final_hidden_states = torch.ops.hpu.mixture_of_experts(
                hidden_states=hidden_states,
                expert_routing_table=expert_routing_table,
                router_weights=router_weights,
                w12=w1_list_slice,
                w3=w2_list_slice,
                permuted_weights=permuted_weights,
                activation=activation,
                experts_min=min_expert,
                experts_max=max_expert)
            if i == 0:
                final_hidden_states = slice_final_hidden_states
            else:
                final_hidden_states += slice_final_hidden_states
            htorch.core.mark_step()
        return final_hidden_states


def pad_weight(weight, block_size):
    """Pads a matrix to make its dimensions multiples of block_size."""
    M, N = weight.shape[-2:]
    block_size_m, block_size_n = block_size
    pad_M = (block_size_m - M % block_size_m) % block_size_m
    pad_N = (block_size_n - N % block_size_n) % block_size_n

    if pad_M == 0 and pad_N == 0:
        return weight, M, N  # No padding needed
    padded_weight = torch.nn.functional.pad(weight, (0, pad_N, 0, pad_M), mode='constant', value=0)
    return padded_weight, M, N  # Return original dimensions for unpadding


def unpad_weight(weight, original_M, original_N, keep_first_dim=False):
    """Removes padding from the matrix to restore its original shape."""
    if (weight.shape[-2] == original_M) and (weight.shape[-1] == original_N):
        return weight
    if keep_first_dim:
        return weight[:, :original_M, :original_N]
    else:
        return weight[:original_M, :original_N]


def pad_block_fp8_weight_naive(weight, weight_scale, block_size):

    assert len(block_size) == 2

    block_size_m, block_size_n = block_size
    weight_scale_m, weight_scale_n = weight_scale.shape[-2:]

    weight, orig_M, orig_N = pad_weight(weight, block_size)
    M, N = weight.shape[-2:]

    assert weight_scale_m == M // block_size_m
    assert weight_scale_n == N // block_size_n

    return weight, orig_M, orig_N


def dequant_block_fp8_weight_naive(weight, weight_scale, block_size, dtype=torch.bfloat16, original_M=None, original_N=None, do_unpad=False):
    if weight_scale is None:
        return weight
    assert len(block_size) == 2

    weight_shape_len = len(weight.shape)

    block_size_m, block_size_n = block_size

    # mul scale
    if weight_shape_len == 2:
        weight_scale_m, weight_scale_n = weight_scale.shape
        weight_scale = weight_scale.view(weight_scale_m, 1, weight_scale_n, 1)
        weight = weight.view(weight_scale_m, block_size_m, weight_scale_n, block_size_n)
        dequant_weight = weight.to(dtype) * weight_scale.to(dtype)
        dequant_weight = dequant_weight.view(weight_scale_m*block_size_m, weight_scale_n*block_size_n)
        keep_first_dim = False
    elif weight_shape_len == 3:
        fd, weight_scale_m, weight_scale_n = weight_scale.shape
        weight_scale = weight_scale.view(fd, weight_scale_m, 1, weight_scale_n, 1)
        weight = weight.view(fd, weight_scale_m, block_size_m, weight_scale_n, block_size_n)
        dequant_weight = weight.to(dtype) * weight_scale.to(dtype)
        dequant_weight = dequant_weight.view(fd, weight_scale_m*block_size_m, weight_scale_n*block_size_n)
        keep_first_dim = True
    else:
        raise ValueError("Only support original weight shape is either 2 or 3")

    if do_unpad:
        dequant_weight = unpad_weight(dequant_weight, original_M, original_N, keep_first_dim=keep_first_dim)

    return dequant_weight


def apply_block_fp8_linear_hpu_dequant(
    input: torch.Tensor,
    weight: torch.Tensor,
    block_size: List[int],
    weight_scale: torch.Tensor,
    input_scale: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    original_M: Optional[torch.Tensor] = None,
    original_N: Optional[torch.Tensor] = None,
    do_unpad: bool = False,
) -> torch.Tensor:
    assert input_scale is None
    # View input as 2D matrix for fp8 methods
    input_2d = input.view(-1, input.shape[-1])
    original_M = original_M.data.item()
    original_N = original_N.data.item()
    weight = dequant_block_fp8_weight_naive(weight, weight_scale, block_size, input.dtype, original_M, original_N, do_unpad)
    output = torch.nn.functional.linear(input_2d, weight, bias=None)
    if bias is not None:
        output = output + bias
    return output.to(dtype=input.dtype).view(*input.shape[:-1], -1)


def fp8_block_linear_postprocess_weights(layer):
    weight, orig_M, orig_N = pad_block_fp8_weight_naive(
        layer.weight.data,
        layer.weight_scale_inv.data,
        layer.quant_config.weight_block_size)
    layer.weight = torch.nn.Parameter(weight, requires_grad=False)
    orig_M = torch.nn.Parameter(torch.tensor(orig_M, dtype=torch.int32), requires_grad=False)
    orig_N = torch.nn.Parameter(torch.tensor(orig_N, dtype=torch.int32), requires_grad=False)
    layer.register_parameter("orig_M", orig_M)
    layer.register_parameter("orig_N", orig_N)
    htorch.core.mark_step()
    return layer


def fp8_block_moe_prepare_weights(layer):
    for index in range(layer.moe_op.num_experts):
        layer.moe_op.w13_list[index].set_weight(layer.w13_weight[index])
        layer.moe_op.w13_list[index].set_scale_inv_fp8(
            layer.w13_weight_scale_inv[index]
        )
        layer.moe_op.w13_list[index].set_weight_block_size(
            layer.quant_config.weight_block_size
        )

        layer.moe_op.w2_list[index].set_weight(layer.w2_weight[index])
        layer.moe_op.w2_list[index].set_scale_inv_fp8(
            layer.w2_weight_scale_inv[index]
        )
        layer.moe_op.w2_list[index].set_weight_block_size(
            layer.quant_config.weight_block_size
        )
    htorch.core.mark_step()
    return layer


class MoeFP8Matmul(torch.nn.Module):
    def __init__(
        self,
        block_size: Tuple[int, int] = (128, 128),
        high_precision=torch.bfloat16,
    ):
        super().__init__()
        self.block_size = block_size
        self.high_precision = high_precision
        self.is_dequantized = False

    def set_weight(self, w: torch.Tensor):
        self.weight = w

    def set_scale_inv_fp8(self, scale_inv_fp8: torch.Tensor):
        self.scale_inv_fp8 = scale_inv_fp8

    def set_high_precision(self, high_precision=torch.bfloat16):
        self.high_precision = high_precision

    def set_weight_block_size(self, block_size: Tuple[int, int] = (128, 128)):
        self.block_size = block_size

    def get_dequant_weight(self):
        return dequant_block_fp8_weight_naive(
            self.weight,
            self.scale_inv_fp8,
            block_size=self.block_size,
            dtype=self.high_precision,
        )

    def forward(self, state, expert_id, w):
        raise NotImplementedError()

    def dequant_block_fp8_weight(self, layer: "MoeFP8Matmul") -> torch.Tensor:
        # This function is called by INC during either the measurement or quantization phase.
        # - In the quantization phase, INC requantizes the BF16 weight to FP8 and updates the weight.
        # - In the measurement phase, INC only measures the BF16 weight without updating it.
        # Tracking the BF16 weight can lead to Out of Memory (OoM) issues, so we avoid storing it.
        # If the weight has already been updated, we return it directly.
        if hasattr(layer, "updated_fp8_weight") and layer.updated_fp8_weight:
            return layer.weight

        dequant_weight = layer.get_dequant_weight()
        layer.is_dequantized = True
        return dequant_weight

    def get_dequant_weights_func(
        self,
    ) -> Optional[Callable[[torch.nn.Module], torch.Tensor]]:
        return self.dequant_block_fp8_weight


class VllmMixtureOfExpertsOpFP8(torch.nn.Module):
    def __init__(
        self, num_experts: int, experts_min: int = 0, experts_max: int = 8
    ):
        super().__init__()
        self.w13_list = torch.nn.ModuleList(
            [MoeFP8Matmul() for _ in range(num_experts)]
        )
        self.w2_list = torch.nn.ModuleList(
            [MoeFP8Matmul() for _ in range(num_experts)]
        )
        max_expert_per_slice = 32
        self.num_experts = num_experts
        self.experts_min = experts_min
        self.experts_max = experts_max
        self.moe_n_slice = 1 if self.num_experts <= max_expert_per_slice \
                else self.num_experts // max_expert_per_slice
        self.num_expert_per_group = self.num_experts // self.moe_n_slice

    def forward(
        self,
        x,
        topk_ids,
        topk_weights,
        permuted_weights=True,
        activation="silu",
    ):
        w13_list = []
        w2_list = []
        for j in range(self.num_experts):
            w13_list.append(self.w13_list[j].get_dequant_weight())
            w2_list.append(self.w2_list[j].get_dequant_weight())
        htorch.core.mark_step()

        for i in range(self.moe_n_slice):
            w13_list_slice = w13_list[i * self.num_expert_per_group:(i + 1) * self.num_expert_per_group]
            w2_list_slice = w2_list[i * self.num_expert_per_group:(i + 1) * self.num_expert_per_group]
            min_expert = self.experts_min + i * self.num_expert_per_group
            max_expert = min_expert + self.num_expert_per_group - 1
            slice_final_hidden_states = torch.ops.hpu.mixture_of_experts(
                hidden_states=x,
                expert_routing_table=topk_ids,
                router_weights=topk_weights,
                w12=w13_list_slice,
                w3=w2_list_slice,
                permuted_weights=permuted_weights,
                activation=activation,
                experts_min=min_expert,
                experts_max=max_expert,
            )
            htorch.core.mark_step()
            if i == 0:
                final_hidden_states = slice_final_hidden_states
            else:
                final_hidden_states += slice_final_hidden_states
        return final_hidden_states

# fp8
def scaled_fp8_quant(
    input: torch.Tensor,
    scale: Optional[torch.Tensor] = None,
    batch_dim_padding: Optional[int] = None,
    scale_ub: Optional[torch.Tensor] = None,
    use_per_token_if_dynamic: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize input tensor to FP8 and return quantized tensor and scale.
    This function supports both static and dynamic quantization: If you
    provide the scale, it will use static scaling and if you omit it,
    the scale will be determined dynamically. The function also allows
    optional padding of the output tensor for downstream kernels that
    will benefit from padding.
    Args:
        input: The input tensor to be quantized to FP8
        scale: Optional scaling factor for the FP8 quantization
        scale_ub: Optional upper bound for scaling factor in dynamic
            per token case
        batch_dim_padding: If specified, pad the first dimension
            of the output to at least this value.
        use_per_token_if_dynamic: Whether to do per_tensor or per_token
            in the dynamic quantization case.
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The output tensor in FP8 and
            scaling factor.
    """
    if batch_dim_padding:
        shape = (max(batch_dim_padding, input.shape[0]), *input.shape[1:])
        output = torch.empty(shape,
                             device=input.device,
                             dtype=torch.float8_e4m3fn)
    else:
        output = torch.empty_like(input, dtype=torch.float8_e4m3fn)
    if scale is None:
        raise "dynamic scaled_fp8_quant not implemented for HPU"
        # TODO: calculate scale to match gaudi2 240 range instead of 448
        if use_per_token_if_dynamic:
            scale = torch.empty((input.numel() // input.shape[-1], 1),
                                device=input.device,
                                dtype=torch.float32)
            torch.ops._C.dynamic_per_token_scaled_fp8_quant(
                output, input, scale, scale_ub)
        else:
            scale = torch.zeros(1, device=input.device, dtype=torch.float32)
            torch.ops._C.dynamic_scaled_fp8_quant(output, input, scale)
    else:
        output = torch.ops.hpu.cast_to_fp8_v2(input,
                                              1 / scale,
                                              False,
                                              False,
                                              dtype=torch.float8_e4m3fn)[0]

    return output, scale
