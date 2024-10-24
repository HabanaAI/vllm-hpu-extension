###############################################################################
# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.
###############################################################################
from typing import Optional, Tuple

import habana_frameworks.torch as htorch
import torch
import os
import torch.nn.functional as F
import math

from vllm.logger import init_logger

logger = init_logger(__name__)
HPUFusedRMSNorm = None
try:
    from habana_frameworks.torch.hpex.normalization import FusedRMSNorm
    HPUFusedRMSNorm = FusedRMSNorm
except ImportError:
    logger.warning("Could not import HPU FusedRMSNorm kernel. "
                   "vLLM will use forward_native implementation of RMSNorm.")
HPUFusedSDPA = None
try:
    from habana_frameworks.torch.hpex.kernels import FusedSDPA
    HPUFusedSDPA = FusedSDPA
except ImportError:
    logger.warning("Could not import HPU FusedSDPA kernel. "
                   "vLLM will use native implementation.")


class SoftmaxNormalization:

    def __init__(self, selected_impl):
        implementations = {
            'wsum': self.wsum,
            'amax': self.amax,
            'head_amax': self.head_amax,
            'wsum_head_amax': self.wsum_head_amax,
            'index_reduce': self.index_reduce,
            'scatter_reduce': self.scatter_reduce,
        }
        supported_impls = implementations.keys()
        for impl in selected_impl:
            assert impl in supported_impls, f'Unsupported pa softmax impl - {impl} . Supported values: {list(supported_impls)}'
        self.selected_impl = [implementations[impl] for impl in selected_impl]

    def __call__(self, attn, **kwargs):
        for impl in self.selected_impl:
            attn = impl(attn, **kwargs)
        return attn

    @staticmethod
    def amax(attn, **rest):
        """Normalize by global maximum values"""
        dims = tuple(range(1, attn.dim()))
        attn_max = attn.amax(dims).amax()
        return attn.sub_(attn_max)

    @staticmethod
    def head_amax(attn, **rest):
        """Normalize by head maximum values"""
        dims = (0, attn.dim() - 1)
        attn_max = attn.amax(dims, keepdim=True)
        return attn.sub_(attn_max)

    @staticmethod
    def wsum(attn, block_mapping, block_scales, **rest):
        """Normalize by weighted sum of block maximums"""
        block_sum_attn = attn.amax(-1)
        missing_dims = block_sum_attn.dim() - block_scales.dim()
        block_sum_attn.mul_(block_scales.reshape(-1, *[1 for _ in range(missing_dims)]))
        block_sum_attn = block2batch(block_sum_attn, block_mapping)
        block_sum_attn = batch2block(block_sum_attn, block_mapping)
        return attn.sub_(block_sum_attn.unsqueeze(-1))

    @staticmethod
    def wsum_head_amax(attn, block_mapping, block_scales, **rest):
        """Perform weighted sum fused with head maximum normalization"""
        attn_max = attn.amax(-1)
        missing_dims = attn_max.dim() - block_scales.dim()
        block_sum_attn = attn_max.mul(block_scales.reshape(-1, *[1 for _ in range(missing_dims)]))
        block_sum_attn = block2batch(block_sum_attn, block_mapping)
        block_sum_attn = batch2block(block_sum_attn, block_mapping)
        attn.sub_(block_sum_attn.unsqueeze(-1))
        attn_max.sub_(block_sum_attn)
        attn_max = attn_max.amax(0, keepdim=True)
        return attn.sub_(attn_max.unsqueeze(-1))

    @staticmethod
    def index_reduce(attn, batch_size, block_groups, **rest):
        """Normalize by max in block groups using index_reduce"""
        block_max = attn.amax(-1)
        grouped_max = torch.full([batch_size + 1, *attn.shape[1:-1]], -math.inf, dtype=attn.dtype, device=attn.device)
        grouped_max.index_reduce_(0, block_groups, block_max, 'amax')
        grouped_max = grouped_max.index_select(0, block_groups)
        attn.sub_(grouped_max.unsqueeze(-1))
        return attn

    @staticmethod
    def scatter_reduce(attn, batch_size, block_groups, **rest):
        """Normalize by max in block groups using scatter_reduce"""
        block_max = attn.amax(-1)
        grouped_max = torch.full([batch_size + 1, *attn.shape[1:-1]], -math.inf, dtype=attn.dtype, device=attn.device)
        indices = block_groups.view(-1, *[1 for _ in grouped_max.shape[1:]]).expand(-1, *grouped_max.shape[1:])
        grouped_max.scatter_reduce_(0, indices, block_max, 'amax')
        grouped_max = grouped_max.index_select(0, block_groups)
        attn.sub_(grouped_max.unsqueeze(-1))
        return attn


normalize = SoftmaxNormalization(os.environ.get('VLLM_PA_SOFTMAX_IMPL', 'wsum_head_amax').split(','))


def batch2block(tensor, block_mapping):
    shape = tuple(tensor.shape)
    return (block_mapping @ tensor.view(shape[0], -1)).view(-1, *shape[1:])


def block2batch(tensor, block_mapping):
    shape = tuple(tensor.shape)
    return (block_mapping.t() @ tensor.view(shape[0], -1)).view(-1, *shape[1:])


def block_softmax(batch_size, attn, block_mapping, block_scales, block_groups):
    attn = normalize(batch_size=batch_size, attn=attn, block_mapping=block_mapping, block_scales=block_scales, block_groups=block_groups)
    attn = attn.exp_()
    sums = attn.sum(dim=-1).unsqueeze(-1)
    block_sum = sums
    sums = block2batch(sums, block_mapping)
    sums = batch2block(sums, block_mapping)
    sums.add_(torch.finfo(sums.dtype).tiny)
    sums = torch.maximum(block_sum, sums)
    attn.div_(sums)
    return attn


def flat_pa(query, key_cache, value_cache, block_list, block_mapping,
            block_bias, block_scales, block_groups, scale, matmul_qk_op, matmul_av_op, keys_fetch_func,
            values_fetch_func):
    batch_size = query.size(0)
    q_heads = query.size(1)
    kv_heads = key_cache.size(2)

    query = batch2block(scale * query, block_mapping).unsqueeze(-2)
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

    attn = matmul_qk_op(query, key) + block_bias
    attn = block_softmax(batch_size, attn, block_mapping, block_scales, block_groups)
    attn = matmul_av_op(attn, value)
    attn = block2batch(attn, block_mapping)
    attn = attn.squeeze(-2)
    if kv_heads != q_heads:
        attn = attn.flatten(1, 2)
    return attn


def silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    return F.silu(x[..., :d]) * x[..., d:]


#TODO: remove after fusedsdpa fix for query_head != kv_head
def repeat_kv(kv: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep).
    The kv go from (batch, num_key_value_heads, seqlen, head_dim) to
    (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = kv.shape
    if n_rep == 1:
        return kv
    kv = kv[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen,
                                     head_dim)
    return kv.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def prompt_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_bias: Optional[torch.Tensor] = None,
    p: float = 0.0,
    scale: Optional[float] = None,
    matmul_qk_op=torch.matmul,
    softmax_op=torch.softmax,
    matmul_av_op=torch.matmul,
    valid_seq_lengths: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)
    query_heads = query.size(1)
    kv_heads = key.size(1)
    if attn_bias is not None or HPUFusedSDPA is None:
        if query_heads != kv_heads:
            query = query.unflatten(1, (kv_heads, -1))
            key = key.unflatten(1, (kv_heads, 1))
            value = value.unflatten(1, (kv_heads, 1))
            if attn_bias is not None:
                attn_bias = attn_bias.unsqueeze(2)
        attn_weights = matmul_qk_op(query * scale, key.transpose(-1, -2))
        if attn_bias is not None:
            attn_weights.add_(attn_bias)
        attn_weights = softmax_op(attn_weights, dim=-1)
        attn_weights = matmul_av_op(attn_weights, value)
        if query_heads != kv_heads:
            attn_weights = attn_weights.flatten(1, 2)
    else:
        #TODO: remove after fusedsdpa fix for query_heads != kv_heads
        if query_heads != kv_heads:
            key = repeat_kv(key, int(query_heads // kv_heads))
            value = repeat_kv(value, int(query_heads // kv_heads))
        softmax_mode = 'fast'
        recompute_mode = True
        attn_weights = FusedSDPA.apply(query, key, value, None, 0.0, True,
                                       scale, softmax_mode, recompute_mode,
                                       valid_seq_lengths, 'right')
    attn_weights = attn_weights.transpose(1, 2)
    return attn_weights


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
        attn_bias = attn_bias.unsqueeze(2)

    attn_weights = matmul_qk_op(query, key.transpose(-1, -2))
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
    scale: float,
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
    y += out * scale


class MoeMatmul(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def set_weight(self, w):
        self.weight = w

    def forward(self, state, expert_id, w):
        return torch.matmul(state, w[expert_id].transpose(0, 1))


def calculate_routing_tensors(score, topk, hidden_states_dtype):
    routing_weights = F.softmax(score, dim=1, dtype=torch.float32)
    routing_weights, selected_experts = torch.topk(routing_weights,
                                                   topk,
                                                   dim=-1)
    routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
    routing_weights = routing_weights.to(hidden_states_dtype)
    return routing_weights, selected_experts


class StaticFusedMOE(torch.nn.Module):

    def __init__(self, num_total_experts):
        super().__init__()
        self.w13_list = torch.nn.ModuleList(
            [MoeMatmul() for _ in range(num_total_experts)])
        self.w2_list = torch.nn.ModuleList(
            [MoeMatmul() for _ in range(num_total_experts)])
        self.num_total_experts = num_total_experts

    def forward(self, hidden_states, w1, w2, score, topk):
        B, D = hidden_states.shape
        routing_weights, selected_experts = calculate_routing_tensors(
                        score, topk, hidden_states.dtype)
        final_hidden_states = torch.zeros((1, B, D),
                                          dtype=hidden_states.dtype,
                                          device=hidden_states.device)
        padded_weights = torch.zeros((B, self.num_total_experts),
                                     dtype=hidden_states.dtype,
                                     device=hidden_states.device)
        padded_weights.scatter_(-1, selected_experts, routing_weights)
        padded_weights = padded_weights.reshape(-1, B, self.num_total_experts)
        padded_weights = padded_weights.permute(2, 0, 1).unsqueeze(-1)
        htorch.core.mark_step()

        for expert_idx in range(self.num_total_experts):
            padded_weight = padded_weights[expert_idx]
            current_state_static = hidden_states.reshape(-1, D)
            w_output = self.w13_list[expert_idx](current_state_static, expert_idx, w1)
            w_output = silu_and_mul(w_output)
            w_output = self.w2_list[expert_idx](w_output, expert_idx, w2)
            current_hidden_states_static = w_output * padded_weight
            final_hidden_states += current_hidden_states_static

        return final_hidden_states.view(-1, D)


class DynamicFusedMOE(torch.nn.Module):

    def __init__(self, num_total_experts):
        super().__init__()
        self.num_total_experts = num_total_experts

    def forward(self, hidden_states, w1, w2, score, topk):
        htorch.core.mark_step()
        routing_weights, selected_experts = calculate_routing_tensors(
                score, topk, hidden_states.dtype)
        # pre-processing for custom op inputs
        experts_range = range(self.num_total_experts)
        w1_list = [w1[i,:,:].squeeze() for i in experts_range]
        w2_list = [w2[i,:,:].squeeze() for i in experts_range]

        final_hidden_states = torch.ops.hpu.mixture_of_experts(
                hidden_states=hidden_states,
                expert_routing_table=selected_experts,
                router_weights=routing_weights,
                w12=w1_list,
                w3=w2_list,
                permuted_weights=True,
                activation="silu",
                experts_min=0,
                experts_max=7
        )

        return final_hidden_states.view(-1, hidden_states.shape[1])


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
        #TODO: calculate scale to match gaudi2 240 range instead of 448
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
