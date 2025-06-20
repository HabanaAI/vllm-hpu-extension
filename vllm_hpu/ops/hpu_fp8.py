from typing import Callable, Optional

import torch
from vllm_hpu import envs
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.fused_moe.layer import FusedMoE

from vllm.model_executor.layers.quantization import fp8
from vllm.model_executor.layers.quantization.fp8 import (Fp8LinearMethod as
                                                         OrigFp8LinearMethod,
                                                         Fp8MoEMethod,
                                                         Fp8Config)
import vllm_hpu.extension.ops as hpu_ops
from vllm_hpu.extension.ops import (VllmMixtureOfExpertsOpFP8PerChannel,
                                    VllmMixtureOfExpertsOpFP8)


class Fp8LinearMethod(OrigFp8LinearMethod):

    def create_weights(self, *args, **kwargs) -> None:
        super().create_weights(*args, **kwargs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.quant_config = self.quant_config
        if self.block_quant:
            layer = hpu_ops.fp8_block_linear_postprocess_weights(
                layer, envs.VLLM_HPU_FORCE_CHANNEL_FP8)
            return

    def apply(self,
              layer: torch.nn.Module,
              x: torch.Tensor,
              bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.block_quant:
            assert self.quant_config.weight_block_size is not None
            return hpu_ops.apply_block_fp8_linear_hpu(
                input=x,
                layer=layer,
                block_size=self.quant_config.weight_block_size,
                bias=bias,
                do_unpad=True,
                force_channel_fp8=envs.VLLM_HPU_FORCE_CHANNEL_FP8,
            )
        weight_scale = layer.weight_scale.transpose(0, 1)
        input_scale = getattr(layer, 'input_scale', None)
        return hpu_ops.apply_fp8_linear_hpu(input=x,
                                            weight=layer.weight,
                                            weight_scale=weight_scale,
                                            input_scale=input_scale,
                                            bias=bias,
                                            trans_B=False)


@CustomOp.register_oot(name='Fp8MoEMethod')
class HPUFp8MoEMethod(Fp8MoEMethod):

    def __init__(self, quant_config: Fp8Config):
        self.quant_config = quant_config
        self.block_quant = self.quant_config.weight_block_size is not None

        # Disable marlin
        self.use_marlin = False

        # disable DeepGemm support.
        self.allow_deep_gemm = False

        self.topk_indices_dtype = None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        num_experts = layer.local_num_experts
        ep_shift = layer.ep_rank * num_experts

        experts_min, experts_max = ep_shift, num_experts + ep_shift - 1
        if self.block_quant and not envs.VLLM_HPU_FORCE_CHANNEL_FP8:
            layer.moe_op = VllmMixtureOfExpertsOpFP8(
                num_experts,
                experts_min,
                experts_max,
            )
        else:
            layer.moe_op = VllmMixtureOfExpertsOpFP8PerChannel(
                num_experts,
                experts_min,
                experts_max,
            )
        if self.block_quant:
            layer = hpu_ops.fp8_block_moe_prepare_weights(
                layer, envs.VLLM_HPU_FORCE_CHANNEL_FP8)
        else:
            layer = hpu_ops.fp8_channel_moe_prepare_weights(layer)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool = False,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        global_num_experts: int = -1,
        expert_map: Optional[torch.Tensor] = None,
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        e_score_correction_bias: Optional[torch.Tensor] = None,
        apply_router_weight_on_input: bool = False,
        activation: str = "silu",
    ) -> torch.Tensor:
        input_shape = x.shape
        x = x.view(-1, x.shape[-1])
        if use_grouped_topk or custom_routing_function is not None:
            topk_weights, topk_ids = FusedMoE.select_experts(
                hidden_states=x,
                router_logits=router_logits,
                use_grouped_topk=use_grouped_topk,
                top_k=top_k,
                renormalize=renormalize,
                topk_group=topk_group,
                num_expert_group=num_expert_group,
                custom_routing_function=custom_routing_function,
                scoring_func=scoring_func,
                e_score_correction_bias=e_score_correction_bias)
        else:
            import torch.nn.functional as F
            topk_weights = F.softmax(router_logits, dim=1, dtype=torch.float32)
            topk_weights, topk_ids = torch.topk(topk_weights, top_k, dim=-1)
            topk_weights /= topk_weights.sum(dim=-1, keepdim=True)
            topk_weights = topk_weights.to(x.dtype)
        topk_ids = topk_ids.view(*x.shape[:-1], -1)
        topk_weights = topk_weights.view(*x.shape[:-1], -1)
        output = layer.moe_op(
            x,
            topk_ids.to(torch.int64),
            topk_weights.to(x.dtype),
            permuted_weights=True,
            activation=activation,
        )
        return output.view(*input_shape)


fp8.Fp8LinearMethod = Fp8LinearMethod
fp8.Fp8MoEMethod = HPUFp8MoEMethod
