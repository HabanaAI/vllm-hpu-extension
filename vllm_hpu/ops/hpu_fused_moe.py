from typing import Callable, Optional

import torch
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.fused_moe.layer import (
    FusedMoE, UnquantizedFusedMoEMethod)


@CustomOp.register("UnquantizedFusedMoEMethod", custom_op=True)
class HPUUnquantizedFusedMoEMethod(UnquantizedFusedMoEMethod):
    """MoE method without quantization."""

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        super().process_weights_after_loading(layer)
        # custom handling for HPU
        num_experts = layer.local_num_experts
        ep_shift = layer.ep_rank * num_experts
        quant_config = layer.quant_config
        from vllm_hpu.extension.ops import (VllmMixtureOfExpertsOp,
                                            VllmMixtureOfExpertsOpFP8,
                                            VllmMixtureOfExpertsOpFP8PerChannel
                                            )

        experts_min, experts_max = ep_shift, num_experts + ep_shift - 1
        if quant_config is None or isinstance(self.quant_method,
                                              UnquantizedFusedMoEMethod):
            moe_op = VllmMixtureOfExpertsOp(
                num_experts,
                experts_min,
                experts_max,
            )
        elif quant_config is not None:
            if hasattr(quant_config, "weight_block_size"):
                moe_op = VllmMixtureOfExpertsOpFP8(
                    num_experts,
                    experts_min,
                    experts_max,
                )
            else:
                moe_op = VllmMixtureOfExpertsOpFP8PerChannel(
                    num_experts,
                    experts_min,
                    experts_max,
                )
        layer.moe_op = moe_op

        for expert_id in range(layer.local_num_experts):
            layer.moe_op.w13_list[expert_id].set_weight(
                layer.w13_weight.data[expert_id])
            layer.moe_op.w2_list[expert_id].set_weight(
                layer.w2_weight.data[expert_id])

    def forward_oot(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        use_grouped_topk: bool,
        top_k: int,
        router_logits: torch.Tensor,
        renormalize: bool,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        global_num_experts: int = -1,
        expert_map: Optional[torch.Tensor] = None,
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        e_score_correction_bias: Optional[torch.Tensor] = None,
        apply_router_weight_on_input: bool = False,
        activation: str = "silu",
        **kwargs,
    ):
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

        return layer.moe_op(
            x,
            topk_ids.to(torch.int64),
            topk_weights.to(x.dtype),
            permuted_weights=True,
            activation=activation,
        ).view(*input_shape)
