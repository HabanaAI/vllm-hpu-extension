from typing import Optional, Union
import torch
from vllm.model_executor.layers.layernorm import \
    RMSNorm
from vllm.model_executor.custom_op import CustomOp


@CustomOp.register("RMSNorm", is_oot_custom_op=True)
class HPURMSNorm(RMSNorm):

    def forward_oot(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        from vllm_hpu.extension.kernels import rms_norm
        HPUFusedRMSNorm = rms_norm()
        if x.dim() < 3:
            # fix an known bug before synapse 1.21 release
            HPUFusedRMSNorm = None
        if HPUFusedRMSNorm is None:
            return self.forward_native(x, residual)
        if residual is not None:
            orig_shape = x.shape
            residual = residual + x.view(residual.shape)
            # Note: HPUFusedRMSNorm requires 3D tensors as inputs
            x = HPUFusedRMSNorm.apply(residual, self.weight,
                                      self.variance_epsilon)
            return x.view(orig_shape), residual

        x = HPUFusedRMSNorm.apply(x, self.weight, self.variance_epsilon)
        return x
