# MIT License

# Copyright (c) 2025 Intel Corporation

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from fractions import Fraction
from typing import Any, Dict, List, Optional

import torch
from torch.nn.parameter import Parameter

from vllm.model_executor.layers.linear import LinearBase, LinearMethodBase
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.parameter import (ChannelQuantScaleParameter,
                                           GroupQuantScaleParameter,
                                           PackedColumnParameter,
                                           PackedvLLMParameter,
                                           RowvLLMParameter)


class GPTQHPUConfig(QuantizationConfig):
    """Config class for GPTQ.

    Reference: https://arxiv.org/abs/2210.17323
    """

    def __init__(
        self,
        weight_bits: int,
        group_size: int,
        desc_act: bool,
        lm_head_quantized: bool,
    ) -> None:
        self.weight_bits = weight_bits
        self.group_size = group_size
        self.desc_act = desc_act
        self.lm_head_quantized = lm_head_quantized
        self.pack_factor = Fraction(32, self.weight_bits)
        if self.weight_bits not in [2, 3, 4, 8]:
            raise ValueError(
                "Currently, only 2/3/4/8-bit weight quantization is "
                f"supported for GPTQ, but got {self.weight_bits} bits.")

    def __repr__(self) -> str:
        return (f"GPTQHPUConfig(weight_bits={self.weight_bits}, "
                f"group_size={self.group_size}, "
                f"desc_act={self.desc_act}),"
                f"lm_head_quantized={self.lm_head_quantized}")

    @classmethod
    def get_name(cls) -> str:
        return "gptq_hpu"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.bfloat16]

    @classmethod
    # Need to figure it out
    def get_min_capability(cls) -> int:
        return 0

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return ["quantize_config.json"]

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "GPTQHPUConfig":
        weight_bits = cls.get_from_keys(config, ["bits"])
        group_size = cls.get_from_keys(config, ["group_size"])
        desc_act = cls.get_from_keys(config, ["desc_act"])
        lm_head_quantized = cls.get_from_keys_or(config, ["lm_head"],
                                                 default=False)
        return cls(weight_bits, group_size, desc_act, lm_head_quantized)
    
    @classmethod
    def override_quantization_method(cls, hf_quant_cfg,
                                     user_quant) -> Optional[str]:

        is_valid_user_quant = user_quant == "gptq_hpu"

        if  is_valid_user_quant:
            return cls.get_name()

        return None

    def get_quant_method(self, layer: torch.nn.Module,
                         prefix: str) -> Optional["GPTQHPULinearMethod"]:
        if (isinstance(layer, LinearBase) or
            (isinstance(layer, ParallelLMHead) and self.lm_head_quantized)):
            return GPTQHPULinearMethod(self)
        return None

    def get_scaled_act_names(self) -> List[str]:
        return []


class GPTQHPULinearMethod(LinearMethodBase):
    """Linear method for GPTQ.

    Args:
        quant_config: The GPTQ quantization config.
    """

    def __init__(self, quant_config: GPTQHPUConfig):
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        del output_size  # Unused.
        weight_loader = extra_weight_attrs.get("weight_loader")
        if input_size_per_partition % self.quant_config.group_size != 0:
            raise ValueError(
                "The input size is not aligned with the quantized "
                "weight shape. This can be caused by too large "
                "tensor parallel size.")
        output_size_per_partition = sum(output_partition_sizes)
        if (output_size_per_partition % self.quant_config.pack_factor.numerator
                != 0):
            raise ValueError(
                "The output size is not aligned with the quantized "
                "weight shape. This can be caused by too large "
                "tensor parallel size.")

        if self.quant_config.group_size != -1:
            group_size = self.quant_config.group_size
        else:
            group_size = input_size
        scale_and_zero_size = input_size // group_size
        scale_and_zero_input_dim = None

        qweight = PackedvLLMParameter(
            data=torch.empty(
                input_size_per_partition // self.quant_config.pack_factor,
                output_size_per_partition,
                dtype=torch.int32,
            ),
            input_dim=0,
            output_dim=1,
            packed_dim=0,
            packed_factor=self.quant_config.pack_factor,
            weight_loader=weight_loader)

        g_idx = RowvLLMParameter(data=torch.tensor(
            [
                i // self.quant_config.group_size
                for i in range(input_size_per_partition)
            ],
            dtype=torch.int32,
        ),
                                 input_dim=0,
                                 weight_loader=weight_loader)
        qzeros_args = {
            "data":
            torch.empty(
                scale_and_zero_size,
                output_size_per_partition // self.quant_config.pack_factor,
                dtype=torch.int32,
            ),
            "weight_loader":
            weight_loader
        }
        weight_scale_args = {
            "data":
            torch.empty(
                scale_and_zero_size,
                output_size_per_partition,
                dtype=params_dtype,
            ),
            "weight_loader":
            weight_loader
        }
        if scale_and_zero_input_dim is None:
            scales = ChannelQuantScaleParameter(output_dim=1,
                                                **weight_scale_args)
            qzeros = PackedColumnParameter(
                output_dim=1,
                packed_dim=1,
                packed_factor=self.quant_config.pack_factor,
                **qzeros_args)

        layer.register_parameter("qweight", qweight)
        layer.register_parameter("g_idx", g_idx)
        layer.register_parameter("qzeros", qzeros)
        layer.register_parameter("scales", scales)


    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:

        self.wf = torch.tensor(list(range(0, 32, self.quant_config.weight_bits)), dtype=torch.int32).unsqueeze(0)
        weight = self.unpack_weight_from_cuda_old_format(layer)
        layer.qweight.data = self.pack_tensor(weight).to('hpu')

        zeros = self.unpack_zeros_from_cuda_old_format(layer).cpu()
        layer.qzeros.data = self.pack_tensor(zeros).to('hpu')

        # for torch.compile
        layer.qweight = Parameter(layer.qweight.data, requires_grad=False)
        layer.qzeros = Parameter(layer.qzeros.data, requires_grad=False)
        layer.g_idx = Parameter(layer.g_idx.data, requires_grad=False)
        layer.scales = Parameter(layer.scales.data, requires_grad=False)


    def apply(self,
              layer: torch.nn.Module,
              x: torch.Tensor,
              bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        out_shape = x.shape[:-1]
        if hasattr(layer, 'output_size_per_partition'):
            out_shape += (layer.output_size_per_partition , )
        else:
            out_shape += (layer.output_size , )

        reshaped_x = x.reshape(-1, x.shape[-1])

        weight = torch.ops.hpu.convert_from_uint4(layer.qweight,
                                    layer.scales, 
                                    layer.qzeros, 
                                    x.dtype,
                                    layer.g_idx)
        output = torch.matmul(reshaped_x, weight)
        
        if bias is not None:
            output.add_(bias)
        return output.reshape(out_shape)


    def pack_tensor(self, input, bits = 4):
        normal = input.to(torch.int32)
        q = torch.sum(torch.bitwise_left_shift(
            normal.reshape(normal.shape[0], -1, (32 // bits)),
            self.wf.unsqueeze(0)), dim=-1
            ).to(torch.int32)
        
        return q
    
    def unpack_zeros_from_cuda_old_format(self, layer):

        bits = self.quant_config.weight_bits
        zeros = torch.bitwise_right_shift(
            torch.unsqueeze(layer.qzeros.to('cpu'), 2).expand(-1, -1, 32 // bits),
            self.wf.unsqueeze(0),
        ).to(torch.int16 if bits == 8 else torch.int8)

        zeros = zeros + 1
        zeros = torch.bitwise_and(
            zeros, (2**bits) - 1
        ).to(layer.scales.dtype)  # NOTE: It appears that casting here after the `zeros = zeros + 1` is important.
        zeros = zeros.reshape(-1, zeros.shape[1] * zeros.shape[2])
        return zeros

    def unpack_weight_from_cuda_old_format(self, layer):

        qweight = layer.qweight.cpu()
        bits = self.quant_config.weight_bits
        
        weight = torch.bitwise_right_shift(
                torch.unsqueeze(qweight, 1).expand(-1, 32 // bits, -1),
                self.wf.unsqueeze(-1),
            ).to(torch.int16 if bits == 8 else torch.int8)
        weight = torch.bitwise_and(weight, (2**bits) - 1)
        weight = weight.reshape((weight.shape[0]*weight.shape[1], weight.shape[2]))
        return weight
