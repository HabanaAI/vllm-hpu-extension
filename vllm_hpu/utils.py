from functools import cache
import os
from vllm.utils import make_tensor_with_pad, TORCH_DTYPE_TO_NUMPY_DTYPE
from typing import (TYPE_CHECKING, Any, Callable, Generic, Literal, NamedTuple,
                    Optional, Sequence, Tuple, Type, TypeVar, Union, cast,
                    overload)
import torch
import numpy as np
import numpy.typing as npt
import math

T = TypeVar("T")
U = TypeVar("U")

@cache
def is_fake_hpu() -> bool:
    return os.environ.get('VLLM_USE_FAKE_HPU', '0') != '0'


@cache
def hpu_device_string():
    device_string = 'hpu' if not is_fake_hpu() else 'cpu'
    return device_string


@cache
def hpu_backend_string():
    backend_string = 'hccl' if not is_fake_hpu() else 'gloo'
    return backend_string


def make_ndarray_with_pad_align(
    x: list[list[T]],
    pad: T,
    dtype: npt.DTypeLike,
    *,
    max_len_align: int = 1024,
) -> npt.NDArray:
    """
    Make a padded array from 2D inputs.
    The padding is applied to the end of each inner list until it reaches
    `max_len`.
    """
    # Unlike for most functions, map is faster than a genexpr over `len`
    max_len = max(map(len, x), default=0)
    max_len_aligned = math.ceil(max_len / max_len_align) * max_len_align
    padded_x = np.full((len(x), max_len_aligned), pad, dtype=dtype)

    for ind, blocktb in enumerate(x):
        assert len(blocktb) <= max_len_aligned
        padded_x[ind, :len(blocktb)] = blocktb

    return padded_x


def make_mrope_positions_tensor_with_pad( \
        input_positions: list[list[int]],
        input_mrope_positions: list[list[list[int]]],
        max_prompt_len: int,
        pad: int) -> list[list[int]]:
    # If no mrope positions, returns a flatten (seq_len,)
    if all(mrope_position is None for mrope_position in input_mrope_positions):
        return make_tensor_with_pad(input_positions,
                                    max_len=max_prompt_len,
                                    pad=0,
                                    dtype=torch.long,
                                    device='cpu').flatten()
    # Otherwise, Qwen2.5-VL expects positions in a (3, seq_len)
    # we are going to pad each seq_data in the list
    # using either MRope values or regular position
    mrope_input_positions: list[list[int]] = [[] for _ in range(3)]
    for idx in range(3):
        for b_idx, input_mrope_position in enumerate(input_mrope_positions):
            if input_mrope_position is not None:
                positions = input_mrope_position[idx]
            else:
                positions = input_positions[b_idx]
            padding_size = max_prompt_len - len(positions)
            assert padding_size >= 0
            padded_positions = positions \
                + (max_prompt_len - len(positions)) * [pad]
            mrope_input_positions[idx].extend(padded_positions)
    return torch.tensor(mrope_input_positions, dtype=torch.long, device='cpu')


def make_tensor_with_pad_align(
    x: list[list[T]],
    pad: T,
    dtype: torch.dtype,
    *,
    max_len_align: int = 1024,
    device: Optional[Union[str, torch.device]] = None,
    pin_memory: bool = False,
) -> torch.Tensor:
    """
    Make a padded tensor from 2D inputs.
    The padding is applied to the end of each inner list until it reaches
    max_len_aligned, max_len_aligned is max_len rounding to the nearest 
    `max_len_align`.
    """
    np_dtype = TORCH_DTYPE_TO_NUMPY_DTYPE[dtype]
    padded_x = make_ndarray_with_pad_align(x,
                                           pad,
                                           np_dtype,
                                           max_len_align=max_len_align)

    tensor = torch.from_numpy(padded_x).to(device)
    if pin_memory:
        tensor = tensor.pin_memory()

    return tensor

