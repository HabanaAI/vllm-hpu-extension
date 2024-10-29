###############################################################################
# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
###############################################################################
import vllm
import torch
import argparse
import os
os.environ["EXPERIMENTAL_WEIGHT_SHARING"] = "0"
os.environ["VLLM_SKIP_WARMUP"] = "true"


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)

    args = parser.parse_args()

    llm = vllm.LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        enforce_eager=True,
        dtype=torch.bfloat16,
        quantization='inc',
        kv_cache_dtype="fp8_inc")
