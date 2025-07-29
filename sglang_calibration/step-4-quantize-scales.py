###############################################################################
# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
###############################################################################
import sglang as sgl
import torch
import argparse
import os

# Set SGLang specific environment variables
os.environ["SGLANG_HPU_SKIP_WARMUP"] = "true"
os.environ["PT_HPU_LAZY_MODE"] = "1"


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)

    args = parser.parse_args()

    llm = sgl.Engine(
        model_path=args.model,
        tp_size=args.tensor_parallel_size,
        dtype=torch.bfloat16,
        quantization="inc",
        disable_radix_cache=True,
        device="hpu",
        trust_remote_code=True,
    )

    llm.shutdown()
