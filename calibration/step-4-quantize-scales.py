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
    parser.add_argument("--distributed-executor-backend", type=str, default="mp", 
                        help="For single node calibration use the default multiprocessing backend. For multi-node calibration use ray backend")

    args = parser.parse_args()

    hpu_lazy_mode = os.environ.get('PT_HPU_LAZY_MODE', '1')
    enforce_eager = True if hpu_lazy_mode == '1' else False

    llm = vllm.LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        enforce_eager=enforce_eager,
        dtype=torch.bfloat16,
        quantization='inc',
        kv_cache_dtype="fp8_inc",
        max_num_prefill_seqs=1,
        trust_remote_code=True,
        distributed_executor_backend=args.distributed_executor_backend,
    )

    llm.llm_engine.model_executor.shutdown()
