###############################################################################
# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
###############################################################################
import vllm
import torch
import argparse
import os
os.environ["PT_HPU_WEIGHT_SHARING"] = "0"
os.environ["VLLM_SKIP_WARMUP"] = "true"


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--block-quant", action="store_true", default=False)
    parser.add_argument("--enforce-eager", action="store_true", default=False)
    parser.add_argument("--expert-parallel", action="store_true", default=False)
    parser.add_argument("--max-num-prefill-seqs", type=int, default=None)
    parser.add_argument("--distributed-executor-backend", choices=["mp", "ray"], default="mp", 
                        help="For single node calibration use the default multiprocessing backend. For multi-node calibration use ray backend")

    args = parser.parse_args()

    llm = vllm.LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        enforce_eager=args.enforce_eager,
        dtype=torch.bfloat16,
        quantization="fp8" if args.block_quant else "inc",
        kv_cache_dtype="fp8_inc",
        max_num_prefill_seqs=args.max_num_prefill_seqs,
        max_model_len=128,
        trust_remote_code=True,
        distributed_executor_backend=args.distributed_executor_backend,
        enable_expert_parallel=args.expert_parallel,
    )

    prompts = [
        "Hello, my name is",
        "0.999 compares to 0.9 is ",
        "The capital of France is",
        "The future of AI is",
    ]
    from vllm import LLM, SamplingParams
    # Create a sampling params object.
    sampling_params = SamplingParams(
        temperature=0, max_tokens=32, ignore_eos=True
    )
    import time
    start = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params)
    end = time.perf_counter()
    print(f"e2e took {end - start} seconds")
    gt = None
    for output_i in range(len(outputs)):
        output = outputs[output_i]
        gt_i = None if gt is None else gt[output_i]
        prompt = output.prompt
        generated_text = output.outputs[0].text
        gen_token_id = output.outputs[0].token_ids
        print("====================================")
        print(f"Prompt: {prompt!r}")
        print(f"Generated text: {generated_text!r}")
        print(f"Generated token: {gen_token_id!r}")
        print(f"Ground truth: {gt_i!r}")
        print("====================================")

    # Skip shutdown when VLLM_USE_V1 is set to "1"
    if not os.environ.get("VLLM_USE_V1") or os.environ.get("VLLM_USE_V1") != "1":
        llm.llm_engine.model_executor.shutdown()
