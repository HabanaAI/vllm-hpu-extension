###############################################################################
# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
###############################################################################
import sglang as sgl
import torch
import pandas as pd
import time
import argparse
import os

# Set SGLang specific environment variables
os.environ["SGLANG_HPU_SKIP_WARMUP"] = "true"
os.environ["PT_HPU_LAZY_MODE"] = "1"


def get_ds(args):
    print(f"Loading dataset: {args.dataset}")
    ds = pd.read_pickle(args.dataset)

    if args.max_dataset_samples:
        ds = ds.head(args.max_dataset_samples)

    return ds


def generate_responses(llm, input_batch, args, sampling_params):
    responses = llm.generate(input_batch, sampling_params)

    total_input_tokens = 0
    total_generated_tokens = 0

    for i, response in enumerate(responses):
        if args.verbose:
            print(f"Prompt: {input_batch[i]};\nAnswer: {response['text']}\n")
        # Note: SGLang response format might be different, adjust as needed
        if 'text' in response:
            total_generated_tokens += len(response['text'].split())  # Rough token count


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, required=True,)
    parser.add_argument("-m", "--model", type=str, required=True,)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-dataset-samples", type=int, default=0)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    calibration_ds = get_ds(args)

    llm = sgl.Engine(
        model_path=args.model,
        dtype=torch.bfloat16,
        quantization="inc",
        tp_size=args.tensor_parallel_size,
        disable_radix_cache=True,
        device="hpu",
        trust_remote_code=True,
    )

    sampling_params = {
        "temperature": 0.0,
        "top_p": 1,
        "max_new_tokens": 1024
    }

    input_batch = []
    dataset_len = len(calibration_ds)
    batch_num = dataset_len // args.batch_size if dataset_len % args.batch_size == 0 else (
        dataset_len // args.batch_size) + 1
    batch_done = 0
    for i, (_, row) in enumerate(calibration_ds.iterrows()):
        input_batch.append(row["input"])
        if i and i % args.batch_size == 0:
            t_start = time.perf_counter()
            generate_responses(llm, input_batch, args, sampling_params)
            t_end = time.perf_counter()
            batch_done += 1
            print(
                f"Batch finished: {i}/{calibration_ds.shape[0]} samples done; ETA: {int((t_end - t_start) * (batch_num - batch_done) // 60)} min")
            input_batch = []
    
    generate_responses(llm, input_batch, args, sampling_params)
    print(
        f"Last batch finished: {i + 1}/{calibration_ds.shape[0]} samples done")

    llm.shutdown()
