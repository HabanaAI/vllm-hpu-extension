###############################################################################
# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
###############################################################################
import vllm
import torch
import pandas as pd
import time
import argparse
import os
os.environ["EXPERIMENTAL_WEIGHT_SHARING"] = "0"
os.environ["VLLM_SKIP_WARMUP"] = "true"


def get_ds(args):
    print(f"Loading dataset: {args.dataset}")
    ds = pd.read_pickle(args.dataset)

    if args.max_dataset_samples:
        ds = ds.head(args.max_dataset_samples)

    return ds


def generate_responses(llm, input_batch, args):
    responses = llm.generate(input_batch, sampling_params, use_tqdm=True)

    total_input_tokens = 0
    total_generated_tokens = 0

    for response in responses:
        if args.verbose:
            print(
                f"Prompt: {response.prompt};\nAnswer: {response.outputs[0].text}\n")
        total_input_tokens += len(response.prompt_token_ids)
        total_generated_tokens += len(response.outputs[0].token_ids)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, required=True)
    parser.add_argument("-m", "--model", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-dataset-samples", type=int, default=0)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    calibration_ds = get_ds(args)

    llm = vllm.LLM(
        model=args.model,
        dtype=torch.bfloat16,
        quantization='inc',
        max_num_seqs=args.batch_size,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len
    )

    sampling_params = vllm.SamplingParams(
        temperature=0.0,
        top_p=1,
        max_tokens=1024)

    input_batch = []
    dataset_len = len(calibration_ds)
    batch_num = dataset_len // args.batch_size if dataset_len % args.batch_size == 0 else (
        dataset_len // args.batch_size) + 1
    batch_done = 0
    for i, (_, row) in enumerate(calibration_ds.iterrows()):
        input_batch.append(row["input"])
        if i and i % args.batch_size == 0:
            t_start = time.perf_counter()
            generate_responses(llm, input_batch, args)
            t_end = time.perf_counter()
            batch_done += 1
            print(
                f"Batch finished: {i}/{calibration_ds.shape[0]} samples done; ETA: {int((t_end - t_start) * (batch_num - batch_done) // 60)} min")
            input_batch = []
    generate_responses(llm, input_batch, args)
    print(
        f"Last batch finished: {i + 1}/{calibration_ds.shape[0]} samples done")

    llm.finish_measurements()
