# FP8 Calibration Procedure

Running inference via [vLLM](https://github.com/vllm-project/vllm) on HPU with FP8 precision is achieved using [Intel® Neural Compressor (INC)](https://docs.habana.ai/en/latest/PyTorch/Inference_on_PyTorch/Quantization/Inference_Using_FP8.html#inference-using-fp8) package. This approach require a model calibration procedure to generate measurements, quantization files, and configurations first. To simplify this process, we've provided the `calibrate_model.sh` script. It requires the following arguments:

- `-m`, i.e., **model stub or path:** Path to your model (if stored locally) or the model ID from the Hugging Face Hub.
- `-d`, i.e., **path to the source dataset:** Path to your dataset in pickle format (".pkl").
- `o`, i.e., **output path:** Path to the directory where the generated measurements, etc., will be stored.

There are also optional arguments, and you can read about them by executing the script with the `-h` option.

The calibration procedure works with any dataset that contains following fields: `system_prompt` and `question`. These fields are used to prepare a calibration dataset with prompts formatted specifically for your model. We recommend to use a public dataset used by MLCommons in Llama2-70b inference submission: https://github.com/mlcommons/inference/tree/master/language/llama2-70b#preprocessed.

Here are some examples of how to use the script:

```bash
./calibrate_model.sh -m /path/to/local/llama3.1/Meta-Llama-3.1-405B-Instruct/ -d dataset-processed.pkl -o /path/to/measurements/vllm-benchmarks/inc -b 128 -t 8 -l 4096
# OR
./calibrate_model.sh -m facebook/opt-125m -d dataset-processed.pkl -o inc/
```

> [!WARNING] 
> Measurements are device-dependent, so you can't use scales collected on Gaudi3 on Gaudi2 accelerators. This behavior can cause accuracy issues.

> [!TIP]
> If you get following error, ensure you set a valid tensor parallelism value, e.g. `-t 8`:
> ```
> RuntimeError: [Rank:0] FATAL ERROR :: MODULE:PT_DEVMEM Allocation failed for size::939524096 (896)MB
> ```

# Run inference with FP8 models

An inference with FP8 precision models using vLLM has been described in [README_GAUDI](https://github.com/HabanaAI/vllm-fork/blob/habana_main/README_GAUDI.md#quantization-fp8-inference-and-model-calibration-process) file.
