# FP8 Calibration Procedure

Running inference via [vLLM](https://github.com/vllm-project/vllm) on HPU with FP8 precision is achieved using [Intel® Neural Compressor (INC)](https://docs.habana.ai/en/latest/PyTorch/Inference_on_PyTorch/Quantization/Inference_Using_FP8.html#inference-using-fp8) package. This approach require a model calibration procedure to generate measurements, quantization files, and configurations first. To simplify this process, we've provided the `calibrate_model.sh` script. It requires the following arguments:

- `-m`, i.e., **model stub or path:** Path to your model (if stored locally) or the model ID from the Hugging Face Hub.
- `-d`, i.e., **path to the source dataset:** Path to your dataset in pickle format (".pkl").
- `o`, i.e., **output path:** Path to the directory where the generated measurements, etc., will be stored.

There are also optional arguments, and you can read about them by executing the script with the `-h` option.

The calibration procedure works with any dataset that contains following fields: `system_prompt` and `question`. These fields are used to prepare a calibration dataset with prompts formatted specifically for your model. We recommend to use a public dataset used by MLCommons in Llama2-70b inference submission: https://github.com/mlcommons/inference/tree/master/language/llama2-70b#preprocessed.

> [!TIP]
> For the [DeepSeek-R1](https://huggingface.co/collections/deepseek-ai/deepseek-r1-678e1e131c0169c0bc89728d) series models, which contains 256 experts, it’s important to provide a diverse and 
> sufficiently large sample set to ensure that all experts are properly activated during calibration.
> Through our experiments, we found that using [NeelNanda/pile-10k](https://huggingface.co/datasets/NeelNanda/pile-10k) and selecting 512 samples with at least 1024 tokens each yields good calibration coverage.

## Options and Usage

To run the ```calibrate_model.sh``` script, follow the steps below:

1. Build and install latest [vllm-fork](https://github.com/HabanaAI/vllm-fork/blob/habana_main/README_GAUDI.md#build-and-install-vllm).
2. Clone the vllm-hpu-extension repository and move to the ```calibration``` subdirectory: 

```bash
cd /root
git clone https://github.com/HabanaAI/vllm-hpu-extension.git
cd vllm-hpu-extension/calibration
pip install -r requirements.txt
```
3. Download the dataset.
> [!NOTE]
> For [DeepSeek-R1](https://huggingface.co/collections/deepseek-ai/deepseek-r1-678e1e131c0169c0bc89728d) series models, it is recommended to use `NeelNanda/pile-10k` as the dataset.

4. Run the ```calibrate_model.sh``` script. Refer to the script options and run examples below. The script generates the ```maxabs_quant_g3.json``` file, which is used for FP8 inference.

### Here are some examples of how to use the script:

```bash
./calibrate_model.sh -m /path/to/local/llama3.1/Meta-Llama-3.1-405B-Instruct/ -d dataset-processed.pkl -o /path/to/measurements/vllm-benchmarks/inc -b 128 -t 8 -l 4096
# OR
./calibrate_model.sh -m facebook/opt-125m -d dataset-processed.pkl -o inc/
# OR Calibrate DeepSeek models with dataset NeelNanda/pile-10k
PT_HPU_LAZY_MODE=1 ./calibrate_model.sh -m deepseek-ai/DeepSeek-R1  -d NeelNanda/pile-10k -o inc/ -t 8
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
