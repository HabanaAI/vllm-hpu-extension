# SGLang Calibration Scripts

This directory contains calibration scripts adapted for SGLang from the original vLLM calibration scripts.


## Usage

```bash
./calibrate_model_sglang.sh -m <model_path> -d <dataset_path> -o <output_path> [options]
```

### Options
- `-m` : Model path (required)
- `-d` : Dataset path (required) 
- `-o` : Output directory for measurements (required)
- `-b` : Batch size (default: 32)
- `-l` : Limit number of samples
- `-t` : Tensor parallel size (default: 1)
- `-g` : Card groups for unification (skipped as SGLang support 1 card quantization currently)

## Files

- `calibrate_model_sglang.sh` : Main calibration script
- `step-0-detect-device.py` : Device detection (same as vLLM)
- `step-1-prepare-calibration-dataset.py` : Dataset preparation with SGLang env vars
- `step-2-measure-scales.py` : Scale measurement using SGLang engine
- `step-3-postprocess-measure.py` : It fixes inconsistencies in attention related operations and cache tensors including deepseek mla ops and tensors  (same as vLLM)
- `step-4-quantize-scales.py` : Verifies if the measured scales can be used in the model or not.
- `step-5-unify_measurements.py` : Measurement unification when multiple cards are involved (same as vLLM) (skipped as SGLang support 1 card quantization currently)