# FP8 Calibration Procedure for VLM models

The model calibration procedure for LLM models is a little bit different than VLM, we need to change it to adapt VLM models. To simplify this process, we've provided the `calibrate_model.sh` script. It requires the following arguments:

- `-m`, i.e., **model stub or path:** Path to your model (if stored locally) or the model ID from the Hugging Face Hub.
- `-d`, i.e., **path to the source dataset:** It's hard-coded to MMMU dataset in VLM models.
- `-o`, i.e., **output path:** Path to the directory where the generated measurements, etc., will be stored.
- `-t`, i.e., **tensor parallel size:** Tensor parallel size to run at.

Here are some examples of how to use the script:

```bash
./calibrate_model.sh \
    -m $MODEL_PATH \
    -o $INC_OUTPUT_PATH \
    -t $TP_SIZE
```
