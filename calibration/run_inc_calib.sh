#!/bin/bash

DEFAULT_MODEL_PATH="/mnt/disk3/yiliu4/DeepSeek-R1-G2-INC-424-Converter207"
FP8_MODEL_PATH="${1:-$DEFAULT_MODEL_PATH}"

QUANT_CONFIG_FILE="./quant_configs/inc_measure_config.json"
timestamp=$(date +%Y%m%d_%H%M%S)
LOG_FILE="prepare.pile.512.${timestamp}.log"

# remove nc_workspace_measure if needed
if [ -e nc_workspace_measure ]; then
    echo "The directory nc_workspace_measure already exists, removing it..."
    rm -rf nc_workspace_measure
fi


echo "============ QUANT_CONFIG file content ==============="
cat ${QUANT_CONFIG_FILE}
echo "======================================================"



echo "Start INC calibration with model ${FP8_MODEL_PATH}, log file ${LOG_FILE}"

export PT_HPU_LAZY_MODE=1

VLLM_HPU_FORCE_CHANNEL_FP8=0 \
QUANT_CONFIG=${QUANT_CONFIG_FILE} \
    python step-2-measure-scales.py \
    --model ${FP8_MODEL_PATH} \
    --max-tokens 32 \
    --batch-size 1 \
    --block-quant \
    --max-dataset-samples 4 \
    --auto-process-dataset \
    --sample-len 1024 \
    --max-model-len 2048 \
    --verbose \
    --tensor-parallel-size 8 \
    --expert-parallel \
    --dataset "NeelNanda/pile-10k" 2>&1 | tee $LOG_FILE