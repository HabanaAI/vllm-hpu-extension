#!/bin/bash
###############################################################################
# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
###############################################################################

set -e
cd "$(dirname "$0")"

ALLOWED_DEVICES=("g2" "g3")

usage() {
    echo
    echo "Calibrate given MODEL_PATH for FP8 inference"
    echo
    echo "usage: ${0} <options>"
    echo
    echo "  -m    - [required] huggingface stub or local directory of the MODEL_PATH"
    echo "  -d    - [required] path to source dataset (details in README)"
    echo "  -o    - [required] path to output directory for fp8 measurements"
    echo "  -b    - batch size to run the measurements at (default: 32)"
    echo "  -l    - limit number of samples in calibration dataset"
    echo "  -t    - tensor parallel size to run at (default: 1); NOTE: if t > 8 then we need a multi-node setup"
    echo "  -g    - groups of cards we want to unify. Card indices seperated by commas and groups seperated by double dash '--', e.g. 0,1--2,3--4,5--6,7 card 0 measurement will be unified with card 1 measurement and so on."
    echo
}

cleanup_tmp() {
	if [[ $(pwd) == *vllm-hpu-extension/calibration ]]; then
		echo "Clearing temporary directory"
		rm -rf nc_workspace
		rm -rf inc_tmp
	else
		echo "Skipping temporary directory removal"
	fi
}

create_measure_config() {
    mkdir -p $1/$2/$3

    model_name_lower=$(echo "$2" | tr '[:upper:]' '[:lower:]')

    if [[ $model_name_lower =~ ^mixtral ]]; then
        tmp_config="{\"method\": \"HOOKS\",\"mode\": \"MEASURE\",\"observer\": \"maxabs\",\"allowlist\": {\"types\": [], \"names\":  []},\"blocklist\": {\"types\": [], \"names\":  [\"self_attn\", \"lm_head\"]},\"quantize_weight\": false,\"dump_stats_path\": \"$1/$2/$3/inc_output\"}"
    elif [[ $model_name_lower =~ ^deepseek ]]; then
        tmp_config="{\"method\": \"HOOKS\",\"mode\": \"MEASURE\",\"observer\": \"maxabs\",\"allowlist\": {\"types\": [], \"names\":  []},\"blocklist\": {\"types\": [], \"names\":  [\"lm_head\", \"mlp\\\.gate\\\b\", \"block2batch_matmul\"]},\"quantize_weight\": false,\"dump_stats_path\": \"$1/$2/$3/inc_output\"}"
    else
        tmp_config="{\"method\": \"HOOKS\",\"mode\": \"MEASURE\",\"observer\": \"maxabs\",\"allowlist\": {\"types\": [], \"names\":  []},\"blocklist\": {\"types\": [], \"names\":  []},\"quantize_weight\": false,\"dump_stats_path\": \"$1/$2/$3/inc_output\"}"
    fi
    echo "$tmp_config" > $1/$2/maxabs_measure_$3.json
}

create_quant_config() {
    mkdir -p $1/$2/$3
    
    model_name_lower=$(echo "$2" | tr '[:upper:]' '[:lower:]')

    #note(kwisniewski98): mixtral models has attention masked to not cause regression in accuracy
    if [[ $model_name_lower =~ ^mixtral ]]; then
        tmp_config="{\"mode\": \"QUANTIZE\",\"observer\": \"maxabs\",\"scale_method\": \"maxabs_hw\",\"allowlist\": {\"types\": [],\"names\": []},\"blocklist\": {\"types\": [],\"names\": [\"self_attn\", \"lm_head\"]},\"dump_stats_path\": \"$1/$2/$3/inc_output\"}"
    elif [[ $model_name_lower =~ ^deepseek ]]; then
        tmp_config="{\"mode\": \"QUANTIZE\",\"observer\": \"maxabs\",\"scale_method\": \"maxabs_hw\", \"scale_format\": \"scalar\", \"allowlist\": {\"types\": [],\"names\": []},\"blocklist\": {\"types\": [],\"names\": [\"lm_head\", \"mlp\\\.gate\\\b\", \"block2batch_matmul\"]},\"dump_stats_path\": \"$1/$2/$3/inc_output\"}"
    else
        tmp_config="{\"mode\": \"QUANTIZE\",\"observer\": \"maxabs\",\"scale_method\": \"maxabs_hw\",\"allowlist\": {\"types\": [],\"names\": []},\"blocklist\": {\"types\": [],\"names\": []},\"dump_stats_path\": \"$1/$2/$3/inc_output\"}"
    fi
    echo "$tmp_config" > $1/$2/maxabs_quant_$3.json
}

extract_last_folder_name() {
    local path="$1"

    path="${path%/}"
    last_folder="$(basename "$path")"
    last_folder="${last_folder,,}"

    echo "$last_folder"
}

cleanup_tmp

EXTRA_FLAGS=""
BATCH_SIZE=32
TP_SIZE=1
MULTI_NODE_SETUP=false
while getopts "m:b:l:t:d:h:o:g:" OPT; do
    case ${OPT} in
        m )
            MODEL_PATH="$OPTARG"
            ;;
        d )
            DATASET_PATH="$OPTARG"
            ;;
        b )
            BATCH_SIZE="$OPTARG"
            ;;
        o )
            FP8_DIR=$(realpath "$OPTARG")
            ;;
        l )
            LIMIT="$OPTARG"
            ;;
        t )
            TP_SIZE="$OPTARG"
            ;;
        g )
            CARD_GROUPS="$OPTARG"
            ;;
        h )
            usage
            ;;
        \? )
            usage
            exit 1
            ;;
    esac
done

if [[ -z "$MODEL_PATH" && -z "$FP8_DIR" && -z "$DATASET_PATH" ]]; then
    echo "Model stub, source dataset path and output path for fp8 measurements must be provided."
    usage
    exit 1
fi

# Store the provided MODEL_PATH name in a variable
MODEL_NAME=$(extract_last_folder_name "$MODEL_PATH")
model_name_lower=$(echo "$MODEL_NAME" | tr '[:upper:]' '[:lower:]')

echo "Step 0 - detecting used device type [g2, g3]"
DEVICE_TYPE=$(python3 step-0-detect-device.py) || (echo "Detecting device process failed" && exit 1)
DEVICE_TYPE="g$DEVICE_TYPE"
echo "Detected device type: $DEVICE_TYPE"
echo "Step 0 done"

# Check if the provided device type is valid
if [[ ! " ${ALLOWED_DEVICES[*]} " =~ " $DEVICE_TYPE " ]]; then
    echo "Invalid device type: $DEVICE_TYPE. Allowed devices: ${ALLOWED_DEVICES[*]}"
    exit 1
fi

if [[ $TP_SIZE -gt 8 ]]; then
    MULTI_NODE_SETUP=true
fi

if $MULTI_NODE_SETUP; then
    RAY_AVAILABLE_RESOURCES=$(python3 -c 'import ray; ray.init(); print(int(ray.available_resources()["HPU"]))')
    if [[ $RAY_AVAILABLE_RESOURCES -lt $TP_SIZE ]]; then
        echo "Required TP size : $TP_SIZE" 
        echo "Available HPU's : $RAY_AVAILABLE_RESOURCES "
        echo "!! Exiting since not enough HPU resources available. You can run 'ray status' to see available resources"
        echo "Refer https://github.com/HabanaAI/vllm-hpu-extension/tree/main/calibration#experimental-multi-node-fp8-calibration for multi-node runs"
        exit 1
    fi

    if [[ ! -e $QUANT_CONFIG ]]; then
        echo " !! Exiting. Invalid QUANT_CONFIG env"
        echo " Multi-node calibration requires QUANT_CONFIG to point to an empty buffer.json file. Refer https://github.com/HabanaAI/vllm-hpu-extension/tree/main/calibration#experimental-multi-node-fp8-calibration"
        exit 1
    fi
fi

create_measure_config $FP8_DIR $MODEL_NAME $DEVICE_TYPE
create_quant_config $FP8_DIR $MODEL_NAME $DEVICE_TYPE

if [[ $TP_SIZE > 1 ]]; then
    export PT_HPU_ENABLE_LAZY_COLLECTIVES=true
fi

if [[ $MODEL_PATH_NAME == llama.*2.* ]]; then
    EXTRA_FLAGS+="--chat-template template/llama-2-chat.jinja "
elif  [[ "$MODEL_PATH" == *"Mixtral-8x7B"* ]]; then
    EXTRA_FLAGS+="--chat-template template/mistral_mixtral.jinja "
fi

if [[ -n $LIMIT ]]; then
    EXTRA_FLAGS+="--max-dataset-samples $LIMIT "
fi

if  [[ "$model_name_lower" == *"deepseek"* ]]; then
    EXTRA_FLAGS_STEP_2="--block-quant "
    EXTRA_ENVS_STEP_2="VLLM_REQUANT_FP8_INC=1 VLLM_ENABLE_RUNTIME_DEQUANT=1 VLLM_MLA_DISABLE_REQUANTIZATION=1 VLLM_MOE_N_SLICE=1 VLLM_EP_SIZE=8"
    EXTRA_FLAGS_STEP_3="--deepseek "
    EXTRA_FLAGS_STEP_4="--block-quant "
fi
if $MULTI_NODE_SETUP; then
    cat $FP8_DIR/$MODEL_NAME/maxabs_measure_$DEVICE_TYPE.json > $QUANT_CONFIG
    sleep 2
else
    export QUANT_CONFIG=$FP8_DIR/$MODEL_NAME/maxabs_measure_$DEVICE_TYPE.json
fi

echo ""
echo "1/4 Preparing calibration dataset"
python3 step-1-prepare-calibration-dataset.py -m $MODEL_PATH -d $DATASET_PATH -o $MODEL_NAME $EXTRA_FLAGS || (echo "Error in step 1" && exit 1)
echo "Step 1/4 done"

echo ""
echo "2/4 Measuring scales"
if $MULTI_NODE_SETUP; then
    env $EXTRA_ENVS_STEP_2 python3 step-2-measure-scales.py -m $MODEL_PATH --tensor-parallel-size $TP_SIZE -d $MODEL_NAME-calibration-dataset.pkl --batch-size $BATCH_SIZE --distributed-executor-backend ray  $EXTRA_FLAGS_STEP_2 || (echo "Error in step 2" && exit 1)
else
    env $EXTRA_ENVS_STEP_2 python3 step-2-measure-scales.py -m $MODEL_PATH --tensor-parallel-size $TP_SIZE -d $MODEL_NAME-calibration-dataset.pkl --batch-size $BATCH_SIZE $EXTRA_FLAGS_STEP_2 || (echo "Error in step 2" && exit 1)
fi
echo "Step 2/4 done"

echo ""
echo "3/4 Postprocessing scales"
python3 step-3-postprocess-measure.py -m $FP8_DIR/$MODEL_NAME/$DEVICE_TYPE/ -o inc_tmp/$MODEL_NAME/$DEVICE_TYPE/  $EXTRA_FLAGS_STEP_3 || (echo "Error in step 3" && exit 1)
cp inc_tmp/$MODEL_NAME/$DEVICE_TYPE/* $FP8_DIR/$MODEL_NAME/$DEVICE_TYPE/
echo "Step 3/4 done"


if $MULTI_NODE_SETUP; then
    cat $FP8_DIR/$MODEL_NAME/maxabs_quant_$DEVICE_TYPE.json > $QUANT_CONFIG
    sleep 2
else
    export QUANT_CONFIG=$FP8_DIR/$MODEL_NAME/maxabs_quant_$DEVICE_TYPE.json
fi

echo ""
echo "4/4 Quantize scales"
if $MULTI_NODE_RUN; then
    python3 step-4-quantize-scales.py --model $MODEL_PATH --tensor-parallel-size $TP_SIZE --distributed-executor-backend ray $EXTRA_FLAGS_STEP_4 || (echo "Error in step 4" && exit 1)
else
    python3 step-4-quantize-scales.py --model $MODEL_PATH --tensor-parallel-size $TP_SIZE $EXTRA_FLAGS_STEP_4 || (echo "Error in step 4" && exit 1)
fi

if [[ -n $CARD_GROUPS ]]; then
    echo ""
    echo "5/5 Unify scales"
    QUANT_DIR=$FP8_DIR/$MODEL_NAME/$DEVICE_TYPE/
    python3 step-5-unify_measurements.py -g "$CARD_GROUPS" -m $QUANT_DIR -o $QUANT_DIR || (echo "Error in step 5" && exit 1)
    echo "Step 5/5 done"
fi
cleanup_tmp
echo "Calibration process done"