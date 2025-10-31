#!/bin/bash
###############################################################################
# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
###############################################################################

set -e
cd "$(dirname "$0")"

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
    echo "  -r    - rank of unified measurements, it should be smaller than original rank number and should be a factor of the original rank number"
    echo "  -u    - use expert parallelism (default: False), expert parallelism unification rule is unique, card 1 expert measurement will be extended to card 0 if unified to x from 2x cards number"
    echo "  -x    - expand measurement files to specific world size (default: not set)"
    echo "  -e    - set this flag to enable enforce_eager execution"
    echo
}

while getopts "m:d:o:b:l:t:r:ux:eh" OPT; do
    case ${OPT} in
        m )
            MODEL_PATH="$OPTARG"
            ;;
        d )
            DATASET_PATH_OR_NAME="$OPTARG"
            ;;
        o )
            FP8_DIR=$(realpath "$OPTARG")
            ;;
        b )
            BATCH_SIZE="$OPTARG"
            ;;
        l )
            LIMIT="$OPTARG"
            ;;
        t )
            TP_SIZE="$OPTARG"
            ;;
        r )
            RANK="$OPTARG"
            ;;
        u )
            USE_EP="true"
            ;;
        x )
            EXPAND_SIZE="$OPTARG"
            ;;
        e )
            ENFORCE_EAGER="true"
            ;;
        h )
            usage
            exit
            ;;
        \? )
            usage
            exit 1
            ;;
    esac
done


if [[ -z "$MODEL_PATH" || -z "$FP8_DIR" || -z "$DATASET_PATH_OR_NAME" ]]; then
    echo "Model stub, source dataset path and output path for fp8 measurements must be provided."
    usage
    exit 1
fi
BATCH_SIZE=${BATCH_SIZE:-"32"}
LIMIT=${LIMIT:-""}
TP_SIZE=${TP_SIZE:-"1"}
RANK=${RANK:-""}
USE_EP=${USE_EP:-""}
ENFORCE_EAGER=${ENFORCE_EAGER:-"false"}

if [[ -n $EXPAND_SIZE && ( -z $RANK || $RANK != 1 ) ]]; then
    echo "EXPAND_SIZE is defined, resetting RANK to 1"
    RANK=1
fi

ALLOWED_DEVICES=("g2" "g3")

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
        block_names="[\"self_attn\", \"lm_head\"]"
    elif [[ $model_name_lower == *"qwen3-vl"* ]]; then
        block_names="[\"lm_head\", \"visual\"]"
    elif [[ $model_name_lower == *"qwen3"* ]]; then
        block_names="[\"lm_head\"]"
    else
        block_names="[]"
    fi
    measure_config=$(cat <<EOF
{
    "method": "HOOKS",
    "mode": "MEASURE",
    "observer": "maxabs",
    "allowlist": {
        "types": [],
        "names": []
    },
    "blocklist": {
        "types": [],
        "names": ${block_names}
    },
    "quantize_weight": false,
    "dump_stats_path": "$1/$2/$3/inc_output"
}
EOF
)
    echo "$measure_config" | jq . > $1/$2/maxabs_measure_$3.json
}

create_quant_config() {
    mkdir -p $1/$2/$3
    
    scale_method="maxabs_hw"
    block_types="[\"Softmax\"]"
    block_names="[\"lm_head\", \"mlp\\\\.gate\\\\b\", \"visual\"]"
    fp8_config="E4M3"
    scale_format="scalar"
    block_types_bf16_attn="[\"VLLMKVCache\", \"Matmul\", \"Softmax\"]"
    block_names_bf16_attn="[\"lm_head\", \"mlp\\\\.gate\\\\b\", \"visual\", \"fused_scaled_dot_product_attention\"]"
    if [[ $model_name_lower == *"deepseek-r1-distill-qwen-7b"* \
            || $model_name_lower == *"qwen2-7b-instruct"* \
            || $model_name_lower == *"qwen2.5-7b-instruct"* ]]; then
        export VLLM_FP32_SOFTMAX=true
        scale_method="unit_scale"
        block_types="$block_types_bf16_attn"
        block_names="$block_names_bf16_attn"
    elif [[ $model_name_lower =~ ^mixtral ]]; then
        scale_format="const"
        block_types="$block_types_bf16_attn"
        block_names="$block_names_bf16_attn"
    elif [[ $model_name_lower == *"deepseek-r1-distill-llama-8b"* ]]; then
        block_types="$block_types_bf16_attn"
        block_names="$block_names_bf16_attn"
    elif [[ $model_name_lower == *"qwen3"* && $model_name_lower != *"qwen3-32b"* ]]; then
        block_types="$block_types_bf16_attn"
        block_names="$block_names_bf16_attn"
        if [[ $model_name_lower == *"qwen3-30b-a3b"* \
            || $model_name_lower == *"qwen3-235b-a22b"* \
            ]]; then
            scale_format="const"
        fi
    fi
    
    if [[ $model_name_lower == *"glm-4."* ]]; then
        scale_format="const"
        block_types="$block_types_bf16_attn"
        block_names="[\"lm_head\", \"mlp\\\\.gate\\\\b\"]"
    fi

    if [[ $scale_format == "const" ]]; then
        export VLLM_DISABLE_MARK_SCALES_AS_CONST=true
    fi

    quant_config=$(cat <<EOF
{
    "mode": "QUANTIZE",
    "observer": "maxabs",
    "scale_method": "${scale_method}",
    "scale_format": "${scale_format}",
    "allowlist": {
        "types": [],
        "names": []
    },
    "blocklist": {
        "types": ${block_types},
        "names": ${block_names}
    },
    "dump_stats_path": "$1/$2/$3/inc_output",
    "fp8_config": "${fp8_config}"
}
EOF
)
    echo "$quant_config" | jq . > $1/$2/maxabs_quant_$3.json
}

extract_last_folder_name() {
    local path="$1"

    path="${path%/}"
    last_folder="$(basename "$path")"
    last_folder="${last_folder,,}"

    echo "$last_folder"
}

cleanup_tmp

EXTRA_FLAGS_STEP_1=""
EXTRA_FLAGS_STEP_2=""
EXTRA_FLAGS_STEP_3=""
EXTRA_FLAGS_STEP_4=""
MULTI_NODE_SETUP="false"

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
    EXTRA_FLAGS_STEP_1+="--chat-template template/llama-2-chat.jinja "
elif  [[ "$MODEL_PATH" == *"Mixtral-8x7B"* ]]; then
    EXTRA_FLAGS_STEP_1+="--chat-template template/mistral_mixtral.jinja "
fi

if [[ -n $LIMIT ]]; then
    EXTRA_FLAGS_STEP_1+="--max-dataset-samples $LIMIT "
fi

SKIP_STEP_1=false
if [[ $DATASET_PATH_OR_NAME == *.pkl ]]; then
    SKIP_STEP_1=false
else
    echo "DATASET_PATH_OR_NAME is not a .pkl file, will prepare calibration dataset based on it."
    SKIP_STEP_1=true
fi

if [[ "$USE_EP" == "true" ]]; then
    EXTRA_FLAGS_STEP_2+="--expert-parallel "
    EXTRA_FLAGS_STEP_4+="--expert-parallel "
fi

if  [[ "$model_name_lower" == *"deepseek"* ]]; then
    EXTRA_FLAGS_STEP_3+="--deepseek "
fi


# Skip step 1 if the DATASET_PATH_OR_NAME is a .pkl file
if $SKIP_STEP_1; then
    EXTRA_FLAGS_STEP_2+="--max-dataset-samples 512 --batch-size 1 --max-tokens 32 "
    EXTRA_FLAGS_STEP_2+="--auto-process-dataset --sample-len 1024 --max-model-len 4096 "
    EXTRA_FLAGS_STEP_2+="--dataset ${DATASET_PATH_OR_NAME} "
fi

if [[ -z "$VLLM_USE_V1" || $VLLM_USE_V1 != "1" ]]; then
    EXTRA_FLAGS_STEP_2+="--max-num-prefill-seqs 1 "
    EXTRA_FLAGS_STEP_4+="--max-num-prefill-seqs 1 "
fi

if $MULTI_NODE_SETUP; then
    cat $FP8_DIR/$MODEL_NAME/maxabs_measure_$DEVICE_TYPE.json > $QUANT_CONFIG
    sleep 2
else
    export QUANT_CONFIG=$FP8_DIR/$MODEL_NAME/maxabs_measure_$DEVICE_TYPE.json
fi

if $ENFORCE_EAGER; then
    EXTRA_FLAGS_STEP_2+="--enforce-eager "
    EXTRA_FLAGS_STEP_4+="--enforce-eager "
fi

if $SKIP_STEP_1; then
    echo "Skipping step 1 - prepare calibration dataset with dataset ${DATASET_PATH_OR_NAME}"
else
    echo ""
    echo "1/4 Preparing calibration dataset"
    python3 step-1-prepare-calibration-dataset.py -m $MODEL_PATH -d $DATASET_PATH_OR_NAME -o $MODEL_NAME $EXTRA_FLAGS_STEP_1 || (echo "Error in step 1" && exit 1)
    echo "Step 1/4 done"
fi

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
if $MULTI_NODE_SETUP; then
    env $EXTRA_ENVS_STEP_4 python3 step-4-quantize-scales.py --model $MODEL_PATH --tensor-parallel-size $TP_SIZE --distributed-executor-backend ray $EXTRA_FLAGS_STEP_4 || (echo "Error in step 4" && exit 1)
else
    env $EXTRA_ENVS_STEP_4 python3 step-4-quantize-scales.py --model $MODEL_PATH --tensor-parallel-size $TP_SIZE $EXTRA_FLAGS_STEP_4 || (echo "Error in step 4" && exit 1)
fi

if [[ -n $RANK ]]; then
    echo ""
    echo "5/5 Unify scales"
    QUANT_DIR=$FP8_DIR/$MODEL_NAME/$DEVICE_TYPE/
    if [[ "$USE_EP" == "true" ]]; then
        EP_ARG="--use_expert_paral"
    fi
    python3 step-5-unify_measurements.py -r $RANK -m $QUANT_DIR -o inc_tmp/$MODEL_NAME/$DEVICE_TYPE/ $EP_ARG || (echo "Error in step 5" && exit 1)
    echo "Step 5/5 done"
fi

if [[ -n $EXPAND_SIZE ]]; then
    echo ""
    echo "Expanding measurements to world size $EXPAND_SIZE"
    QUANT_DIR=$FP8_DIR/$MODEL_NAME/$DEVICE_TYPE/
    python3 step-6-expand-measurements.py -m inc_tmp/$MODEL_NAME/$DEVICE_TYPE -o $QUANT_DIR -w $EXPAND_SIZE  || (echo "Error in expanding measurements" && exit 1)
    echo "Expanding measurements done"
fi

cp inc_tmp/$MODEL_NAME/$DEVICE_TYPE/* $FP8_DIR/$MODEL_NAME/$DEVICE_TYPE/

cleanup_tmp
echo "Calibration process done"
