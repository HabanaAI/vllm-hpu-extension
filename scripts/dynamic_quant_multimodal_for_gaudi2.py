import torch
from safetensors import safe_open
from safetensors.torch import save_file
from glob import glob
import os
import json

import argparse

FP8_MAX = 240.0


def calc_maxabs_scale(xmaxabs, fullscale, backoff=1):
    scale = xmaxabs / (fullscale * backoff)
    return scale


def dynamic_quant(data):
    amax = (torch.abs(data)).max(dim=1).values + 1e-8
    scale = calc_maxabs_scale(amax, FP8_MAX, 1.0)
    scale = scale.to(data.dtype)
    data_fp8 = data / scale.unsqueeze(1)
    cliped_qtensor = torch.clamp(data_fp8, -FP8_MAX, FP8_MAX)
    cliped_qtensor_fp8 = cliped_qtensor.to(torch.float8_e4m3fn)
    return cliped_qtensor_fp8, scale.float()


def change_config_json(output_path):
    config_json_file_path = os.path.join(output_path, "config.json")
    with open(config_json_file_path, 'r') as f:
        config_json = json.load(f)
    
    quantization_config = config_json.get("quantization_config", {})
    quantization_config["activation_scheme"]  = "dynamic"
    quantization_config["fmt"]                = "e4m3"
    quantization_config["quant_method"]       = "fp8"
    quantization_config["quant_scheme"]      = "per_channel"

    config_json["quantization_config"] = quantization_config
    with open(config_json_file_path, "w") as f:
        json.dump(config_json, f, indent=2)


def copy_other_files(input_path, output_path):
    import shutil

    for file in os.listdir(input_path):
        if file.endswith(".json") or file.endswith(".py"):
            print(f"copying {file} to {output_path}")
            shutil.copyfile(
                os.path.join(input_path, file),
                os.path.join(output_path, file),
            )
    change_config_json(output_path)


def convert_files(input_path, output_path):
    all_safetensors = glob(f"{input_path}/*.safetensors")
    # sort by file name
    all_safetensors.sort()
    model_list={}

    for safetensors_path in all_safetensors:
        print(f"processing {safetensors_path}")
        tensors = {}
        with safe_open(safetensors_path, framework="pt", device="cpu") as tensor_file:
            for k in tensor_file.keys():
                tensor = tensor_file.get_tensor(k)
                if "proj" in k and 'visual' not in k:
                    weight_fp8, scale = dynamic_quant(tensor)
                    weight_scale_name = k + "_scale"
                    tensors.update({weight_scale_name: scale})
                    tensors.update({k: weight_fp8})
                    model_list.update({weight_scale_name: safetensors_path.split("/")[-1]})
                    model_list.update({k: safetensors_path.split("/")[-1]})
                else:
                    print(f"skip {k}.")
                    tensors.update({k: tensor})
                    model_list.update({k: safetensors_path.split("/")[-1]})
        new_tensor_path = safetensors_path.replace(input_path, output_path)
        save_file(tensors, new_tensor_path)
        print(f"saving to {new_tensor_path}")

    result = {"weight_map" : model_list, "metadata" : {}}
    out_json_path=os.path.join(output_path, "model.safetensors.index.json")
    with open(out_json_path, "w") as f:
        json.dump(result, f, indent=2)
    f.close


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert tensors to float8 format."
    )
    parser.add_argument(
        "-i",
        "--input_path",
        default="/data/Qwen3-VL-30B-A3B-Instruct",
        help="Path to the official model weights.",
    )
    parser.add_argument(
        "-o",
        "--output_path",
        default="/data/Qwen3-VL-30B-A3B-Instruct-FP8-G2-Dynamic",
        help="Path to the output directory.",
    )
    args = parser.parse_args()
    input_path = args.input_path
    output_path = args.output_path

    # create output directory if it does not exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    copy_other_files(input_path, output_path)
    convert_files(input_path, output_path)
