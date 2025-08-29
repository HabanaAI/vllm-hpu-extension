import torch
from safetensors import safe_open
from safetensors.torch import save_file
from glob import glob
import os


import argparse

FP8_MAX = torch.finfo(torch.float8_e4m3fnuz).max


def calc_maxabs_scale(xmaxabs, fullscale, backoff=1.0):
    scale = xmaxabs / (fullscale * backoff)
    return scale


def quant_per_tensor(data):
    amax = (torch.abs(data)).max() + 1e-8
    scale = calc_maxabs_scale(amax, FP8_MAX, 1.0)
    scale = scale.to(data.dtype)
    data_fp8 = data / scale
    cliped_qtensor = torch.clamp(data_fp8, -FP8_MAX, FP8_MAX)
    cliped_qtensor_fp8 = cliped_qtensor.to(torch.float8_e4m3fn)
    return cliped_qtensor_fp8, scale.float()


def copy_other_files(input_path, output_path):
    import shutil

    for file in os.listdir(input_path):
        if file.endswith(".json") or file.endswith(".py") or file.endswith(".tiktoken"):
            print(f"copying {file} to {output_path}")
            shutil.copyfile(
                os.path.join(input_path, file),
                os.path.join(output_path, file),
            )


# Use Case: Hunyuan_V1 Dense and MOE models
def convert_files_per_tensor(input_path, output_path):
    all_safetensors = glob(f"{input_path}/*.safetensors")
    for safetensors_path in all_safetensors:
        print(f"processing {safetensors_path}")
        tensors = {}
        with safe_open(
            safetensors_path, framework="pt", device="cpu"
        ) as tensor_file:
            for k in tensor_file.keys():
                tensor = tensor_file.get_tensor(k)
                if "input_scale" in k:
                    result = (tensor * 448.0 / 240.0).float()
                    tensors.update({k : result})
                elif "weight_scale" in k:
                    weight_name = k.rstrip("_scale")
                    weight_scale_nv = tensor.float()
                    weight_nv = tensor_file.get_tensor(weight_name).float()
                    weight_fp32 = weight_scale_nv * weight_nv
                    weight_fp8, scale = quant_per_tensor(weight_fp32)
                    tensors.update({k: scale})
                    tensors.update({weight_name: weight_fp8})
                elif "proj.weight" in k:
                    continue
                else:
                    tensors.update({k : tensor})
        new_tensor_path = safetensors_path.replace(input_path, output_path)
        print(f"saving to {new_tensor_path}")
        save_file(tensors, new_tensor_path)


def convert_files(input_path, output_path):
    all_safetensors = glob(f"{input_path}/*.safetensors")
    # sort by file name
    all_safetensors.sort()
    for safetensors_path in all_safetensors:
        tensors = {}
        print(f"processing {safetensors_path}")
        with safe_open(
            safetensors_path, framework="pt", device="cpu"
        ) as tensor_file:
            for k in tensor_file.keys():
                tensor = tensor_file.get_tensor(k)
                if "proj" in k:
                    if k.endswith("weight"):
                        tensor = (tensor.float() * 240.0 / 448.0).to(
                            torch.float8_e4m3fn
                        )
                    elif k.endswith("weight_scale") or k.endswith(
                        "input_scale"
                    ):
                        tensor = tensor.float() * 448.0 / 240.0
                    elif k.endswith("weight_scale_inv") or k.endswith(
                        "input_scale_inv"
                    ):
                        # "scale_inv" in deepseek-r1 is actually "scale"
                        tensor = tensor.float() * 448.0 / 240.0
                    elif tensor.dtype == torch.bfloat16:
                        print(f"skip converting {k} as it is bfloat16")
                    else:
                        raise NotImplementedError(f"Cannot covert {k}")
                else:
                    print(f"skip {k}.")
                tensors[k] = tensor
        new_tensor_path = safetensors_path.replace(input_path, output_path)
        print(f"saving to {new_tensor_path}")
        save_file(tensors, new_tensor_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert tensors to float8fnuz format."
    )
    parser.add_argument(
        "-i",
        "--input_path",
        help="Path to the official model weights.",
    )
    parser.add_argument(
        "-o",
        "--output_path",
        help="Path to the output directory.",
    )
    parser.add_argument(
        "-t",
        "--per_tensor",
        action='store_true',
        help="Convert FP8 models using per tensor quantization for both input \
                and weight scales. Default is per channel quantization for weight scale.",
    )
    args = parser.parse_args()
    input_path = args.input_path
    output_path = args.output_path
    per_tensor = args.per_tensor

    # create output directory if it does not exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    copy_other_files(input_path, output_path)
    if per_tensor:
        convert_files_per_tensor(input_path, output_path)
    else:
        convert_files(input_path, output_path)
