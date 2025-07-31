###############################################################################
# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
###############################################################################
import argparse
import glob
import json
import os
import re
import sys

import numpy as np


def find_measurement_path(measurement, measurements_dir_path, scales, group_size):
    measurment_card = "_" + measurement + "_" + str(group_size)
    for measurment_file in os.listdir(measurements_dir_path):
        filename = os.fsdecode(measurment_file)
        if not filename.endswith(".json") or "_mod_list" in filename or measurment_card not in filename:
            continue
        if scales:
            if "MAXABS" in filename:
                return os.path.join(measurements_dir_path, measurment_file)
        else:
            if "MAXABS" not in filename:
                return os.path.join(measurements_dir_path, measurment_file)


def is_fused_moe_op(node_name):
    return True if "moe" in node_name.lower() and ".w13_list" not in node_name and ".w2_list" not in node_name else False


def is_moe_experts(node_name):
    # model.layers.3.mlp.experts.moe_op.w13_list.0
    return True if "moe" in node_name.lower() and (".w13_list" in node_name or ".w2_list" in node_name) else False


def get_expert_id(node_name):
    parts = node_name.split(".")
    assert parts[-1].isdigit()
    expert_id = int(parts[-1])
    return expert_id


def get_expert_prefix(node_name):
    parts = node_name.split(".")
    assert parts[-1].isdigit()
    prefix = ".".join(parts[:-1])
    return prefix


def get_local_expert_num(data):
    expert_id = -1
    for mod_name in data:
        if is_moe_experts(mod_name):
            idx = get_expert_id(mod_name)
            expert_id = max(expert_id, idx)
    return expert_id + 1


def unify_measurements(
    measurement_group,
    measurements_dir_path,
    output_path,
    groups_size,
    groups_num,
    local_rank,
    world_size,
    scales=False,
    use_ep=False,
    
):
    measurements_paths = []
    group_name = ""
    # save all the jsons paths in the given measurement group
    for measurement in measurement_group:
        measurement_path = find_measurement_path(measurement, measurements_dir_path, scales, groups_size)
        if measurement_path is not None:
            measurements_paths.append(measurement_path)
        group_name += measurement

    if len(measurements_paths) == 0:
        print("Error: invalid measurement paths. No *.json files or no *mod_list.json files.")
        return
    print(f"got {len(measurements_paths)} measurements for group {group_name} with index {local_rank} out of {groups_num} groups")
    for measurement_path in measurements_paths:
        print(f"measurement path: {measurement_path}")
    # save all the jsons content in the given measurement group
    measurements_jsons = []
    for measurement_path in measurements_paths:
        with open(measurement_path, "r") as f:
            js = json.load(f)
            measurements_jsons.append(js["Nodes"])
    # create a name for the unified json that will be created for this measurement group
    unified_json_name = (
        find_measurement_path(
            measurement_group[0], measurements_dir_path, scales, groups_size)
        .split("/")[-1]
        .replace(
            "_" + measurement_group[0] + "_" + str(groups_size),
            "_" + str(local_rank) + "_" + str(world_size)
        )
    )
    print(f"generate unified json name: {unified_json_name} for group {group_name} with index {local_rank} out of {world_size} ranks")
    unified_json_path = os.path.join(output_path, unified_json_name)

    # Create a replica of the first measurement json file
    with open(measurements_paths[0], "r") as origin, open(unified_json_path, "w") as copy:
        copy.write(origin.read())
    with open(unified_json_path, "r") as json_file:
        unified_json = json.load(json_file)
        unified_json["LocalRank"] = local_rank

    # expert_num is original local_expert_num, it is used only when use_ep is True
    expert_num = get_local_expert_num(unified_json["Nodes"]) if use_ep else -1
    total_experts = expert_num
    # breakpoint()
    # iterate all unified json nodes
    for node_name, node_values in unified_json["Nodes"].items():
        max_outputs = None
        if node_values.get("outputs") is not None:
            max_outputs = node_values["outputs"]

        
        # iterate over all the measurment group and take the maximum for each tensor and its channel

        for idx, measurement_json in enumerate(measurements_jsons):
            if is_fused_moe_op(node_name):
                if use_ep and is_fused_moe_op(node_name):
                    node_res = measurement_json[node_name]["outputs"]
                    node_res_output = node_res[0]
                    node_res_experts_intermediate_amax = node_res[1:]
                    num_intermediate_amax = len(node_res_experts_intermediate_amax)
                    assert num_intermediate_amax == total_experts, f"the number of intermediate amax should be {total_experts}, but got {num_intermediate_amax}"
                    ep_size = world_size
                    ep_rank = local_rank
                    num_local_experts = total_experts // ep_size
                    expert_start_index = ep_rank * num_local_experts
                    expert_end_index = expert_start_index + num_local_experts
                    node_intermediate_amax = node_res_experts_intermediate_amax[expert_start_index:expert_end_index]
                    assert len(node_intermediate_amax) == num_local_experts, f"len(node_intermediate_amax) should be {num_local_experts}, but got {len(node_intermediate_amax)}"
                    max_outputs = [node_res_output, *node_intermediate_amax]
                    print(f"select {len(max_outputs)} outputs for {node_name} ep_rank {ep_rank} with expert_start_index {expert_start_index} and expert_end_index {expert_end_index}")

        if max_outputs is not None:
            if is_fused_moe_op(node_name):
                unified_json["Nodes"][node_name]["outputs"] = max_outputs

    global_rank = None
    local_rank = local_rank
    mode = ""
    layers = {}
    with open(unified_json_path, "w") as json_file:
        json.dump(unified_json, json_file, indent=4)
    mode = unified_json["Mode"]
    nodes = unified_json["Nodes"]

    # create unified npz file from the unified json
    unified_npz_path = os.path.join(
        output_path, unified_json_name.replace(".json", ".npz"))
    for layer, dlayer in nodes.items():
        layers[layer] = {}
        layers[layer]["inputs"] = [np.array(x) for x in dlayer["inputs"]]
        if dlayer.get("outputs") is not None:
            layers[layer]["outputs"] = [np.array(x) for x in dlayer["outputs"]]
        if dlayer.get("params") is not None and dlayer["params"].get("weight") is not None:
            layers[layer]["params"] = {}
            layers[layer]["params"]["weight"] = np.array(
                dlayer["params"]["weight"])
    df = {"GlobalRank": global_rank, "LocalRank": local_rank,
          "Mode": mode, "Nodes": layers}
    with open(unified_npz_path, "w"):
        np.savez(unified_npz_path, df)

import dataclasses

@dataclasses.dataclass
class TempArg:
    measurements: str = None
    rank: int = None
    out: str = None
    use_expert_paral: bool = False
    
    

def parse_args(args):
    parser = argparse.ArgumentParser(
        description="Run the measurements parser", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-m", "--measurements", type=str, help="path to the directory of the measurements that will be unified"
    )
    parser.add_argument(
        "-r",
        "--rank",
        type=int,
        help="rank of unified measurements"
    )
    parser.add_argument(
        "-o",
        "--out",
        type=str,
        default=os.getcwd(),
        help="path to the directory where the unified measurements will be written",
    )
    parser.add_argument(
        "-u",
        "--use_expert_paral",
        action="store_true",
        help="unify original measurement results based on expert parallelism rules",
    )
    return parser.parse_args(args)


def prepare_group_list(measurements_path, rank):
    measure_files = glob.glob(os.path.join(measurements_path, "*_mod_list.json"))
    if len(measure_files) > 0:
        # take original rank=8 as an example, target file name: string_0_8_mod_list.json
        matched = re.match(r"^(\w+)_(\d+)_(\d+)_(\w+)_(\w+)\.json$", os.path.basename(measure_files[0]))
        if matched:
            total_rank = int(matched.group(3))
            # assert (rank < total_rank) and (total_rank % rank) == 0, f"Original total_rank {total_rank} should be larger than your target rank {rank} and be divisible by it"
            group_size = total_rank // rank
            group_list = [[str(i * group_size + j) for j in range(group_size)] for i in range(rank)]
            print("Card grouping list >> {}".format(group_list))
            return group_list
        else:
            raise ValueError("Unrecognized file name!")
    else:
        raise ValueError("*_mod_list.json doesn't exist in {}".format(measurements_path))

def main(args):
    args = parse_args(args)
    output_path = args.out
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    measurements_path = args.measurements
    groups = prepare_group_list(measurements_path, args.rank)

    num_jsons_drange = 0
    num_jsons_scales = 0
    for path in os.listdir(measurements_path):
        if path.endswith(".json"):
            if "MAXABS" in path:
                num_jsons_scales += 1
            elif "mod_list" not in path:
                num_jsons_drange += 1
    assert (
        os.path.isdir(measurements_path)
        and (num_jsons_drange % len(groups)) == 0
        and (num_jsons_scales % len(groups)) == 0
    )
    
    # breakpoint()
    target_world_size = 8
    for ep_rank in range(target_world_size):
        unify_measurements(
            measurement_group=["0"],
            measurements_dir_path=measurements_path,
            output_path=output_path,
            groups_size=1,
            groups_num=1,
            local_rank=ep_rank,
            # group, measurements_path, output_path, num_jsons_drange, len(groups), local_rank,
            scales=False,
            use_ep=args.use_expert_paral,
            world_size=target_world_size
        )


    print("finished measurement unifier script")


if __name__ == "__main__":
    main(sys.argv[1:])


# python expand-mm.py -r 1 -m ./Yi30/inc-woq-2282samples-514-g2-unified-tp1  -o ./Yi30/inc-woq-2282samples-514-g2-unified-tp1-expand-tp8 -u