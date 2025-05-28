# FP8 Calibration Procedure

Running inference via [vLLM](https://github.com/vllm-project/vllm) on HPU with FP8 precision is achieved using [Intel® Neural Compressor (INC)](https://docs.habana.ai/en/latest/PyTorch/Inference_on_PyTorch/Quantization/Inference_Using_FP8.html#inference-using-fp8) package. This approach requires a model calibration procedure to generate measurements, quantization files, and configurations first. To simplify this process, we've provided the `calibrate_model.sh` script. It requires the following arguments:

- `-m`, i.e., **model stub or path:** Path to your model (if stored locally) or the model ID from the Hugging Face Hub.
- `-d`, i.e., **path to the source dataset:** Path to your dataset in pickle format (".pkl").
- `-o`, i.e., **output path:** Path to the directory where the generated measurements, etc., will be stored.

There are also optional arguments, and you can read about them by executing the script with the `-h` option.

The calibration procedure works with any dataset that contains following fields: `system_prompt` and `question`. These fields are used to prepare a calibration dataset with prompts formatted specifically for your model. We recommend to use a public dataset used by MLCommons in Llama2-70b inference submission: https://github.com/mlcommons/inference/tree/master/language/llama2-70b#preprocessed.

## Options and Usage

To run the ```calibrate_model.sh``` script, follow the steps below:

1. Build and install latest [vllm-fork](https://github.com/HabanaAI/vllm-fork/blob/habana_main/README_GAUDI.md#build-and-install-vllm).
2. Clone the vllm-hpu-extension repository and move to the ```calibration``` subdirectory: 

```bash
cd /root
git clone https://github.com/HabanaAI/vllm-hpu-extension.git
cd vllm-hpu-extension/calibration
```
3. Download and process the dataset .pkl file by using the ```download_dataset.sh``` script.

4. Run the ```calibrate_model.sh``` script. Refer to the script options and run examples below. The script generates the ```maxabs_quant_g3.json``` file, which is used for FP8 inference.

### Here are some examples of how to use the script:

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

# Multi-node FP8 Calibration 

Following section details the procedure for calibrating models that do not fit into a single Gaudi node. For illustration we have used the Llama 3.1 405B model running in Tensor Parallelism(TP)-16 mode spanning two Gaudi2 nodes.<br>

> [!NOTE] 
> Following steps are to be executed within a [Gaudi Pytorch container](https://docs.habana.ai/en/latest/Installation_Guide/Additional_Installation/Docker_Installation.html#use-intel-gaudi-containers)


#### Step 1: Pre-requisites

- Install latest [vllm-fork](https://github.com/HabanaAI/vllm-fork/blob/habana_main/README_GAUDI.md#build-and-install-vllm)
- Ensure that all nodes in the multi-node setup are connected to an NFS mount (Network File System).
- Create workspace directory on NFS, clone the calibration scripts repo and create an empty file `quant_config_buffer.json`.
```bash
mkdir <nfs-mount-path>/my_workspace && cd <nfs-mount-path>/my_workspace
git clone https://github.com/HabanaAI/vllm-hpu-extension.git && cd vllm-hpu-extension/calibration
touch quant_config_buffer.json 
```
- Check if all Gaudi NIC ports are up <br>
Note : Following commands should be run on the host and NOT inside the container. <br>
```bash
cd /opt/habanalabs/qual/gaudi2/bin 
./manage_network_ifs.sh --status 
# All the ports should be in 'up' state. Try flipping the state
./manage_network_ifs.sh --down 
./manage_network_ifs.sh --up
# Give it a minute for the NIC's to flip and check the status again
```
- Set following envs at all nodes:
```bash
# Check the network interface for outbound/inbound comms. Command 'ip a' or 'ifconfig' should list all the interfaces
export GLOO_SOCKET_IFNAME=eth0
export HCCL_SOCKET_IFNAME=eth0
export QUANT_CONFIG="<nfs-path-to-config>/quant_config_buffer.json"
```


#### Step 2: Start a Ray cluster to accommodate the required TP size.

```bash
# Start Ray on head node
ray start --head --port=6379

# Add worker nodes to the Ray cluster
ray start --address='<ip-of-ray-head-node>:6379'

# Check if the cluster has required number of HPU's
ray status
```

#### Step 3: Run model calibration script

```bash
./calibrate_model.sh -m meta-llama/Llama-3.1-405B-Instruct -d <path-to-dataset>/open_orca_gpt4_tokenized_llama.calibration_1000.pkl -o <nfs-path-to-calibration-output>/fp8_output -l 4096 -t 16 -b 128
```
Running the above command will create calibration measurement files in the specified output directory, organized into model-specific subdirectories.

> [!NOTE] 
> The current calibration procedure works correctly only when the multi-node configuration has more than 8 cards.


#### Step 4: (Optional) Measurement unification

This is an optional step and is used to reduce the target tensor parallelism level by unifying the measurement scales. For example, you can perform FP8 calibration on the Llama 3.1 405B model using 2x Gaudi2 nodes with Tensor Parallelism (TP) set to 16, and then use the unification script to reduce the TP to 8. This can be achieved in two ways: 
1. Add `-g` optional parameter to `calibration_model.sh` script, e.g.
```bash
./calibrate_model.sh -m meta-llama/Llama-3.1-405B-Instruct -d <path-to-dataset>/open_orca_gpt4_tokenized_llama.calibration_1000.pkl -o <nfs-path-to-calibration-output>/fp8_output -l 4096 -t 16 -b 128 -g "0,8--1,9--2,10--3,11--4,12--5,13--6,14--7,15"
```
2. If calibration has already been performed, use the following command to convert existing scales:
```bash
python3 step-5-unify_measurements.py -g "0,8--1,9--2,10--3,11--4,12--5,13--6,14--7,15"  -m <nfs-path-to-calibration-output>/fp8_output/llama-3.1-405b-instruct/g2/ -o <nfs-path-to-calibration-output>/fp8_output/llama-3.1-405b-instruct/g2/
```
-  `-g`, i.e. **card grouping** to use during unification. Card indices separated by commas and groups separated by double dashes.
-  `-m`, i.e. **calibration output path** containing the measurement files.
-  `-o`, i.e. **unification output directory** where unification output will be written.

> [!TIP]
> It is a good practice to store unification results in the source directory. This allows you to run the vLLM server with FP8 precision and different TP values without modifying the directory specified in the `QUANT_CONFIG` environment variable.

Below examples in case you want to convert scales from TP=16 to TP=4 and 2:
- conversion of scales TP=16 -> TP=4:
```bash
python3 step-5-unify_measurements.py -g "0,8,1,9--2,10,3,11--4,12,5,13--6,14,7,15"  -m <nfs-path-to-calibration-output>/fp8_output/llama-3.1-405b-instruct/g2/ -o <nfs-path-to-calibration-output>/fp8_output/llama-3.1-405b-instruct/g2/
```
- conversion of scales TP=16 -> TP=2:
```bash
python3 step-5-unify_measurements.py -g "0,8,1,9,2,10,3,11--4,12,5,13,6,14,7,15"  -m <nfs-path-to-calibration-output>/fp8_output/llama-3.1-405b-instruct/g2/ -o <nfs-path-to-calibration-output>/fp8_output/llama-3.1-405b-instruct/g2/
```


#### Step 5: Serving the FP8 quantized model

```bash
export QUANT_CONFIG='<nfs-path-to-calibration-output>/fp8_output/llama-3.1-405b-instruct/maxabs_quant_g2.json'
vllm serve meta-llama/Llama-3.1-405B-Instruct --quantization inc --kv-cache-dtype fp8_inc --weights-load-device cpu --tensor-parallel-size 8 --max-model-len 2048
```

> [!NOTE] 
> Detailed information about serving with vLLM (including multi-node serving) you can find in [README_GAUDI](https://github.com/HabanaAI/vllm-fork/blob/habana_main/README_GAUDI.md) within vllm-fork repo.