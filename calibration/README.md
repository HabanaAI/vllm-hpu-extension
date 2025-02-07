# FP8 Calibration Procedure

Running inference via [vLLM](https://github.com/vllm-project/vllm) on HPU with FP8 precision is achieved using [Intel® Neural Compressor (INC)](https://docs.habana.ai/en/latest/PyTorch/Inference_on_PyTorch/Quantization/Inference_Using_FP8.html#inference-using-fp8) package. This approach requires a model calibration procedure to generate measurements, quantization files, and configurations first. To simplify this process, we've provided the `calibrate_model.sh` script. It requires the following arguments:

- `-m`, i.e., **model stub or path:** Path to your model (if stored locally) or the model ID from the Hugging Face Hub.
- `-d`, i.e., **path to the source dataset:** Path to your dataset in pickle format (".pkl").
- `o`, i.e., **output path:** Path to the directory where the generated measurements, etc., will be stored.

There are also optional arguments, and you can read about them by executing the script with the `-h` option.

The calibration procedure works with any dataset that contains following fields: `system_prompt` and `question`. These fields are used to prepare a calibration dataset with prompts formatted specifically for your model. We recommend to use a public dataset used by MLCommons in Llama2-70b inference submission: https://github.com/mlcommons/inference/tree/master/language/llama2-70b#preprocessed.

Here are some examples of how to use the script:

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

> [!WARNING] 
> !! Mutli-node calibration is an experimental feature and could have stability issues.

Following section details the procedure for calibrating models that do not fit into a single Gaudi node. For illustration we have used the Llama 3.1 405B model running in Tensor Parallelism(TP)-16 mode spanning two Guadi2 nodes.<br>
Note : Following steps are to be executed within a [Gaudi Pytorch container](https://docs.habana.ai/en/latest/Installation_Guide/Additional_Installation/Docker_Installation.html#use-intel-gaudi-containers)

#### Step 1: Pre-requisites
  - Install latest [vllm-fork](https://github.com/HabanaAI/vllm-fork/blob/habana_main/README_GAUDI.md#build-and-install-vllm)
  - Ensure that all nodes in the multi-node setup are connected to an NFS mount (Network File System).
  - Create workspace directory on NFS, clone the calibration scripts repo and create an empty file 'quant_config_buffer.json'.
    ```bash
    mkdir <nfs-mount-path>/my_workspace && cd <nfs-mount-path>/my_workspace
    git clone https://github.com/HabanaAI/vllm-hpu-extension.git && cd vllm-hpu-extension/calibration
    touch quant_config_buffer.json 
    ```

#### Step 2: Start a Ray cluster to accommodate the required TP size. 
```bash
# Export the required env variables separately on all nodes.
export PT_HPU_ENABLE_LAZY_COLLECTIVES=true
export EXPERIMENTAL_WEIGHT_SHARING="0"
export VLLM_SKIP_WARMUP="true"
# Check the network interface for outbound/inbound comms. Command 'ip a' or 'ifconfig' should list all the interfaces
export GLOO_SOCKET_IFNAME=eth0
export QUANT_CONFIG="<path-to-config>/quant_config_buffer.json"

# Start Ray on head node
ray start --head --port=6379

# Add worker nodes to the Ray cluster
ray start --address='<ip-of-ray-head-node>:6379'

# Check if the cluster has required number of HPU's
ray status
```

#### Step 3: Run model calibration script
```bash
./calibrate_model.sh -m meta-llama/Llama-3.1-405B-Instruct -d <path-to-dataset>/open_orca_gpt4_tokenized_llama.calibration_1000.pkl -o <path-to-calibration-output>/fp8_output -l 10 -t 16 -b 1
```
Running the above command should create the calibration measurement files in the specified output directory with model specific sub-directories.<br>
<details><summary>Arguments used</summary>
-m for model-id/path<br> 
-d dataset pickle path<br> 
-o output directory on nfs<br> 
-l limit number of data samples used for calibration to the specified value<br> 
-t tensor parallelism<br> 
-b batch_size for calibration<br> </details>

<details><summary>Common issues</summary> 
  
  1. Facing error "nic/port is down". <br>
  This happens when the Gaudi card nic ports are down. On every node check the port status as below.<br>
  Note : Following commands should be run on the host and NOT inside the container. <br>
     
```bash
cd /opt/habanalabs/qual/gaudi2/bin 
./manage_network_ifs.sh --status 
# All the ports should be in 'up' state. Try flipping the state
./manage_network_ifs.sh --down 
./manage_network_ifs.sh --up 
```  
  </details>


#### Step 4: (optional) Measurement unification <p>
This is an optional step and is used to reduce the target tensor parallelism level by unifying the measurement scales.<br> For eg: You can perform FP8 calibration on the Llama 3.1 405B model on 2x Gaudi2 nodes with Tensor Parallelism = 16 and then use the unification script to reduce the TP to 8. Refer sample command below
```bash
python step-5-unify_measurements.py -g "0,1--2,3--4,5--6,7--8,9--10,11--12,13--14,15"  -m <path-to-calibration-output>/fp8_output/llama-3.1-405b-instruct/g2/ -o ./unification_files_8x
```
<details><summary>Arguments used</summary>
-g - card grouping to use during unification, card indices separated by commas and groups separated by double dash<br>
-m - calibration output path which has the measurement files <br>
-o - output directory where unification output gets written<br></details>

#### Step 5: Serving the FP8 quantized model <p>
```bash
export QUANT_CONFIG='<path-to-calibration-output>/fp8_output/llama-3.1-405b-instruct/maxabs_quant_g2.json'
vllm serve meta-llama/Llama-3.1-405B-Instruct --quantization inc --kv-cache-dtype fp8_inc --weights-load-device cpu --tensor-parallel-size 8
```
Note : For serving the output after unification, edit the QUANT_CONFIG file to point the 'dump_stats_path' value to the unification output directory

