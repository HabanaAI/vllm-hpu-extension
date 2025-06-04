# deploy

```
#git clone https://github.com/vllm-project/vllm.git
#git clone https://github.com/vllm-project/vllm-hpu.git

git clone -b vllm-upstream-plugin-enhancement https://github.com/HabanaAI/vllm-fork.git; mv vllm-fork vllm;
git clone -b plugin/vllm-hpu https://github.com/HabanaAI/vllm-fork.git; mv vllm-fork vllm-hpu;

cd vllm; pip install -r requirements/common.txt; pip install triton==3.1.0 setuptools-scm>=8; VLLM_TARGET_DEVICE=empty pip install -e .  --no-build-isolation; cd ..
cd vllm-hpu; pip install -e .; cd..
```

# test

```
cd vllm-hpu/examples; PT_HPU_LAZY_MODE=1 python test_plugin.py
```

expected output
```
INFO 06-04 01:18:14 [importing.py:17] Triton not installed or not compatible; certain GPU-related functions will not be available.
WARNING 06-04 01:18:14 [importing.py:29] Triton is not installed. Using dummy decorators. Install it via `pip install triton` to enable kernel compilation.
INFO 06-04 01:18:15 [__init__.py:39] Available plugins for group vllm.platform_plugins:
INFO 06-04 01:18:15 [__init__.py:41] - hpu -> vllm_hpu:register
INFO 06-04 01:18:15 [__init__.py:44] All plugins in this group will be loaded. Set `VLLM_PLUGINS` to control which plugins to load.
INFO 06-04 01:18:15 [__init__.py:235] Platform plugin hpu is activated
WARNING 06-04 01:18:17 [_custom_ops.py:22] Failed to import from vllm._C with ModuleNotFoundError("No module named 'vllm._C'")
INFO 06-04 01:18:25 [config.py:822] This model supports multiple tasks: {'embed', 'generate', 'reward', 'score', 'classify'}. Defaulting to 'generate'.
INFO 06-04 01:18:25 [config.py:3247] Downcasting torch.float32 to torch.bfloat16.
INFO 06-04 01:18:25 [arg_utils.py:1628] hpu is experimental on VLLM_USE_V1=1. Falling back to V0 Engine.
INFO 06-04 01:18:25 [config.py:1967] Disabled the custom all-reduce kernel because it is not supported on current platform.
INFO 06-04 01:18:25 [llm_engine.py:231] Initializing a V0 LLM engine (v0.9.1.dev116+gc57d577e8.d20250603) with config: model='/mnt/weka/data/pytorch/llama3.1/Meta-Llama-3.1-8B/', speculative_config=None, tokenizer='/mnt/weka/data/pytorch/llama3.1/Meta-Llama-3.1-8B/', skip_tokenizer_init=False, tokenizer_mode=auto, revision=None, override_neuron_config={}, tokenizer_revision=None, trust_remote_code=False, dtype=torch.bfloat16, max_seq_len=4096, download_dir=None, load_format=auto, tensor_parallel_size=1, pipeline_parallel_size=1, disable_custom_all_reduce=True, quantization=None, enforce_eager=True, kv_cache_dtype=auto,  device_config=hpu, decoding_config=DecodingConfig(backend='auto', disable_fallback=False, disable_any_whitespace=False, disable_additional_properties=False, reasoning_backend=''), observability_config=ObservabilityConfig(show_hidden_metrics_for_version=None, otlp_traces_endpoint=None, collect_detailed_traces=None), seed=0, served_model_name=/mnt/weka/data/pytorch/llama3.1/Meta-Llama-3.1-8B/, num_scheduler_steps=1, multi_step_stream_outputs=True, enable_prefix_caching=None, chunked_prefill_enabled=False, use_async_output_proc=True, pooler_config=None, compilation_config={"level":0,"debug_dump_path":"","cache_dir":"","backend":"","custom_ops":[],"splitting_ops":[],"use_inductor":true,"compile_sizes":[],"inductor_compile_config":{"enable_auto_functionalized_v2":false},"inductor_passes":{},"use_cudagraph":false,"cudagraph_num_of_warmups":0,"cudagraph_capture_sizes":[],"cudagraph_copy_inputs":false,"full_cuda_graph":false,"max_capture_size":0,"local_cache_dir":null}, use_cached_outputs=False, 
WARNING 06-04 01:18:28 [utils.py:2722] Methods add_prompt_adapter,cache_config,compilation_config,current_platform,list_prompt_adapters,load_config,pin_prompt_adapter,remove_prompt_adapter,scheduler_config not implemented in <vllm_hpu.worker.hpu_worker.HPUWorker object at 0x7f1204edb310>
Pin memory is not supported on HPU.
============================= HPU PT BRIDGE CONFIGURATION ON RANK = 0 ============= 
 PT_HPU_LAZY_MODE = 1
 PT_HPU_RECIPE_CACHE_CONFIG = ,false,1024
 PT_HPU_MAX_COMPOUND_OP_SIZE = 9223372036854775807
 PT_HPU_LAZY_ACC_PAR_MODE = 1
 PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES = 0
 PT_HPU_EAGER_PIPELINE_ENABLE = 1
 PT_HPU_EAGER_COLLECTIVE_PIPELINE_ENABLE = 1
 PT_HPU_ENABLE_LAZY_COLLECTIVES = 0
---------------------------: System Configuration :---------------------------
Num CPU Cores : 224
CPU RAM       : 1007 GB
------------------------------------------------------------------------------
INFO 06-04 01:18:28 [parallel_state.py:1065] rank 0 in world size 1 is assigned as DP rank 0, PP rank 0, TP rank 0, EP rank 0
Detected flags: [-compile_one_hot -cpu -flex_attention -fp32_softmax +fsdpa -fused_block_softmax_adjustment -gaudi -gaudi2 +gaudi3]

Loading safetensors checkpoint shards:   0% Completed | 0/7 [00:00<?, ?it/s]
Loading safetensors checkpoint shards:  14% Completed | 1/7 [00:00<00:04,  1.30it/s]
Loading safetensors checkpoint shards:  29% Completed | 2/7 [00:01<00:02,  1.70it/s]
Loading safetensors checkpoint shards:  43% Completed | 3/7 [00:01<00:02,  1.70it/s]
Loading safetensors checkpoint shards:  57% Completed | 4/7 [00:02<00:01,  1.65it/s]
Loading safetensors checkpoint shards:  71% Completed | 5/7 [00:02<00:01,  1.73it/s]
Loading safetensors checkpoint shards:  86% Completed | 6/7 [00:03<00:00,  1.74it/s]
Loading safetensors checkpoint shards: 100% Completed | 7/7 [00:04<00:00,  1.81it/s]
Loading safetensors checkpoint shards: 100% Completed | 7/7 [00:04<00:00,  1.73it/s]

INFO 06-04 01:18:34 [default_loader.py:272] Loading weights took 4.12 seconds
INFO 06-04 01:18:37 [executor_base.py:113] # hpu blocks: 6399, # CPU blocks: 256
INFO 06-04 01:18:37 [executor_base.py:118] Maximum concurrency for 4096 tokens per request: 199.97x
INFO 06-04 01:18:37 [llm_engine.py:429] init engine (profile, create kv cache, warmup model) took 2.73 seconds

Adding requests:   0%|          | 0/4 [00:00<?, ?it/s]
Adding requests: 100%|██████████| 4/4 [00:00<00:00, 1949.93it/s]

Processed prompts:   0%|          | 0/4 [00:00<?, ?it/s, est. speed input: 0.00 toks/s, output: 0.00 toks/s]Configuration: ('prompt', 4, 128) was not warmed-up!
Configuration: ('decode', 4, 128) was not warmed-up!

Processed prompts:  25%|██▌       | 1/4 [00:04<00:12,  4.22s/it, est. speed input: 1.42 toks/s, output: 11.85 toks/s]
Processed prompts: 100%|██████████| 4/4 [00:04<00:00,  1.06s/it, est. speed input: 7.11 toks/s, output: 47.39 toks/s]
Prompt: 'Hello, my name is', Generated text: ' K. The 1. The 1. The 1. The 1. The 1. The 1. The 1. The 1. The 1. The 1. The 1. The 1.'
Prompt: '0.999 compares to 0.9 is ', Generated text: '1 compares to 0.9\n0.999 compares to 0.9 is 1 compares to 0.9\nPost by mathnasty » Mon Mar 11, 2013 10:00 pm\nI have a'
Prompt: 'The capital of France is', Generated text: ' a city of many faces. It is a city of history, culture, and art. It is a city of fashion, food, and wine. It is a city of romance, passion, and adventure. It is a city that has something to'
Prompt: 'The future of AI is', Generated text: ' here, and it’s already changing the way we live and work. From self-driving cars to virtual assistants, AI is becoming more and more integrated into our daily lives. But what does this mean for the future of work? In this blog post,'
Exception ignored in: <function HPUModelRunner.__del__ at 0x7f11eff56440>
Traceback (most recent call last):
  File "/software/users/chendixue/dev/vllm-hpu/vllm_hpu/worker/hpu_model_runner.py", line 3130, in __del__
  File "/software/users/chendixue/dev/vllm-hpu/vllm_hpu/worker/hpu_model_runner.py", line 3117, in shutdown_inc
ImportError: sys.meta_path is None, Python is likely shutting down

```
