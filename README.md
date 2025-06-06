# deploy

```

# install vllm, before upstream PR merged, need to use fork firstly.
#git clone https://github.com/vllm-project/vllm.git
VLLM_TARGET_DEVICE=hpu pip install git+https://github.com/HabanaAI/vllm-fork.git@vllm-upstream-plugin-enhancement

# install plugin
git clone -b plugin/vllm-hpu https://github.com/HabanaAI/vllm-fork.git; mv vllm-fork vllm-hpu;
cd vllm-hpu; pip install -e .; pip uninstall -y triton;  cd..
```

# supported model and feature

| model_name | worker | task | acc metrics | acc score |
|----------- | ------ | ---- | ----------- | --------- |
| Meta-Llama-3.1-8B-Instruct | v0 | gsm8k_cot_llama | exact_match,strict-match | 0.8066 |
| Meta-Llama-3.1-8B-Instruct | v1 | gsm8k_cot_llama | exact_match,strict-match | 0.8105 |
| Qwen3-30B-A3B | v0 | gsm8k | exact_match,strict-match | 0.9023 |
| Qwen3-30B-A3B | v1 | gsm8k | exact_match,strict-match | 0.9062 |

```
cd vllm-hpu/examples; bash run_tests.sh
```

# test

```
cd vllm-hpu/examples;

# v0
PT_HPU_LAZY_MODE=1 python test_plugin.py

# v1
PT_HPU_LAZY_MODE=1 VLLM_USE_V1=1 VLLM_CONTIGUOUS_PA=false python test_plugin.py
```

expected output

```
INFO 06-05 01:12:17 [__init__.py:31] Available plugins for group vllm.platform_plugins:
INFO 06-05 01:12:17 [__init__.py:33] - hpu -> vllm_hpu:register
INFO 06-05 01:12:17 [__init__.py:36] All plugins in this group will be loaded. Set `VLLM_PLUGINS` to control which plugins to load.
INFO 06-05 01:12:17 [__init__.py:234] Platform plugin hpu is activated
WARNING 06-05 01:12:18 [_custom_ops.py:21] Failed to import from vllm._C with ModuleNotFoundError("No module named 'vllm._C'")
INFO 06-05 01:12:19 [__init__.py:31] Available plugins for group vllm.general_plugins:
INFO 06-05 01:12:19 [__init__.py:33] - lora_filesystem_resolver -> vllm.plugins.lora_resolvers.filesystem_resolver:register_filesystem_resolver
INFO 06-05 01:12:19 [__init__.py:33] - hpu_custom_ops -> vllm_hpu:register_ops
INFO 06-05 01:12:19 [__init__.py:36] All plugins in this group will be loaded. Set `VLLM_PLUGINS` to control which plugins to load.
INFO 06-05 01:12:27 [config.py:793] This model supports multiple tasks: {'embed', 'generate', 'classify', 'score', 'reward'}. Defaulting to 'generate'.
INFO 06-05 01:12:27 [arg_utils.py:1594] hpu is experimental on VLLM_USE_V1=1. Falling back to V0 Engine.
INFO 06-05 01:12:27 [config.py:1909] Disabled the custom all-reduce kernel because it is not supported on current platform.
INFO 06-05 01:12:27 [llm_engine.py:230] Initializing a V0 LLM engine (v0.9.1.dev172+gd459fae0a.d20250604) with config: model='/mnt/weka/llm/Qwen3/Qwen3-30B-A3B/', speculative_config=None, tokenizer='Qwen3/Qwen3-30B-A3B/', skip_tokenizer_init=False, tokenizer_mode=auto, revision=None, override_neuron_config={}, tokenizer_revision=None, trust_remote_code=True, dtype=torch.bfloat16, max_seq_len=4096, download_dir=None, load_format=auto, tensor_parallel_size=1, pipeline_parallel_size=1, disable_custom_all_reduce=True, quantization=None, enforce_eager=False, kv_cache_dtype=auto,  device_config=hpu, decoding_config=DecodingConfig(backend='auto', disable_fallback=False, disable_any_whitespace=False, disable_additional_properties=False, reasoning_backend=''), observability_config=ObservabilityConfig(show_hidden_metrics_for_version=None, otlp_traces_endpoint=None, collect_detailed_traces=None), seed=0, served_model_name=/mnt/weka/llm/Qwen3/Qwen3-30B-A3B/, num_scheduler_steps=1, multi_step_stream_outputs=True, enable_prefix_caching=None, chunked_prefill_enabled=False, use_async_output_proc=True, pooler_config=None, compilation_config={"compile_sizes": [], "inductor_compile_config": {"enable_auto_functionalized_v2": false}, "cudagraph_capture_sizes": [256, 248, 240, 232, 224, 216, 208, 200, 192, 184, 176, 168, 160, 152, 144, 136, 128, 120, 112, 104, 96, 88, 80, 72, 64, 56, 48, 40, 32, 24, 16, 8, 4, 2, 1], "max_capture_size": 256}, use_cached_outputs=False, 
WARNING 06-05 01:12:28 [utils.py:2671] Methods add_prompt_adapter,cache_config,compilation_config,current_platform,list_prompt_adapters,load_config,pin_prompt_adapter,remove_prompt_adapter,scheduler_config not implemented in <vllm_hpu.worker.hpu_worker.HPUWorker object at 0x7f7b5dcb9c90>
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
INFO 06-05 01:12:29 [parallel_state.py:1064] rank 0 in world size 1 is assigned as DP rank 0, PP rank 0, TP rank 0, EP rank 0
Detected flags: [-compile_one_hot -cpu -flex_attention -fp32_softmax +fsdpa -fused_block_softmax_adjustment -gaudi -gaudi2 +gaudi3]

Loading safetensors checkpoint shards:   0% Completed | 0/16 [00:00<?, ?it/s]
...

Loading safetensors checkpoint shards: 100% Completed | 16/16 [00:13<00:00,  1.17it/s]

INFO 06-05 01:12:44 [default_loader.py:280] Loading weights took 13.73 seconds
INFO 06-05 01:14:11 [executor_base.py:112] # hpu blocks: 4742, # CPU blocks: 341
INFO 06-05 01:14:11 [executor_base.py:117] Maximum concurrency for 4096 tokens per request: 148.19x
INFO 06-05 01:14:13 [llm_engine.py:428] init engine (profile, create kv cache, warmup model) took 12.23 seconds

Adding requests:   0%|          | 0/4 [00:00<?, ?it/s]
Adding requests: 100%|██████████| 4/4 [00:00<00:00, 1987.35it/s]

Processed prompts:   0%|          | 0/4 [00:00<?, ?it/s, est. speed input: 0.00 toks/s, output: 0.00 toks/s]Configuration: ('prompt', 4, 128) was not warmed-up!
Configuration: ('decode', 4, 128) was not warmed-up!

Processed prompts:  25%|██▌       | 1/4 [00:06<00:20,  6.80s/it, est. speed input: 0.74 toks/s, output: 7.35 toks/s]
Processed prompts: 100%|██████████| 4/4 [00:06<00:00,  1.70s/it, est. speed input: 4.12 toks/s, output: 29.42 toks/s]
Prompt: 'Hello, my name is', Generated text: ' Sarah and the ** 111111111111111111111111111111111111111111111'
Prompt: '0.999 compares to 0.9 is ', Generated text: '0.999 is greater than 0.9. So, 0.999 > 0.9. So, the answer is greater than.\n\nBut wait, the question is asking "0.999 compares to'
Prompt: 'The capital of France is', Generated text: ' Paris. The capital of the United Kingdom is London. The capital of the United States is Washington, D.C. The capital of Brazil is Brasília. The capital of Japan is Tokyo. The capital of India is New Delhi. The capital of Australia'
Prompt: 'The future of AI is', Generated text: ' not just about the technology itself, but about how it is used and the impact it has on society. As AI continues to evolve, it is important to consider the ethical implications of its use. One of the main concerns is the potential for AI to'
Exception ignored in: <function HPUModelRunner.__del__ at 0x7f7b4ab89510>
Traceback (most recent call last):
  File "vllm-hpu/vllm_hpu/worker/hpu_model_runner.py", line 3130, in __del__
ImportError: sys.meta_path is None, Python is likely shutting down

```
