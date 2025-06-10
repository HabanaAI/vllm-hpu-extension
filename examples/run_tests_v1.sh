# !/bin/bash

# Default tests, disable_prefix_cache, hpu graph 

# llama
VLLM_SKIP_WARMUP=True PT_HPU_LAZY_MODE=1 VLLM_USE_V1=1 VLLM_CONTIGUOUS_PA=false \
pytest -v -s ../tests/models/language/generation/test_common.py --model_card_path ../tests/models/language/generation/model_cards/Meta-Llama-3.1-8B-Instruct.yaml

# APC tests
VLLM_SKIP_WARMUP=True PT_HPU_LAZY_MODE=1 VLLM_USE_V1=1 VLLM_CONTIGUOUS_PA=false ENABLE_APC=1 \
pytest -v -s ../tests/models/language/generation/test_common.py --model_card_path ../tests/models/language/generation/model_cards/Meta-Llama-3.1-8B-Instruct.yaml

# Eager tests
VLLM_SKIP_WARMUP=True PT_HPU_LAZY_MODE=1 VLLM_USE_V1=1 VLLM_CONTIGUOUS_PA=false ENFORCE_EAGER=1 \
pytest -v -s ../tests/models/language/generation/test_common.py --model_card_path ../tests/models/language/generation/model_cards/Meta-Llama-3.1-8B-Instruct.yaml

# hpu graph warmup 
PT_HPU_LAZY_MODE=1 VLLM_USE_V1=1 VLLM_CONTIGUOUS_PA=false \
pytest -v -s ../tests/models/language/generation/test_common.py --model_card_path ../tests/models/language/generation/model_cards/Meta-Llama-3.1-8B-Instruct.yaml

# TP=2
VLLM_SKIP_WARMUP=True PT_HPU_LAZY_MODE=1 VLLM_USE_V1=1 VLLM_CONTIGUOUS_PA=false TP_SIZE=2 \
pytest -v -s ../tests/models/language/generation/test_common.py --model_card_path ../tests/models/language/generation/model_cards/Meta-Llama-3.1-8B-Instruct.yaml

# ======================== Model validate ========================== #
# qwen3 moe
VLLM_SKIP_WARMUP=True PT_HPU_LAZY_MODE=1 VLLM_USE_V1=1 VLLM_CONTIGUOUS_PA=false \
pytest -v -s ../tests/models/language/generation/test_common.py --model_card_path ../tests/models/language/generation/model_cards/Qwen3-30B-A3B.yaml

# granite
VLLM_SKIP_WARMUP=True PT_HPU_LAZY_MODE=1 VLLM_USE_V1=1 VLLM_CONTIGUOUS_PA=false \
pytest -v -s ../tests/models/language/generation/test_common.py --model_card_path ../tests/models/language/generation/model_cards/granite-8b.yaml

# mistral
VLLM_SKIP_WARMUP=True PT_HPU_LAZY_MODE=1 VLLM_USE_V1=1 VLLM_CONTIGUOUS_PA=false \
pytest -v -s ../tests/models/language/generation/test_common.py --model_card_path ../tests/models/language/generation/model_cards/Mistral-7B-Instruct-v0.3.yaml

# mixtral TP=2
VLLM_SKIP_WARMUP=True PT_HPU_LAZY_MODE=1 VLLM_USE_V1=1 VLLM_CONTIGUOUS_PA=false \
pytest -v -s ../tests/models/language/generation/test_common.py --model_card_path ../tests/models/language/generation/model_cards/Mixtral-8x7B-Instruct-v0.1.yaml

# llama4-moe TP=4 (text generation)
VLLM_SKIP_WARMUP=True PT_HPU_LAZY_MODE=1 VLLM_USE_V1=1 VLLM_CONTIGUOUS_PA=false \
pytest -v -s ../tests/models/language/generation/test_common.py --model_card_path ../tests/models/language/generation/model_cards/Llama-4-Scout-17B-16E-Instruct.yaml

# llama4-moe TP=4 (vision ineference)
VLLM_SKIP_WARMUP=True PT_HPU_LAZY_MODE=1 VLLM_USE_V1=1 VLLM_CONTIGUOUS_PA=false \
pytest -v -s ../tests/models/language/generation/test_common.py --model_card_path ../tests/models/language/generation/model_cards/Llama-4-Scout-17B-16E-Instruct-vision.yaml
