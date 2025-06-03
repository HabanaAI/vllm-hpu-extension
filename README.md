# deploy

```
git clone https://github.com/vllm-project/vllm.git
git clone https://github.com/vllm-project/vllm-hpu.git

cd vllm; VLLM_TARGET_DEVICE=empty pip install -e .  --no-build-isolation; cd ..
cd vllm-hpu; pip install -e .; cd..
```