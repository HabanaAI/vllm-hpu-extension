from vllm import LLM, SamplingParams
import os

os.environ["VLLM_SKIP_WARMUP"] = "true"
prompts = [
    "Hello, my name is",
    "0.999 compares to 0.9 is ",
    "The capital of France is",
    "The future of AI is",
]
sampling_params = SamplingParams(temperature=0, max_tokens=50)
model = "/mnt/weka/llm/Qwen3/Qwen3-32B/"
model = "/mnt/weka/data/pytorch/llama3.1/Meta-Llama-3.1-8B/"
llm = LLM(model=model, max_model_len=4096, enforce_eager=True)

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")