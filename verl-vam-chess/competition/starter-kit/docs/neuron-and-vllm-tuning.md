### Submitting models to Neuron: pick the right `--neuron.model-type` (and tune vLLM if you need to)

When you run `aicrowd submit-model`, the platform spins up a **vLLM server** for your model. You can pass a handful of `--vllm.*` flags to control things like max context length, dtype, batching limits, LoRA settings, and a few inference-time parameters.

But there’s one flag you **must** get right for Neuron hardware:

```bash
--neuron.model-type <model-type>
```

#### Why this matters (Neuron compilation in one paragraph)
AWS Inferentia/Trainium (Neuron) doesn’t run your PyTorch model “as-is” the way a typical GPU setup might. The model needs to be **compiled** into a Neuron-compatible artifact before it can run on the accelerator.

Because the compilation path is **model-architecture-specific**, the submission system needs to know which backend/architecture you’re using. Hence `--neuron.model-type`.

If you don’t set `--neuron.model-type`, the submission will **default to `qwen3`**.

---

## Supported model types (backends)
The NxD Inference model hub currently supports architectures including: **Llama (text)**, **Llama (multimodal)**, **Llama4**, **Mixtral**, **DBRX**, **Qwen2.5**, **Qwen3**, and **FLUX.1 (beta)**. 

https://github.com/aws-neuron/neuronx-distributed-inference/blob/main/src/neuronx_distributed_inference/inference_demo.py#L51-L56

(For the others, the exact string is typically the obvious lowercase name, aligned to the supported architecture list above. If you’re unsure, match it to your model family—e.g., Mixtral → `mixtral`, Qwen3 → `qwen3`—because picking the wrong type can lead to compile failures or incorrect behavior.)

---

## Supported vLLM server flags
`aicrowd submit-model` currently supports these vLLM arguments:

```text
--vllm.max-model-len
--vllm.dtype
--vllm.kv-cache-dtype
--vllm.quantization
--vllm.load-format
--vllm.rope-theta
--vllm.rope-scaling
--vllm.max-num-batched-tokens
--vllm.max-num-seqs
--vllm.enforce-eager true
--vllm.enable-lora true
--vllm.lora-dtype
--vllm.lora-extra-vocab-size
--vllm.enable-prefix-caching true
--vllm-env.allow-long-max-model-len

# inference time parameters
--vllm-inference.max-tokens
```

---

## Example: complete submission command
At minimum, set your repo/tag and the Neuron model type:

```bash
aicrowd submit-model \
  --hf-repo <repo> \
  --hf-repo-tag <branch/tag> \
  --neuron.model-type llama
```

Or, for DBRX:

```bash
aicrowd submit-model \
  --hf-repo <repo> \
  --hf-repo-tag <branch/tag> \
  --neuron.model-type dbrx
```

If you need tighter control over serving behavior, add the relevant `--vllm.*` flags on the same command.