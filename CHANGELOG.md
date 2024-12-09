# CHANGELOG
## v1.1.0
- Support group-wise quantization (w4a16 group sizes of 32/64/128, w8a8 group sizes of 128/256/512).
- Support joint inference with LoRA model loading
- Support storage and preloading of prompt cache.
- Support gguf model conversion (currently only support q4_0 and fp16).
- Optimize initialization, prefill, and decode time.
- Support four input types: prompt, embedding, token, and multimodal.
- Add PC-based simulation accuracy testing and inference interface support for rkllm-toolkit.
- Add gdq algorithm to improve 4-bit quantization accuracy.
- Add mixed quantization algorithm, supporting a combination of grouped and non-grouped quantization based on specified ratios.
- Add support for models such as Llama3, Gemma2, and MiniCPM3.
- Resolve catastrophic forgetting issue when the number of tokens exceeds max_context.

## v1.0.1
 - Optimize model conversion memory occupation
 - Optimize inference memory occupation
 - Increase prefill speed
 - Reduce initialization time
 - Improve quantization accuracy
 - Add support for Gemma, ChatGLM3, MiniCPM, InternLM2, and Phi-3
 - Add Server invocation
 - Add inference interruption interface
 - Add logprob and token_id to the return value

## v1.0.0
 - Support the conversion and deployment of LLM models on RK3588/RK3576 platforms
 - Compatible with Hugging Face model architectures
 - Currently support the models Llama, Qwen, Qwen2, and Phi-2
 - Support quantization with w8a8 and w4a16 precision