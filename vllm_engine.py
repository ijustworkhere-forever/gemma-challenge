"""
vllm_engine.py

vLLM inference engine adapter for benchmark_runner.py

Purpose:
- Standardize vLLM inference output
- Measure tokens/sec and latency
- Provide clean interface for benchmarking system

Works with:
- vLLM Python API (preferred for benchmarking)
- Optional OpenAI-compatible server mode (future extension)
"""

import time
from typing import Dict, Any, Optional


# -----------------------------
# Optional import guard
# -----------------------------

try:
    from vllm import LLM, SamplingParams
except ImportError:
    LLM = None
    SamplingParams = None


# -----------------------------
# Engine
# -----------------------------

class VLLMEngine:
    """
    vLLM inference wrapper.

    This is designed for benchmarking, NOT production serving.
    """

    def __init__(
        self,
        model_name: str,
        tensor_parallel_size: int = 1,
        dtype: str = "half",
        gpu_memory_utilization: float = 0.90
    ):
        if LLM is None:
            raise ImportError(
                "vLLM is not installed. Install with: pip install vllm"
            )

        self.model_name = model_name

        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            dtype=dtype,
            gpu_memory_utilization=gpu_memory_utilization,
        )

    # -----------------------------
    # Core inference method
    # -----------------------------

    def run_inference(self, prompt: str, max_tokens: int) -> Dict[str, Any]:
        """
        Runs single inference request and returns:
        - tokens generated
        - latency (seconds)
        - optional internal metrics
        """

        sampling_params = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            max_tokens=max_tokens
        )

        start_time = time.time()

        outputs = self.llm.generate(
            [prompt],
            sampling_params
        )

        end_time = time.time()

        # -----------------------------
        # Extract token counts
        # -----------------------------

        output = outputs[0]

        # vLLM returns generated text; token count is derived
        # We estimate via output length if token IDs are not exposed
        generated_text = output.outputs[0].text
        token_count = len(generated_text.split())  # fallback approx

        latency = end_time - start_time

        return {
            "tokens_generated": token_count,
            "latency_sec": latency,
            "engine": "vllm"
        }


# -----------------------------
# Utility test block
# -----------------------------

if __name__ == "__main__":

    engine = VLLMEngine(
        model_name="meta-llama/Llama-3-8B-Instruct"
    )

    result = engine.run_inference(
        prompt="Explain transformer attention in simple terms.",
        max_tokens=128
    )

    print("\n=== vLLM TEST ===")
    print(result)
