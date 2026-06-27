"""
benchmark_runner.py

Unified benchmark harness for Gemma Challenge optimization work.

Responsibilities:
- Run inference benchmark
- Collect TPS / latency / basic GPU stats
- Save structured JSON output
- Compare baseline vs candidate runs (optional extension point)

This is NOT tied to any single engine.
You plug in an inference backend via `EngineRunner`.
"""

import time
import json
import statistics
import subprocess
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
from vllm_engine import VLLMEngine

engine = VLLMEngine(model_name=config.model_name)

# -----------------------------
# CONFIG
# -----------------------------

@dataclass
class BenchmarkConfig:
    engine_name: str
    model_name: str = "gemma-4-E4B-it"
    batch_size: int = 1
    max_tokens: int = 256
    prompt: str = "Write a short explanation of transformer models."
    runs: int = 5
    warmup_runs: int = 2
    output_dir: str = "auto_bench"
    use_nsight: bool = False


# -----------------------------
# METRICS
# -----------------------------

@dataclass
class BenchmarkResult:
    engine: str
    tps_mean: float
    tps_stdev: float
    latency_mean: float
    latency_stdev: float
    total_runs: int


# -----------------------------
# ENGINE INTERFACE
# -----------------------------

class EngineRunner:
    """
    Abstract inference engine wrapper.

    You should implement run_inference() for:
    - vLLM
    - TensorRT-LLM
    - custom PyTorch loop
    """

    def run_inference(self, prompt: str, max_tokens: int) -> Dict:
        """
        Must return:
        {
            "tokens_generated": int,
            "latency_sec": float
        }
        """
        raise NotImplementedError


# -----------------------------
# EXAMPLE MOCK ENGINE
# -----------------------------

class MockEngine(EngineRunner):
    """
    Placeholder engine for testing pipeline.

    Replace this with real vLLM / TRT / custom engine wrapper.
    """

    def run_inference(self, prompt: str, max_tokens: int) -> Dict:
        start = time.time()

        # simulate compute
        tokens = max_tokens
        time.sleep(0.05 + (tokens * 0.0002))

        end = time.time()

        return {
            "tokens_generated": tokens,
            "latency_sec": end - start
        }


# -----------------------------
# BENCHMARK RUNNER
# -----------------------------

class BenchmarkRunner:

    def __init__(self, config: BenchmarkConfig, engine: EngineRunner):
        self.config = config
        self.engine = engine

    def run_single(self) -> Dict:
        result = self.engine.run_inference(
            prompt=self.config.prompt,
            max_tokens=self.config.max_tokens
        )

        tokens = result["tokens_generated"]
        latency = result["latency_sec"]

        tps = tokens / latency if latency > 0 else 0.0

        return {
            "tokens": tokens,
            "latency": latency,
            "tps": tps
        }

    def run(self) -> BenchmarkResult:

        tps_list: List[float] = []
        latency_list: List[float] = []

        # -------------------------
        # Warmup
        # -------------------------
        for _ in range(self.config.warmup_runs):
            self.run_single()

        # -------------------------
        # Measured Runs
        # -------------------------
        for i in range(self.config.runs):
            print(f"Running benchmark {i+1}/{self.config.runs}")

            result = self.run_single()

            tps_list.append(result["tps"])
            latency_list.append(result["latency"])

        # -------------------------
        # Statistics
        # -------------------------
        tps_mean = statistics.mean(tps_list)
        tps_stdev = statistics.stdev(tps_list) if len(tps_list) > 1 else 0.0

        latency_mean = statistics.mean(latency_list)
        latency_stdev = statistics.stdev(latency_list) if len(latency_list) > 1 else 0.0

        return BenchmarkResult(
            engine=self.config.engine_name,
            tps_mean=tps_mean,
            tps_stdev=tps_stdev,
            latency_mean=latency_mean,
            latency_stdev=latency_stdev,
            total_runs=self.config.runs
        )


# -----------------------------
# OPTIONAL NSIGHT HOOK
# -----------------------------

def run_with_nsight(command: List[str], output_name: str):
    """
    Wrap execution with Nsight Systems profiling.

    Example:
    nsys profile --output=trace python script.py
    """

    nsys_cmd = [
        "nsys",
        "profile",
        "--trace=cuda,nvtx,osrt",
        f"--output={output_name}"
    ] + command

    subprocess.run(nsys_cmd)


# -----------------------------
# SAVE RESULTS
# -----------------------------

def save_result(config: BenchmarkConfig, result: BenchmarkResult):

    output = {
        "config": asdict(config),
        "result": asdict(result),
        "timestamp": time.time()
    }

    path = f"{config.output_dir}/result_{config.engine_name}.json"

    with open(path, "w") as f:
        json.dump(output, f, indent=4)

    print(f"\nSaved benchmark → {path}")


# -----------------------------
# MAIN
# -----------------------------

def main():

    config = BenchmarkConfig(
        engine_name="mock_engine",
        runs=5,
        warmup_runs=2
    )

    engine = MockEngine()

    runner = BenchmarkRunner(config, engine)

    result = runner.run()

    save_result(config, result)

    print("\n=== BENCHMARK RESULT ===")
    print(f"Engine: {result.engine}")
    print(f"TPS Mean: {result.tps_mean:.2f}")
    print(f"TPS Std: {result.tps_stdev:.2f}")
    print(f"Latency Mean: {result.latency_mean:.4f}s")


if __name__ == "__main__":
    main()
