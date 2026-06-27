"""
optimization_orchestrator.py

End-to-end orchestration system for Gemma inference optimization.

Purpose:
- Run benchmark
- Capture Nsight trace (optional)
- Analyze performance
- Generate ideas
- Log experiment
- Recommend next action

This is the "control loop" of the entire repo.
"""

import os
import json
import time
import subprocess
from datetime import datetime

from benchmark_runner import BenchmarkRunner, BenchmarkConfig
from vllm_engine import VLLMEngine
from nsight_parser import run_trace_analysis


# -----------------------------
# CONFIG
# -----------------------------

@dataclass
class OrchestratorConfig:
    model_name: str = "gemma-4-E4B-it"
    engine: str = "vllm"
    runs: int = 5
    warmup_runs: int = 2
    max_tokens: int = 256
    prompt: str = "Explain attention in transformers."
    enable_nsight: bool = False
    trace_output_dir: str = "traces"
    experiment_dir: str = "experiments"


# -----------------------------
# EXPERIMENT LOGGER
# -----------------------------

class ExperimentLogger:

    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)

    def log(self, data: dict) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.base_dir, f"exp_{timestamp}.json")

        with open(path, "w") as f:
            json.dump(data, f, indent=4)

        return path


# -----------------------------
# NSIGHT WRAPPER
# -----------------------------

class NsightRunner:

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def run(self, command: list, name: str) -> str:
        """
        Runs Nsight Systems profiling around a command.
        """

        trace_path = os.path.join(self.output_dir, name)

        nsys_cmd = [
            "nsys",
            "profile",
            "--trace=cuda,nvtx,osrt",
            f"--output={trace_path}"
        ] + command

        print(f"[Nsight] Running trace: {name}")
        subprocess.run(nsys_cmd)

        return trace_path + ".json"


# -----------------------------
# CORE ORCHESTRATOR
# -----------------------------

class OptimizationOrchestrator:

    def __init__(self, config: OrchestratorConfig):
        self.config = config

        self.logger = ExperimentLogger(config.experiment_dir)
        self.nsight = NsightRunner(config.trace_output_dir)

        self.engine = VLLMEngine(model_name=config.model_name)

    # -----------------------------
    # Run benchmark
    # -----------------------------

    def run_benchmark(self):

        bench_config = BenchmarkConfig(
            engine_name="vllm",
            model_name=self.config.model_name,
            runs=self.config.runs,
            warmup_runs=self.config.warmup_runs,
            max_tokens=self.config.max_tokens,
            prompt=self.config.prompt
        )

        runner = BenchmarkRunner(bench_config, self.engine)

        return runner.run()

    # -----------------------------
    # Optional Nsight run
    # -----------------------------

    def run_with_profiling(self):

        cmd = [
            "python",
            "benchmark_runner.py"
        ]

        trace_file = self.nsight.run(
            command=cmd,
            name="trace"
        )

        return trace_file

    # -----------------------------
    # Full pipeline
    # -----------------------------

    def run(self):

        print("\n==============================")
        print("STARTING OPTIMIZATION RUN")
        print("==============================\n")

        start_time = time.time()

        # -------------------------
        # Step 1: Benchmark
        # -------------------------
        print("[1/4] Running benchmark...")
        result = self.run_benchmark()

        # -------------------------
        # Step 2: Optional profiling
        # -------------------------
        trace_file = None

        if self.config.enable_nsight:
            print("[2/4] Running Nsight profiling...")
            trace_file = self.run_with_profiling()

        # -------------------------
        # Step 3: Log experiment
        # -------------------------
        print("[3/4] Logging experiment...")

        experiment_data = {
            "timestamp": datetime.now().isoformat(),
            "engine": self.config.engine,
            "model": self.config.model_name,
            "benchmark": {
                "tps_mean": result.tps_mean,
                "tps_stdev": result.tps_stdev,
                "latency_mean": result.latency_mean,
                "latency_stdev": result.latency_stdev,
            },
            "trace_file": trace_file
        }

        log_path = self.logger.log(experiment_data)

        # -------------------------
        # Step 4: Analyze trace
        # -------------------------
        if trace_file and os.path.exists(trace_file):
            print("[4/4] Analyzing Nsight trace...")
            run_trace_analysis(trace_file)

        end_time = time.time()

        # -------------------------
        # Summary
        # -------------------------
        print("\n==============================")
        print("RUN COMPLETE")
        print("==============================")
        print(f"TPS Mean: {result.tps_mean:.2f}")
        print(f"Latency: {result.latency_mean:.4f}s")
        print(f"Experiment log: {log_path}")
        print(f"Total time: {end_time - start_time:.2f}s")


# -----------------------------
# ENTRYPOINT
# -----------------------------

if __name__ == "__main__":

    from dataclasses import dataclass

    config = OrchestratorConfig(
        enable_nsight=False  # turn on when ready for full profiling
    )

    orchestrator = OptimizationOrchestrator(config)
    orchestrator.run()
