"""
autonomous_optimizer_agent.py

Autonomous optimization loop for inference performance tuning.

Purpose:
- Run benchmark experiments
- Evaluate results
- Suggest + execute next experiment
- Stop when improvements plateau

This is a controlled optimization agent, NOT an unrestricted self-modifying system.
"""

import os
import json
import time
import copy
from dataclasses import dataclass
from typing import Dict, Any, Optional, List

from benchmark_runner import BenchmarkRunner, BenchmarkConfig
from vllm_engine import VLLMEngine
from next_experiment_suggester import run_suggestion_pipeline


# -----------------------------
# CONFIG
# -----------------------------

@dataclass
class AgentConfig:
    model_name: str = "gemma-4-E4B-it"
    max_iterations: int = 5
    max_tokens: int = 256
    runs_per_experiment: int = 3
    warmup_runs: int = 1

    improvement_threshold: float = 2.0  # TPS %
    plateau_patience: int = 2

    experiment_dir: str = "auto_experiments"


# -----------------------------
# EXPERIMENT STATE
# -----------------------------

@dataclass
class ExperimentState:
    iteration: int
    best_tps: float
    best_config: Dict[str, Any]
    plateau_counter: int


# -----------------------------
# EXPERIMENT VARIANTS
# -----------------------------

def generate_variants(base_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Generate controlled experiment variations.

    IMPORTANT:
    This is NOT random mutation.
    Only safe, known optimization knobs.
    """

    variants = []

    # Variant 1: increase batch efficiency
    v1 = copy.deepcopy(base_config)
    v1["batch_size"] = min(base_config.get("batch_size", 1) + 1, 8)
    variants.append(v1)

    # Variant 2: reduce max tokens (latency optimization)
    v2 = copy.deepcopy(base_config)
    v2["max_tokens"] = max(base_config.get("max_tokens", 256) - 64, 64)
    variants.append(v2)

    # Variant 3: increase warmup stability
    v3 = copy.deepcopy(base_config)
    v3["warmup_runs"] = base_config.get("warmup_runs", 1) + 2
    variants.append(v3)

    return variants


# -----------------------------
# RUN SINGLE EXPERIMENT
# -----------------------------

def run_experiment(config: Dict[str, Any]) -> Dict[str, Any]:

    engine = VLLMEngine(model_name=config["model_name"])

    bench_config = BenchmarkConfig(
        engine_name="vllm",
        model_name=config["model_name"],
        batch_size=config.get("batch_size", 1),
        max_tokens=config.get("max_tokens", 256),
        runs=config.get("runs", 3),
        warmup_runs=config.get("warmup_runs", 1),
        prompt=config.get("prompt", "Explain transformers simply.")
    )

    runner = BenchmarkRunner(bench_config, engine)

    result = runner.run()

    return {
        "config": config,
        "tps_mean": result.tps_mean,
        "tps_stdev": result.tps_stdev,
        "latency_mean": result.latency_mean
    }


# -----------------------------
# EVALUATION
# -----------------------------

def is_improvement(new_tps: float, best_tps: float, threshold: float) -> bool:
    if best_tps == 0:
        return True
    return ((new_tps - best_tps) / best_tps) * 100.0 >= threshold


# -----------------------------
# MAIN AGENT
# -----------------------------

class AutonomousOptimizer:

    def __init__(self, config: AgentConfig):

        self.config = config

        self.base_config = {
            "model_name": config.model_name,
            "batch_size": 1,
            "max_tokens": config.max_tokens,
            "runs": config.runs_per_experiment,
            "warmup_runs": config.warmup_runs,
            "prompt": "Explain attention mechanisms in transformers."
        }

        self.state = ExperimentState(
            iteration=0,
            best_tps=0.0,
            best_config=self.base_config,
            plateau_counter=0
        )

        os.makedirs(config.experiment_dir, exist_ok=True)

    # -----------------------------
    # RUN LOOP
    # -----------------------------

    def run(self):

        print("\n==============================")
        print("AUTONOMOUS OPTIMIZER STARTED")
        print("==============================\n")

        for i in range(self.config.max_iterations):

            self.state.iteration = i + 1

            print(f"\n[ITERATION {i+1}] Running experiment set...")

            variants = generate_variants(self.state.best_config)

            results = []

            # -------------------------
            # Run all variants
            # -------------------------
            for v in variants:

                print(f"Running variant: {v}")

                result = run_experiment(v)
                results.append(result)

                print(f"TPS: {result['tps_mean']:.2f}")

            # -------------------------
            # Select best
            # -------------------------
            best_run = max(results, key=lambda x: x["tps_mean"])

            print(f"\nBest TPS this round: {best_run['tps_mean']:.2f}")

            # -------------------------
            # Check improvement
            # -------------------------
            improved = is_improvement(
                best_run["tps_mean"],
                self.state.best_tps,
                self.config.improvement_threshold
            )

            if improved:
                print("Improvement detected ✅")
                self.state.best_tps = best_run["tps_mean"]
                self.state.best_config = best_run["config"]
                self.state.plateau_counter = 0
            else:
                print("No meaningful improvement ❌")
                self.state.plateau_counter += 1

            # -------------------------
            # Plateau detection
            # -------------------------
            if self.state.plateau_counter >= self.config.plateau_patience:
                print("\nPLATEAU DETECTED — stopping early")
                break

            # -------------------------
            # Save iteration log
            # -------------------------
            self._save_iteration(i, results, best_run)

        # -------------------------
        # Final suggestion pass
        # -------------------------
        print("\nGenerating next experiment suggestions...\n")

        run_suggestion_pipeline(
            benchmark_path=self._latest_benchmark_path()
        )

        print("\n==============================")
        print("OPTIMIZATION COMPLETE")
        print("==============================")

        print(f"Best TPS achieved: {self.state.best_tps:.2f}")

    # -----------------------------
    # LOGGING
    # -----------------------------

    def _save_iteration(self, i: int, results: list, best: dict):

        path = os.path.join(
            self.config.experiment_dir,
            f"iteration_{i+1}.json"
        )

        with open(path, "w") as f:
            json.dump({
                "iteration": i + 1,
                "results": results,
                "best": best
            }, f, indent=4)

    # -----------------------------
    # FIND LATEST FILE
    # -----------------------------

    def _latest_benchmark_path(self) -> str:
        files = sorted([
            os.path.join(self.config.experiment_dir, f)
            for f in os.listdir(self.config.experiment_dir)
        ])

        return files[-1] if files else ""


# -----------------------------
# ENTRYPOINT
# -----------------------------

if __name__ == "__main__":

    config = AgentConfig(
        max_iterations=3,  # keep small for safety
        runs_per_experiment=3
    )

    agent = AutonomousOptimizer(config)
    agent.run()
