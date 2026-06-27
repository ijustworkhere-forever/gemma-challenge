"""
next_experiment_suggester.py

Autonomous experiment recommendation system.

Purpose:
- Read benchmark + profiling signals
- Identify dominant bottlenecks
- Suggest next highest-impact optimization experiment
- Prioritize based on expected ROI

This is the "decision brain" of the optimization loop.
"""

import os
import json
from typing import Dict, List, Any, Optional


# -----------------------------
# UTIL
# -----------------------------

def load_json(path: str) -> Optional[dict]:
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


# -----------------------------
# EXPERIMENT TYPES
# -----------------------------

EXPERIMENT_LIBRARY = {

    "scheduler_improvement": {
        "goal": "Increase GPU utilization by reducing idle time",
        "expected_gain": 5,
        "category": "cpu_scheduler",
    },

    "cuda_graphs_decode": {
        "goal": "Reduce kernel launch overhead in decode loop",
        "expected_gain": 8,
        "category": "latency",
    },

    "kv_cache_optimization": {
        "goal": "Improve memory access pattern in KV cache",
        "expected_gain": 10,
        "category": "memory",
    },

    "attention_kernel_tuning": {
        "goal": "Optimize FlashAttention / Triton kernel performance",
        "expected_gain": 12,
        "category": "compute",
    },

    "mlp_fusion": {
        "goal": "Fuse GEMM + activation in MLP block",
        "expected_gain": 6,
        "category": "compute",
    },

    "sampling_gpu_offload": {
        "goal": "Move sampling from CPU → GPU",
        "expected_gain": 4,
        "category": "cpu",
    },

    "batching_strategy_tuning": {
        "goal": "Improve throughput via dynamic batching",
        "expected_gain": 7,
        "category": "scheduler",
    },
}


# -----------------------------
# ANALYZER
# -----------------------------

class BottleneckReader:

    def infer_bottleneck(self, benchmark: Dict, trace: Dict = None) -> str:

        tps = benchmark.get("benchmark", {}).get("tps_mean", 0)

        # simple heuristic fallback (can be replaced by nsight_parser output)
        if trace:
            bottleneck = trace.get("bottleneck_type", "unknown")
            if bottleneck != "unknown":
                return bottleneck

        # fallback heuristic
        if tps < 20:
            return "underutilized_gpu_cpu_or_scheduler_issue"

        if tps < 50:
            return "memory_or_attention_bottleneck"

        return "compute_bound_or_balanced"


# -----------------------------
# SCORER
# -----------------------------

class ExperimentScorer:

    def score(self, bottleneck: str) -> List[Dict[str, Any]]:

        ranked = []

        for name, exp in EXPERIMENT_LIBRARY.items():

            score = exp["expected_gain"]

            # boost matching category relevance
            if bottleneck in exp["category"]:
                score += 5

            ranked.append({
                "experiment": name,
                "score": score,
                "goal": exp["goal"],
                "expected_gain": exp["expected_gain"]
            })

        return sorted(ranked, key=lambda x: x["score"], reverse=True)


# -----------------------------
# REPORT GENERATOR
# -----------------------------

class SuggestionReport:

    def generate(self, bottleneck: str, ranked: List[Dict]) -> str:

        lines = []

        lines.append("# NEXT EXPERIMENT SUGGESTIONS\n")
        lines.append(f"## Detected Bottleneck: {bottleneck}\n")

        lines.append("## Ranked Experiments\n")

        for i, exp in enumerate(ranked[:5]):

            lines.append(f"### {i+1}. {exp['experiment']}")
            lines.append(f"- Expected Gain: +{exp['expected_gain']}%")
            lines.append(f"- Score: {exp['score']}")
            lines.append(f"- Goal: {exp['goal']}\n")

        lines.append("\n---\n")
        lines.append("## Recommended Next Step\n")
        lines.append(f"👉 **{ranked[0]['experiment']}**\n")
        lines.append(f"{ranked[0]['goal']}\n")

        return "\n".join(lines)


# -----------------------------
# MAIN PIPELINE
# -----------------------------

def run_suggestion_pipeline(
    benchmark_path: str,
    trace_path: str = None
):

    benchmark = load_json(benchmark_path)

    trace = None
    if trace_path:
        trace = load_json(trace_path)

    reader = BottleneckReader()
    scorer = ExperimentScorer()
    reporter = SuggestionReport()

    bottleneck = reader.infer_bottleneck(benchmark, trace)
    ranked = scorer.score(bottleneck)

    report = reporter.generate(bottleneck, ranked)

    print("\n==============================")
    print("NEXT EXPERIMENT SUGGESTIONS")
    print("==============================\n")

    print(report)

    # optionally write to file
    with open("NEXT_EXPERIMENT.md", "w") as f:
        f.write(report)


# -----------------------------
# ENTRYPOINT
# -----------------------------

if __name__ == "__main__":

    import sys

    if len(sys.argv) < 2:
        print("Usage: python next_experiment_suggester.py <benchmark.json> [trace.json]")
        exit(1)

    benchmark_path = sys.argv[1]
    trace_path = sys.argv[2] if len(sys.argv) > 2 else None

    run_suggestion_pipeline(benchmark_path, trace_path)
