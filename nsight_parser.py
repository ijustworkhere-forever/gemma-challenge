"""
nsight_parser.py

Post-processing tool for Nsight Systems traces.

Purpose:
- Extract performance signals from GPU traces
- Identify bottlenecks (compute, memory, latency, CPU, sync)
- Generate structured optimization ideas for IDEAS.md

This is NOT a full Nsight SDK replacement.
It assumes exported JSON or simplified trace summaries.
"""

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Any, Optional


# -----------------------------
# DATA MODELS
# -----------------------------

@dataclass
class KernelEvent:
    name: str
    duration_ms: float


@dataclass
class TraceSummary:
    total_gpu_time_ms: float
    total_idle_time_ms: float
    kernel_count: int
    top_kernels: List[KernelEvent]


# -----------------------------
# TRACE LOADER
# -----------------------------

class NsightTraceLoader:
    """
    Loads simplified Nsight export (JSON format recommended).

    Expected format (example):
    {
        "gpu_time_ms": 1200,
        "idle_time_ms": 300,
        "kernels": [
            {"name": "attention_kernel", "duration_ms": 5.2},
            {"name": "mlp_kernel", "duration_ms": 3.1}
        ]
    }
    """

    def load(self, path: str) -> TraceSummary:
        with open(path, "r") as f:
            data = json.load(f)

        kernels = [
            KernelEvent(k["name"], k["duration_ms"])
            for k in data.get("kernels", [])
        ]

        kernels_sorted = sorted(
            kernels,
            key=lambda x: x.duration_ms,
            reverse=True
        )

        return TraceSummary(
            total_gpu_time_ms=data.get("gpu_time_ms", 0),
            total_idle_time_ms=data.get("idle_time_ms", 0),
            kernel_count=len(kernels),
            top_kernels=kernels_sorted[:5]
        )


# -----------------------------
# BOTTLENECK ANALYZER
# -----------------------------

class BottleneckAnalyzer:

    def analyze(self, trace: TraceSummary) -> Dict[str, Any]:

        gpu_utilization = self._gpu_util(trace)
        idle_ratio = self._idle_ratio(trace)

        bottleneck_type = self._classify(trace, gpu_utilization, idle_ratio)

        return {
            "gpu_utilization": gpu_utilization,
            "idle_ratio": idle_ratio,
            "bottleneck_type": bottleneck_type,
            "top_kernels": [
                {"name": k.name, "ms": k.duration_ms}
                for k in trace.top_kernels
            ]
        }

    def _gpu_util(self, trace: TraceSummary) -> float:
        total = trace.total_gpu_time_ms + trace.total_idle_time_ms
        if total == 0:
            return 0.0
        return trace.total_gpu_time_ms / total

    def _idle_ratio(self, trace: TraceSummary) -> float:
        total = trace.total_gpu_time_ms + trace.total_idle_time_ms
        if total == 0:
            return 0.0
        return trace.total_idle_time_ms / total

    def _classify(
        self,
        trace: TraceSummary,
        gpu_util: float,
        idle_ratio: float
    ) -> str:

        # -----------------------------
        # CPU / Scheduler bottleneck
        # -----------------------------
        if gpu_util < 0.80:
            return "underutilized_gpu_cpu_or_scheduler_issue"

        # -----------------------------
        # Latency-bound (too many kernels)
        # -----------------------------
        if trace.kernel_count > 200:
            return "kernel_launch_overhead_latency_bound"

        # -----------------------------
        # Memory-bound heuristic
        # -----------------------------
        top_kernel = trace.top_kernels[0] if trace.top_kernels else None
        if top_kernel and "attention" in top_kernel.name.lower():
            if top_kernel.duration_ms > 5.0:
                return "likely_memory_or_attention_bottleneck"

        # -----------------------------
        # Idle time dominance
        # -----------------------------
        if idle_ratio > 0.3:
            return "gpu_starvation_or_sync_issue"

        return "compute_bound_or_balanced"


# -----------------------------
# IDEA GENERATOR
# -----------------------------

class IdeaGenerator:

    def generate(self, analysis: Dict[str, Any]) -> str:

        bottleneck = analysis["bottleneck_type"]
        gpu_util = analysis["gpu_utilization"]
        idle = analysis["idle_ratio"]

        ideas = []

        # -----------------------------
        # GPU underutilized
        # -----------------------------
        if bottleneck == "underutilized_gpu_cpu_or_scheduler_issue":
            ideas.append(
                "- Scheduler inefficiency likely → investigate batching strategy"
            )
            ideas.append(
                "- CPU may be blocking GPU → profile tokenizer/sampling"
            )

        # -----------------------------
        # Kernel launch overhead
        # -----------------------------
        if bottleneck == "kernel_launch_overhead_latency_bound":
            ideas.append(
                "- Too many small kernels → consider CUDA Graphs"
            )
            ideas.append(
                "- Decode loop likely fragmented → investigate fusion"
            )

        # -----------------------------
        # Memory / attention bottleneck
        # -----------------------------
        if bottleneck == "likely_memory_or_attention_bottleneck":
            ideas.append(
                "- Attention kernel is dominant → tune FlashAttention / Triton kernel"
            )
            ideas.append(
                "- KV cache access pattern may be inefficient"
            )

        # -----------------------------
        # GPU starvation
        # -----------------------------
        if bottleneck == "gpu_starvation_or_sync_issue":
            ideas.append(
                "- GPU idle time high → likely CPU sync or scheduler stalls"
            )
            ideas.append(
                "- Investigate async pipeline + queue depth"
            )

        # -----------------------------
        # Default fallback
        # -----------------------------
        if not ideas:
            ideas.append("- System appears balanced → look for micro-optimizations")

        return "\n".join(ideas)


# -----------------------------
# IDEAS.md WRITER
# -----------------------------

class IdeasWriter:

    def append(self, ideas_text: str, output_path: str = "IDEAS.md"):

        if not os.path.exists(output_path):
            content = "# IDEAS.md\n\n"
        else:
            with open(output_path, "r") as f:
                content = f.read()

        content += "\n\n## Auto-generated Ideas\n\n"
        content += ideas_text
        content += "\n"

        with open(output_path, "w") as f:
            f.write(content)


# -----------------------------
# PIPELINE ENTRYPOINT
# -----------------------------

def run_trace_analysis(trace_file: str):

    loader = NsightTraceLoader()
    analyzer = BottleneckAnalyzer()
    generator = IdeaGenerator()
    writer = IdeasWriter()

    trace = loader.load(trace_file)
    analysis = analyzer.analyze(trace)
    ideas = generator.generate(analysis)

    print("\n=== NSIGHT ANALYSIS ===")
    print(json.dumps(analysis, indent=2))

    print("\n=== GENERATED IDEAS ===")
    print(ideas)

    writer.append(ideas)


# -----------------------------
# CLI
# -----------------------------

if __name__ == "__main__":

    import sys

    if len(sys.argv) < 2:
        print("Usage: python nsight_parser.py <trace.json>")
        exit(1)

    run_trace_analysis(sys.argv[1])
