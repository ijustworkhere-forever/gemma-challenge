"""
experiment_compiler.py

Experiment synthesis system for inference optimization.

Purpose:
- Convert IDEAS.md + profiling signals into structured experiment configs
- Expand search space safely (no arbitrary code generation)
- Feed autonomous optimizer with new meaningful variants

This is NOT a code generator.
It is a structured experiment design system.
"""

import os
import re
import json
from typing import List, Dict, Any, Optional


# -----------------------------
# IDEA PARSER
# -----------------------------

class IdeaExtractor:

    def extract(self, ideas_path: str = "IDEAS.md") -> List[str]:

        if not os.path.exists(ideas_path):
            return []

        with open(ideas_path, "r") as f:
            content = f.read()

        # Extract bullet-style ideas
        ideas = re.findall(r"- (.+)", content)

        return ideas


# -----------------------------
# EXPERIMENT SPEC FORMAT
# -----------------------------

"""
We define a SAFE experiment schema:

{
    "name": "...",
    "type": "scheduler | memory | compute | latency",
    "params": {
        "batch_size_delta": int,
        "max_tokens_delta": int,
        "warmup_delta": int,
        "enable_cuda_graphs": bool,
        "enable_fusion": bool
    },
    "expected_gain": float
}
"""


# -----------------------------
# EXPERIMENT COMPILER
# -----------------------------

class ExperimentCompiler:

    def compile_from_ideas(self, ideas: List[str]) -> List[Dict[str, Any]]:

        experiments = []

        for idea in ideas:

            idea_lower = idea.lower()

            # -----------------------------
            # CUDA Graphs related
            # -----------------------------
            if "cuda graph" in idea_lower:
                experiments.append({
                    "name": "cuda_graphs_decode_expanded",
                    "type": "latency",
                    "params": {
                        "enable_cuda_graphs": True,
                        "batch_size_delta": 0,
                        "max_tokens_delta": 0,
                        "warmup_delta": 2
                    },
                    "expected_gain": 8.0
                })

            # -----------------------------
            # KV cache / memory
            # -----------------------------
            elif "kv cache" in idea_lower:
                experiments.append({
                    "name": "kv_cache_layout_tuning",
                    "type": "memory",
                    "params": {
                        "batch_size_delta": 1,
                        "max_tokens_delta": 0,
                        "warmup_delta": 1
                    },
                    "expected_gain": 10.0
                })

            # -----------------------------
            # Attention kernel tuning
            # -----------------------------
            elif "attention" in idea_lower:
                experiments.append({
                    "name": "attention_kernel_tuning_expanded",
                    "type": "compute",
                    "params": {
                        "enable_fusion": True,
                        "batch_size_delta": 1,
                        "max_tokens_delta": -32,
                        "warmup_delta": 1
                    },
                    "expected_gain": 12.0
                })

            # -----------------------------
            # Scheduler / batching
            # -----------------------------
            elif "batch" in idea_lower or "scheduler" in idea_lower:
                experiments.append({
                    "name": "scheduler_batching_tuning",
                    "type": "scheduler",
                    "params": {
                        "batch_size_delta": 2,
                        "max_tokens_delta": 0,
                        "warmup_delta": 1
                    },
                    "expected_gain": 7.0
                })

            # -----------------------------
            # CPU / sampling issues
            # -----------------------------
            elif "cpu" in idea_lower or "sampling" in idea_lower:
                experiments.append({
                    "name": "sampling_offload_experiment",
                    "type": "cpu",
                    "params": {
                        "batch_size_delta": 0,
                        "max_tokens_delta": 0,
                        "warmup_delta": 1
                    },
                    "expected_gain": 4.0
                })

        return experiments


# -----------------------------
# EXPERIMENT EXPANDER
# -----------------------------

class ExperimentExpander:

    def expand(self, experiments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:

        expanded = []

        for exp in experiments:

            base = exp["params"]

            # Variant A: conservative tuning
            conservative = {
                **exp,
                "name": exp["name"] + "_conservative",
                "params": {
                    **base,
                    "batch_size_delta": base.get("batch_size_delta", 0),
                    "max_tokens_delta": base.get("max_tokens_delta", 0)
                }
            }

            # Variant B: aggressive tuning
            aggressive = {
                **exp,
                "name": exp["name"] + "_aggressive",
                "params": {
                    **base,
                    "batch_size_delta": base.get("batch_size_delta", 0) + 1,
                    "max_tokens_delta": base.get("max_tokens_delta", 0) - 16
                }
            }

            expanded.append(conservative)
            expanded.append(aggressive)

        return expanded


# -----------------------------
# SAFETY FILTER
# -----------------------------

class ExperimentSafetyFilter:

    def filter(self, experiments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:

        safe = []

        for exp in experiments:

            p = exp["params"]

            # Hard constraints (prevents instability)

            if p.get("batch_size_delta", 0) > 4:
                continue

            if p.get("max_tokens_delta", 0) < -128:
                continue

            if p.get("warmup_delta", 0) > 5:
                continue

            safe.append(exp)

        return safe


# -----------------------------
# PIPELINE
# -----------------------------

def compile_experiments(
    ideas_path: str = "IDEAS.md",
    output_path: str = "compiled_experiments.json"
):

    extractor = IdeaExtractor()
    compiler = ExperimentCompiler()
    expander = ExperimentExpander()
    safety = ExperimentSafetyFilter()

    ideas = extractor.extract(ideas_path)

    print("\n=== EXTRACTED IDEAS ===")
    for i in ideas:
        print("-", i)

    experiments = compiler.compile_from_ideas(ideas)
    expanded = expander.expand(experiments)
    safe = safety.filter(expanded)

    with open(output_path, "w") as f:
        json.dump(safe, f, indent=4)

    print("\n=== COMPILED EXPERIMENTS ===")
    for e in safe:
        print(e["name"])

    print(f"\nSaved → {output_path}")

    return safe


# -----------------------------
# ENTRYPOINT
# -----------------------------

if __name__ == "__main__":

    compile_experiments()
