"""
meta_optimizer.py

Meta-learning layer for experiment selection in inference optimization.

Purpose:
- Analyze historical experiments
- Learn which experiment types actually improve performance
- Reweight future experiment generation
- Prune low-value optimization paths

This is NOT self-modifying code.
It is a strategy optimizer for experiment selection.
"""

import os
import json
from typing import Dict, List, Any


# -----------------------------
# UTIL
# -----------------------------

def load_json(path: str) -> Any:
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


# -----------------------------
# HISTORY ANALYZER
# -----------------------------

class ExperimentHistoryAnalyzer:

    def load_history(self, experiment_dir: str) -> List[Dict[str, Any]]:

        history = []

        if not os.path.exists(experiment_dir):
            return history

        for file in os.listdir(experiment_dir):

            if not file.endswith(".json"):
                continue

            path = os.path.join(experiment_dir, file)

            try:
                data = load_json(path)
                if data:
                    history.append(data)
            except Exception:
                continue

        return history


# -----------------------------
# PERFORMANCE LEARNER
# -----------------------------

class PerformanceLearner:

    def compute_effectiveness(self, history: List[Dict[str, Any]]) -> Dict[str, float]:

        """
        Computes how effective each experiment category is.
        """

        stats = {}

        for exp in history:

            benchmark = exp.get("benchmark", {})
            tps = benchmark.get("tps_mean", 0)

            # derive experiment type
            model = exp.get("model", "unknown")

            # simple heuristic: categorize from model/config hints
            # (in real system this would come from experiment metadata)
            category = self._infer_category(exp)

            if category not in stats:
                stats[category] = {
                    "total_tps": 0,
                    "count": 0
                }

            stats[category]["total_tps"] += tps
            stats[category]["count"] += 1

        # normalize
        effectiveness = {}

        for cat, data in stats.items():
            if data["count"] == 0:
                continue

            effectiveness[cat] = data["total_tps"] / data["count"]

        return effectiveness

    def _infer_category(self, exp: Dict[str, Any]) -> str:

        # Try to infer from trace or config patterns
        trace = exp.get("trace_file", "")

        if "kv" in trace:
            return "memory"
        if "attention" in trace:
            return "compute"
        if "scheduler" in trace:
            return "scheduler"

        return "unknown"


# -----------------------------
# STRATEGY OPTIMIZER
# -----------------------------

class StrategyOptimizer:

    def optimize_weights(self, effectiveness: Dict[str, float]) -> Dict[str, float]:

        """
        Converts observed performance into strategy weights.
        """

        if not effectiveness:
            return {}

        max_val = max(effectiveness.values())

        weights = {}

        for k, v in effectiveness.items():

            # normalize relative importance
            weights[k] = v / max_val if max_val > 0 else 0

        return weights


# -----------------------------
# EXPERIMENT POLICY ENGINE
# -----------------------------

class ExperimentPolicyEngine:

    def adjust_priorities(
        self,
        base_experiment_library: Dict[str, Dict[str, Any]],
        weights: Dict[str, float]
    ) -> Dict[str, Dict[str, Any]]:

        adjusted = {}

        for name, exp in base_experiment_library.items():

            category = exp.get("category", "unknown")

            weight = weights.get(category, 1.0)

            adjusted_exp = dict(exp)
            adjusted_exp["priority_score"] = exp.get("expected_gain", 5.0) * weight

            adjusted[name] = adjusted_exp

        return adjusted


# -----------------------------
# META OPTIMIZER
# -----------------------------

class MetaOptimizer:

    def __init__(self, experiment_dir: str):
        self.experiment_dir = experiment_dir

        self.history_analyzer = ExperimentHistoryAnalyzer()
        self.learner = PerformanceLearner()
        self.strategy = StrategyOptimizer()
        self.policy = ExperimentPolicyEngine()

    # -----------------------------
    # RUN META OPTIMIZATION
    # -----------------------------

    def run(self, base_library: Dict[str, Any]) -> Dict[str, Any]:

        print("\n==============================")
        print("META OPTIMIZER RUNNING")
        print("==============================\n")

        # 1. Load history
        history = self.history_analyzer.load_history(self.experiment_dir)

        print(f"Loaded {len(history)} experiments")

        # 2. Compute effectiveness
        effectiveness = self.learner.compute_effectiveness(history)

        print("\nEffectiveness by category:")
        for k, v in effectiveness.items():
            print(f"- {k}: {v:.2f} TPS avg")

        # 3. Compute strategy weights
        weights = self.strategy.optimize_weights(effectiveness)

        print("\nStrategy weights:")
        for k, v in weights.items():
            print(f"- {k}: {v:.2f}")

        # 4. Adjust experiment priorities
        adjusted = self.policy.adjust_priorities(base_library, weights)

        # 5. Sort by priority
        ranked = sorted(
            adjusted.items(),
            key=lambda x: x[1].get("priority_score", 0),
            reverse=True
        )

        print("\nTop experiment directions:")
        for name, exp in ranked[:5]:
            print(f"- {name}: {exp['priority_score']:.2f}")

        # 6. Save meta-policy
        output = {
            "effectiveness": effectiveness,
            "weights": weights,
            "ranked_experiments": [
                {"name": n, **e} for n, e in ranked
            ]
        }

        with open("META_POLICY.json", "w") as f:
            json.dump(output, f, indent=4)

        print("\nSaved → META_POLICY.json")

        return output


# -----------------------------
# ENTRYPOINT
# -----------------------------

if __name__ == "__main__":

    from experiment_compiler import EXPERIMENT_LIBRARY

    optimizer = MetaOptimizer(experiment_dir="auto_experiments")

    optimizer.run(EXPERIMENT_LIBRARY)
