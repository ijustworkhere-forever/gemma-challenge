"""
swarm_leaderboard_server.py

Global leaderboard + memory system for distributed research swarm.

Purpose:
- Aggregate experiment results across runs/providers
- Normalize performance metrics
- Maintain persistent ranking history
- Track best-performing strategies over time
- Provide API-style access for other modules

This is the "long-term memory" layer of the system.
"""

import os
import json
import time
from typing import Dict, List, Any
from datetime import datetime


# =========================================================
# STORAGE LAYER
# =========================================================

class LeaderboardStorage:

    def __init__(self, path: str = "leaderboard.json"):

        self.path = path

        if not os.path.exists(self.path):
            self._init_file()

    def _init_file(self):

        with open(self.path, "w") as f:
            json.dump({
                "runs": [],
                "experiments": [],
                "best": None
            }, f, indent=4)

    def load(self) -> Dict[str, Any]:

        with open(self.path, "r") as f:
            return json.load(f)

    def save(self, data: Dict[str, Any]):

        with open(self.path, "w") as f:
            json.dump(data, f, indent=4)


# =========================================================
# NORMALIZER
# =========================================================

class MetricNormalizer:

    """
    Normalizes across providers so scores are comparable.
    """

    def normalize_latency(self, latency: float, provider: str) -> float:

        # crude but effective normalization model
        # (real systems would use calibrated baselines)

        provider_penalty = {
            "openrouter": 1.0,
            "huggingface": 1.2,
            "vertex": 0.9,
            "local": 0.7
        }.get(provider, 1.0)

        return latency * provider_penalty

    def score(self, latency: float, tokens: int) -> float:

        # throughput-like score (higher is better)
        if latency <= 0:
            return 0.0

        return tokens / latency


# =========================================================
# LEADERBOARD ENGINE
# =========================================================

class SwarmLeaderboard:

    def __init__(self, storage_path: str = "leaderboard.json"):

        self.storage = LeaderboardStorage(storage_path)
        self.normalizer = MetricNormalizer()

    # -----------------------------
    # SUBMIT RESULT
    # -----------------------------

    def submit(self, result: Dict[str, Any]):

        data = self.storage.load()

        provider = result.get("provider", "unknown")
        latency = result.get("latency", 0.0)
        tokens = result.get("max_tokens", 128)

        norm_latency = self.normalizer.normalize_latency(latency, provider)
        score = self.normalizer.score(norm_latency, tokens)

        entry = {
            "timestamp": datetime.now().isoformat(),
            "provider": provider,
            "latency": latency,
            "normalized_latency": norm_latency,
            "tokens": tokens,
            "score": score,
            "prompt": result.get("prompt", ""),
            "task_id": result.get("task_id", "")
        }

        data["experiments"].append(entry)

        # update best
        if data["best"] is None or score > data["best"]["score"]:
            data["best"] = entry

        self.storage.save(data)

        return entry

    # -----------------------------
    # RANKING
    # -----------------------------

    def top_k(self, k: int = 10) -> List[Dict[str, Any]]:

        data = self.storage.load()

        sorted_exps = sorted(
            data["experiments"],
            key=lambda x: x["score"],
            reverse=True
        )

        return sorted_exps[:k]

    # -----------------------------
    # SUMMARY STATS
    # -----------------------------

    def summary(self) -> Dict[str, Any]:

        data = self.storage.load()

        if not data["experiments"]:
            return {}

        scores = [e["score"] for e in data["experiments"]]

        providers = {}

        for e in data["experiments"]:
            p = e["provider"]
            providers.setdefault(p, []).append(e["score"])

        provider_avg = {
            p: sum(v) / len(v)
            for p, v in providers.items()
        }

        return {
            "total_experiments": len(data["experiments"]),
            "best_score": max(scores),
            "avg_score": sum(scores) / len(scores),
            "provider_performance": provider_avg
        }


# =========================================================
# API LAYER (simple local interface)
# =========================================================

class LeaderboardAPI:

    def __init__(self, leaderboard: SwarmLeaderboard):

        self.lb = leaderboard

    def submit_result(self, result: Dict[str, Any]):

        return self.lb.submit(result)

    def get_top(self, k: int = 5):

        return self.lb.top_k(k)

    def get_summary(self):

        return self.lb.summary()


# =========================================================
# OPTIONAL DEMO USAGE
# =========================================================

if __name__ == "__main__":

    lb = SwarmLeaderboard()

    api = LeaderboardAPI(lb)

    # simulate results
    api.submit_result({
        "provider": "openrouter",
        "latency": 0.8,
        "max_tokens": 256,
        "prompt": "Explain attention",
        "task_id": "demo1"
    })

    api.submit_result({
        "provider": "huggingface",
        "latency": 1.2,
        "max_tokens": 256,
        "prompt": "Explain attention",
        "task_id": "demo2"
    })

    print("\nTOP RESULTS:")
    print(api.get_top(5))

    print("\nSUMMARY:")
    print(api.get_summary())
