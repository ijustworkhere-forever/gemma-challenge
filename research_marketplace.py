"""
research_marketplace.py

Self-optimizing research marketplace for distributed inference swarm.

Purpose:
- Track high-performing experiments ("strategies")
- Promote successful strategies across swarm nodes
- Decay poor strategies over time
- Enable reuse and remixing of experiment configurations
- Create evolutionary pressure toward better optimization paths

This is NOT an autonomous execution system.
It is a strategy diffusion + ranking layer.
"""

import os
import json
import time
from typing import Dict, List, Any
from datetime import datetime


# =========================================================
# STORAGE
# =========================================================

class StrategyStore:

    def __init__(self, path: str = "strategy_marketplace.json"):

        self.path = path

        if not os.path.exists(self.path):
            self._init()

    def _init(self):

        with open(self.path, "w") as f:
            json.dump({
                "strategies": [],
                "history": []
            }, f, indent=4)

    def load(self):

        with open(self.path, "r") as f:
            return json.load(f)

    def save(self, data):

        with open(self.path, "w") as f:
            json.dump(data, f, indent=4)


# =========================================================
# STRATEGY MODEL
# =========================================================

class StrategyEngine:

    def create_strategy(self, experiment: Dict[str, Any]) -> Dict[str, Any]:

        return {
            "id": experiment.get("task_id", str(time.time())),
            "provider": experiment.get("provider"),
            "prompt": experiment.get("prompt"),
            "max_tokens": experiment.get("max_tokens"),
            "score": experiment.get("score", 0.0),
            "created_at": datetime.now().isoformat(),
            "usage_count": 0,
            "success_rate": 1.0
        }


# =========================================================
# MARKETPLACE CORE
# =========================================================

class ResearchMarketplace:

    def __init__(self, path: str = "strategy_marketplace.json"):

        self.store = StrategyStore(path)
        self.engine = StrategyEngine()

    # -----------------------------
    # SUBMIT STRATEGY
    # -----------------------------

    def submit(self, experiment: Dict[str, Any]):

        data = self.store.load()

        strategy = self.engine.create_strategy(experiment)

        data["strategies"].append(strategy)

        self.store.save(data)

        return strategy

    # -----------------------------
    # PROMOTE STRATEGIES
    # -----------------------------

    def promote_top(self, top_k: int = 5):

        data = self.store.load()

        strategies = sorted(
            data["strategies"],
            key=lambda x: x["score"],
            reverse=True
        )[:top_k]

        # increase influence weight
        for s in strategies:
            s["usage_count"] += 1

        self.store.save(data)

        return strategies

    # -----------------------------
    # DECAY BAD STRATEGIES
    # -----------------------------

    def decay(self, threshold: float = 0.5):

        data = self.store.load()

        filtered = []

        for s in data["strategies"]:

            # decay rule
            if s["score"] < threshold:
                s["success_rate"] *= 0.95

            # remove if too weak over time
            if s["success_rate"] < 0.2:
                continue

            filtered.append(s)

        data["strategies"] = filtered

        self.store.save(data)

        return len(filtered)

    # -----------------------------
    # RECOMMEND STRATEGIES
    # -----------------------------

    def recommend(self, k: int = 5):

        data = self.store.load()

        if not data["strategies"]:
            return []

        # weighted ranking (score + usage influence)
        ranked = sorted(
            data["strategies"],
            key=lambda x: (
                x["score"] * 0.7 +
                x["usage_count"] * 0.3
            ),
            reverse=True
        )

        return ranked[:k]

    # -----------------------------
    # EVOLUTION STEP
    # -----------------------------

    def evolve(self):

        """
        Simulates "market evolution":
        - promote good strategies
        - decay weak ones
        - reinforce successful patterns
        """

        print("\n==============================")
        print(" STRATEGY MARKET EVOLUTION ")
        print("==============================\n")

        promoted = self.promote_top()
        remaining = self.decay()

        print(f"Promoted strategies: {len(promoted)}")
        print(f"Remaining strategies: {remaining}")

        return {
            "promoted": promoted,
            "remaining": remaining
        }


# =========================================================
# INTEGRATION LAYER (hooks into leaderboard)
# =========================================================

class MarketplaceBridge:

    def __init__(self, marketplace: ResearchMarketplace):

        self.marketplace = marketplace

    def ingest_leaderboard_result(self, leaderboard_entry: Dict[str, Any]):

        """
        Converts leaderboard results into strategies.
        """

        strategy = self.marketplace.submit(leaderboard_entry)

        return strategy


# =========================================================
# DEMO
# =========================================================

if __name__ == "__main__":

    marketplace = ResearchMarketplace()

    bridge = MarketplaceBridge(marketplace)

    # simulate experiments
    bridge.ingest_leaderboard_result({
        "provider": "openrouter",
        "prompt": "Explain attention mechanism",
        "max_tokens": 256,
        "score": 0.92,
        "task_id": "exp_1"
    })

    bridge.ingest_leaderboard_result({
        "provider": "huggingface",
        "prompt": "Explain attention mechanism",
        "max_tokens": 128,
        "score": 0.81,
        "task_id": "exp_2"
    })

    print("\nRECOMMENDATIONS:")
    print(marketplace.recommend())

    marketplace.evolve()
