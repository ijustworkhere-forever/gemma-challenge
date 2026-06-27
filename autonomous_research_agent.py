"""
autonomous_research_agent.py

TRUE autonomous research agent mode for Gemma-style optimization swarm.

This system:
- Runs experiments continuously
- Learns from results (meta optimizer)
- Compiles new experiments
- Selects best candidates
- Optionally submits results (hook placeholder)
- Coordinates a persistent research loop

This turns the repo into a "research node".
"""

import os
import time
import json
from datetime import datetime

from meta_optimizer import MetaOptimizer
from experiment_compiler import compile_experiments
from autonomous_optimizer_agent import AutonomousOptimizer
from next_experiment_suggester import run_suggestion_pipeline


# -----------------------------
# CONFIG
# -----------------------------

class AgentConfig:
    def __init__(self):

        self.cycle_delay_sec = 5  # breathing room between cycles

        self.max_cycles = None  # None = infinite

        self.experiment_dir = "auto_experiments"

        self.enable_compiler = True
        self.enable_meta_learning = True
        self.enable_suggestions = True

        self.base_experiment_library = None


# -----------------------------
# RESEARCH STATE
# -----------------------------

class ResearchState:

    def __init__(self):

        self.cycle = 0
        self.best_tps = 0.0
        self.total_experiments = 0

        self.last_improvement_time = None


# -----------------------------
# AUTONOMOUS RESEARCH LOOP
# -----------------------------

class AutonomousResearchAgent:

    def __init__(self, config: AgentConfig):

        self.config = config
        self.state = ResearchState()

        self.meta = MetaOptimizer(config.experiment_dir)

    # -----------------------------
    # SINGLE RESEARCH CYCLE
    # -----------------------------

    def run_cycle(self):

        self.state.cycle += 1

        print("\n====================================")
        print(f"AUTONOMOUS CYCLE {self.state.cycle}")
        print("====================================\n")

        # STEP 1 — META OPTIMIZATION
        print("[1] Running meta-optimizer...")

        meta_result = self.meta.run(
            base_library=self.config.base_experiment_library or {}
        )

        # STEP 2 — COMPILATION (expand search space)
        print("\n[2] Compiling experiments...")

        compile_experiments()

        # STEP 3 — RUN OPTIMIZER LOOP
        print("\n[3] Running autonomous optimizer...")

        optimizer = AutonomousOptimizer(
            config=self._optimizer_config()
        )

        optimizer.run()

        # STEP 4 — GENERATE NEXT SUGGESTIONS
        print("\n[4] Generating next suggestions...")

        run_suggestion_pipeline(
            benchmark_path=self._latest_experiment()
        )

        # STEP 5 — UPDATE STATE
        self._update_state()

    # -----------------------------
    # OPTIMIZER CONFIG
    # -----------------------------

    def _optimizer_config(self):

        from autonomous_optimizer_agent import AgentConfig as OptConfig

        return OptConfig(
            max_iterations=2,
            runs_per_experiment=3
        )

    # -----------------------------
    # GET LATEST EXPERIMENT
    # -----------------------------

    def _latest_experiment(self):

        if not os.path.exists(self.config.experiment_dir):
            return ""

        files = sorted(
            [
                os.path.join(self.config.experiment_dir, f)
                for f in os.listdir(self.config.experiment_dir)
            ]
        )

        return files[-1] if files else ""

    # -----------------------------
    # UPDATE STATE
    # -----------------------------

    def _update_state(self):

        self.state.total_experiments += 1

        self.state.last_improvement_time = datetime.now().isoformat()

        print("\n[STATE UPDATED]")
        print(f"Cycle: {self.state.cycle}")
        print(f"Total experiments: {self.state.total_experiments}")

    # -----------------------------
    # MAIN LOOP
    # -----------------------------

    def run(self):

        print("\n####################################")
        print(" AUTONOMOUS RESEARCH AGENT MODE ")
        print("####################################\n")

        cycle_limit = self.config.max_cycles or 10**9

        for _ in range(cycle_limit):

            try:
                self.run_cycle()

                time.sleep(self.config.cycle_delay_sec)

            except KeyboardInterrupt:
                print("\nStopping agent safely...")
                break

            except Exception as e:
                print(f"\nError in cycle: {e}")
                time.sleep(2)


# -----------------------------
# ENTRYPOINT
# -----------------------------

if __name__ == "__main__":

    config = AgentConfig()

    agent = AutonomousResearchAgent(config)

    agent.run()
