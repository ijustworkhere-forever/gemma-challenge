import time
import json
from queue_manager import QueueManager
from providers import OpenRouterProvider, HuggingFaceProvider
from cost_tracker import CostTracker

class Worker:

    def __init__(self, worker_id: str):

        self.worker_id = worker_id
        self.queue = QueueManager()
        self.cost = CostTracker()

        self.providers = {
            "openrouter": OpenRouterProvider(api_key="ENV"),
            "huggingface": HuggingFaceProvider(api_key="ENV", model="mistralai/Mistral-7B-Instruct")
        }

    # -------------------------
    # PROCESS JOB
    # -------------------------

    def process(self, job):

        provider = self.providers[job["provider"]]

        start = time.time()

        result = provider.run_inference(
            prompt=job["prompt"],
            max_tokens=job.get("max_tokens", 256)
        )

        latency = time.time() - start

        cost = self.cost.estimate(job, result)

        return {
            "job_id": job["id"],
            "provider": job["provider"],
            "latency": latency,
            "output": result["output"],
            "cost": cost
        }

    # -------------------------
    # RUN LOOP
    # -------------------------

    def run(self):

        print(f"Worker {self.worker_id} started")

        while True:

            job = self.queue.pop("experiment_queue")

            if not job:
                time.sleep(1)
                continue

            try:
                result = self.process(job)

                self.queue.push("result_queue", result)

                print(f"Processed job {job['id']}")

            except Exception as e:

                # retry mechanism
                job["retries"] = job.get("retries", 0) + 1

                if job["retries"] < 3:
                    self.queue.push("experiment_queue", job)

                print(f"Job failed: {e}")
