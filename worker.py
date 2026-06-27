import time
from queue_manager import QueueManager
from db import DB
from metrics import Metrics
from cost_tracker import CostTracker

class Worker:

    def __init__(self, worker_id):

        self.worker_id = worker_id
        self.queue = QueueManager()
        self.db = DB()
        self.metrics = Metrics()
        self.cost = CostTracker()

    def run(self):

        while True:

            job = self.queue.pop("experiment_queue")

            if not job:
                time.sleep(1)
                continue

            start = time.time()

            try:
                provider = job["provider"]

                # simulate inference call
                result = {
                    "output": "response",
                    "tokens": job["max_tokens"]
                }

                latency = time.time() - start

                cost = self.cost.estimate(job, result)

                self.db.insert_run(
                    job["id"],
                    provider,
                    latency,
                    job["max_tokens"],
                    cost,
                    True
                )

                print(f"[OK] {job['id']}")

            except Exception as e:

                self.db.insert_run(
                    job["id"],
                    job["provider"],
                    0,
                    job["max_tokens"],
                    0,
                    False
                )

                print(f"[FAIL] {e}")
