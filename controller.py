import uuid
from queue_manager import QueueManager

class Controller:

    def __init__(self):

        self.queue = QueueManager()

    def generate_experiment(self):

        return {
            "id": str(uuid.uuid4()),
            "prompt": "Explain attention mechanism",
            "provider": "openrouter",
            "max_tokens": 256
        }

    def run(self):

        while True:

            job = self.generate_experiment()

            self.queue.push("experiment_queue", job)

            print(f"Queued experiment {job['id']}")
