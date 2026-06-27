import redis
import json
import time

class QueueManager:

    def __init__(self, host="localhost", port=6379):

        self.r = redis.Redis(host=host, port=port, decode_responses=True)

    # -------------------------
    # PUSH JOB
    # -------------------------

    def push(self, queue_name: str, job: dict):

        self.r.lpush(queue_name, json.dumps(job))

    # -------------------------
    # POP JOB (blocking)
    # -------------------------

    def pop(self, queue_name: str, timeout=5):

        result = self.r.brpop(queue_name, timeout=timeout)

        if result:
            return json.loads(result[1])

        return None
