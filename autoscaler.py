import redis
import subprocess
import time

class AutoScaler:

    def __init__(self, target_workers=3):

        self.r = redis.Redis(decode_responses=True)
        self.target = target_workers

    def queue_depth(self):

        return self.r.llen("experiment_queue")

    def scale(self):

        depth = self.queue_depth()

        print(f"Queue depth: {depth}")

        if depth > 10:

            print("Scaling UP workers")

            subprocess.Popen("python worker.py", shell=True)

        elif depth < 2:

            print("System idle (no scaling)")
