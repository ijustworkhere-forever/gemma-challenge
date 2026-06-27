import time
from controller import Controller

class Scheduler:

    def __init__(self):

        self.controller = Controller()

    def run(self):

        while True:

            print("[SCHEDULER] generating batch...")

            for _ in range(5):
                self.controller.queue_experiment()

            time.sleep(10)
