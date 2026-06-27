from prometheus_client import Gauge, start_http_server

queue_depth = Gauge("queue_depth", "Number of queued jobs")
avg_latency = Gauge("avg_latency", "Average latency of workers")
total_cost = Gauge("total_cost", "Total API cost")

class Metrics:

    def __init__(self):

        start_http_server(8000)

    def update_queue_depth(self, value):
        queue_depth.set(value)

    def update_latency(self, value):
        avg_latency.set(value)

    def update_cost(self, value):
        total_cost.set(value)
