class CostTracker:

    def __init__(self):

        # rough estimates (you can refine later)
        self.cost_map = {
            "openrouter": 0.00002,  # per token estimate
            "huggingface": 0.00001,
            "vertex": 0.00003
        }

    def estimate(self, job, result):

        provider = job["provider"]
        tokens = job.get("max_tokens", 256)

        return self.cost_map.get(provider, 0.00002) * tokens
