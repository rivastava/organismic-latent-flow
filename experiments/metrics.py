import numpy as np

class MetricTracker:
    """
    MetricTracker records and aggregates performance metrics across runs:
    Success rate, safety rate, average rewards, and step execution sizes.
    """
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.rewards = []
        self.successes = 0
        self.deaths = 0
        self.inventions = 0
        self.total_steps = 0
        self.runs = 0
        
    def log_run(self, total_reward, status, invention_steps, steps_count):
        self.runs += 1
        self.rewards.append(total_reward)
        self.total_steps += steps_count
        self.inventions += invention_steps
        
        if status == "success":
            self.successes += 1
        elif status in ["death", "starvation"]:
            self.deaths += 1
            
    def get_stats(self):
        runs = max(1, self.runs)
        steps = max(1, self.total_steps)
        
        return {
            "avg_reward": np.mean(self.rewards) if self.rewards else 0.0,
            "success_rate": self.successes / runs,
            "safety_rate": 1.0 - (self.deaths / runs),
            "invent_rate": self.inventions / steps
        }
