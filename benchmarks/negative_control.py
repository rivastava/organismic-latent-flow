import numpy as np
from benchmarks.base import BaseBenchmark

class NegativeControlEnv(BaseBenchmark):
    """
    4. Negative Control task.
    Consequence mappings are randomized per episode to verify no shortcut leakage.
    """
    def setup_task(self):
        # Place entities
        self.entities[0]["pos"] = np.array([0.5, 0.5], dtype=np.float32)
        self.entities[1]["pos"] = np.array([-0.5, -0.5], dtype=np.float32)
        
        # Completely randomized features
        self.entities[0]["feats"] = self.rng.uniform(-1, 1, 4).astype(np.float32)
        self.entities[1]["feats"] = self.rng.uniform(-1, 1, 4).astype(np.float32)
        
        # Randomly choose which slot is good for this episode
        self.good_index = self.rng.choice([0, 1])
        self.context = np.array([1.0, 0.0], dtype=np.float32)
        
    def step(self, action):
        dx, dy, u, done, info = self.base_step(action)
        reward = 0.0
        
        # Entity 0 collision
        if self.entities[0]["active"]:
            if np.linalg.norm(self.agent_pos - self.entities[0]["pos"]) < 0.12:
                self.entities[0]["active"] = False
                if self.good_index == 0:
                    reward += 1.0
                    self.self_state[0] = 0.0
                    done = True
                    info["status"] = "success"
                else:
                    reward -= 1.0
                    self.viability = 0.0
                    done = True
                    info["status"] = "death"
                    
        # Entity 1 collision
        if self.entities[1]["active"]:
            if np.linalg.norm(self.agent_pos - self.entities[1]["pos"]) < 0.12:
                self.entities[1]["active"] = False
                if self.good_index == 1:
                    reward += 1.0
                    self.self_state[0] = 0.0
                    done = True
                    info["status"] = "success"
                else:
                    reward -= 1.0
                    self.viability = 0.0
                    done = True
                    info["status"] = "death"
                    
        return self._get_obs(), reward, done, info
