import numpy as np
from benchmarks.base import BaseBenchmark

class TriadicBindingEnv(BaseBenchmark):
    """
    7. Triadic object + context + self-state binding.
    Entity 0 is food only when context is [1,0] AND hunger is high.
    Otherwise, Entity 0 is toxic. Entity 1 is always neutral/safe.
    """
    def setup_task(self):
        self.entities[0]["pos"] = np.array([0.5, 0.5], dtype=np.float32)
        self.entities[0]["feats"] = np.array([0.7, 0.7, 0.0, 0.0], dtype=np.float32)
        
        self.entities[1]["pos"] = np.array([-0.5, -0.5], dtype=np.float32)
        self.entities[1]["feats"] = np.array([0.0, 0.0, 0.5, 0.5], dtype=np.float32)
        
        # Randomize context and self_state
        self.context = self.rng.choice([np.array([1.0, 0.0]), np.array([0.0, 1.0])]).astype(np.float32)
        self.self_state = self.rng.choice([np.array([0.9, 0.1]), np.array([0.1, 0.9])]).astype(np.float32)
        
    def step(self, action):
        dx, dy, u, done, info = self.base_step(action)
        reward = 0.0
        
        is_hungry = (self.self_state[0] > 0.5)
        is_context_correct = (self.context[0] > 0.5)
        is_food_safe = is_context_correct and is_hungry
        
        # Entity 0 collision
        if self.entities[0]["active"]:
            if np.linalg.norm(self.agent_pos - self.entities[0]["pos"]) < 0.12:
                self.entities[0]["active"] = False
                if is_food_safe:
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
                reward += 0.1
                done = True
                info["status"] = "success" # safe exit
                
        return self._get_obs(), reward, done, info
