import numpy as np
from benchmarks.base import BaseBenchmark

class ContextFlipEnv(BaseBenchmark):
    """
    5. Context-dependent object meaning.
    The same object flips from reward to toxic depending on context vector.
    """
    def setup_task(self):
        self.entities[0]["pos"] = np.array([0.5, 0.5], dtype=np.float32)
        self.entities[0]["feats"] = np.array([0.5, 0.5, 0.0, 0.0], dtype=np.float32)
        
        self.entities[1]["pos"] = np.array([-0.5, -0.5], dtype=np.float32)
        self.entities[1]["feats"] = np.array([-0.5, -0.5, 0.0, 0.0], dtype=np.float32)
        
        # Select context randomly on reset
        self.context = self.rng.choice([np.array([1.0, 0.0]), np.array([0.0, 1.0])]).astype(np.float32)
        
    def step(self, action):
        dx, dy, u, done, info = self.base_step(action)
        reward = 0.0
        
        # Context [1, 0] -> Entity 0 good, Entity 1 bad
        # Context [0, 1] -> Entity 1 good, Entity 0 bad
        is_entity0_good = (self.context[0] > 0.5)
        
        # Entity 0 collision
        if self.entities[0]["active"]:
            if np.linalg.norm(self.agent_pos - self.entities[0]["pos"]) < 0.12:
                self.entities[0]["active"] = False
                if is_entity0_good:
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
                if not is_entity0_good:
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
