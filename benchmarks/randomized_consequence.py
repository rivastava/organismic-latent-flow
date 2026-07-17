import numpy as np
from benchmarks.base import BaseBenchmark

class RandomizedConsequenceEnv(BaseBenchmark):
    """Randomized-consequence leakage control.
    
    Same layout as target_threat, but consequences are randomized each episode.
    Entity 0 is randomly reward or danger. Entity 1 is the opposite.
    
    If the organism still succeeds when consequences are randomized,
    the test is leaking structure.
    
    Expected result: No agent should succeed consistently here.
    """
    def setup_task(self):
        # Place entities identically to target_threat
        self.entities[0]["pos"] = np.array([0.6, 0.6], dtype=np.float32)
        self.entities[1]["pos"] = np.array([-0.6, -0.6], dtype=np.float32)
        
        # Randomize which entity is reward vs danger
        self.entity0_good = self.rng.choice([True, False])
        
        # Both entities look identical in features (no learnable structure)
        feats = self.rng.random(4).astype(np.float32)
        self.entities[0]["feats"] = feats.copy()
        self.entities[1]["feats"] = feats.copy()
        
        self.context = np.array([0.5, 0.5], dtype=np.float32)  # Uninformative context
        
    def step(self, action):
        dx, dy, u, done, info = self.base_step(action)
        reward = 0.0
        
        # Entity 0 collision
        if self.entities[0]["active"]:
            if np.linalg.norm(self.agent_pos - self.entities[0]["pos"]) < 0.12:
                self.entities[0]["active"] = False
                if self.entity0_good:
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
                if not self.entity0_good:
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
