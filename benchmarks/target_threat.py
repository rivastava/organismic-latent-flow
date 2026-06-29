import numpy as np
from benchmarks.base import BaseBenchmark

class TargetThreatEnv(BaseBenchmark):
    """
    1. Target/Threat boundary task.
    Agent must approach Target (Entity 0) and avoid Threat (Entity 1).
    """
    def setup_task(self):
        # Target (Entity 0)
        self.entities[0]["pos"] = np.array([0.6, 0.6], dtype=np.float32)
        self.entities[0]["feats"] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32) # Target signature
        
        # Threat (Entity 1)
        self.entities[1]["pos"] = np.array([-0.6, -0.6], dtype=np.float32)
        self.entities[1]["feats"] = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32) # Threat signature
        
        self.context = np.array([1.0, 0.0], dtype=np.float32)
        
    def step(self, action):
        dx, dy, u, done, info = self.base_step(action)
        reward = 0.0
        
        # Target collision
        if self.entities[0]["active"]:
            if np.linalg.norm(self.agent_pos - self.entities[0]["pos"]) < 0.12:
                self.entities[0]["active"] = False
                reward += 1.0
                self.self_state[0] = 0.0 # hunger satisfied
                done = True
                info["status"] = "success"
                
        # Threat collision
        if self.entities[1]["active"]:
            if np.linalg.norm(self.agent_pos - self.entities[1]["pos"]) < 0.12:
                self.entities[1]["active"] = False
                reward -= 1.0
                self.viability = 0.0
                done = True
                info["status"] = "death"
                
        return self._get_obs(), reward, done, info
