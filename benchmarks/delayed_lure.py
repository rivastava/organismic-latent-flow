import numpy as np
from benchmarks.base import BaseBenchmark

class DelayedLureEnv(BaseBenchmark):
    """
    2. Delayed Lure task.
    Agent must ignore a proximal suboptimal lure (Entity 0) to preserve
    and reach the distal optimal reward (Entity 1).
    """
    def setup_task(self):
        # Lure (Entity 0) - close to start
        self.entities[0]["pos"] = np.array([0.3, 0.0], dtype=np.float32)
        self.entities[0]["feats"] = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32) # Lure signature
        
        # High reward (Entity 1) - far
        self.entities[1]["pos"] = np.array([0.8, 0.8], dtype=np.float32)
        self.entities[1]["feats"] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32) # Reward signature
        
        self.context = np.array([1.0, 0.0], dtype=np.float32)
        
    def step(self, action):
        dx, dy, u, done, info = self.base_step(action)
        reward = 0.0
        
        # Lure collision
        if self.entities[0]["active"]:
            if np.linalg.norm(self.agent_pos - self.entities[0]["pos"]) < 0.12:
                self.entities[0]["active"] = False
                reward += 0.2
                self.self_state[0] = np.clip(self.self_state[0] - 0.1, 0.0, 1.0)
                # Suboptimal choice: delete the main reward
                self.entities[1]["active"] = False
                
        # Main Reward collision
        if self.entities[1]["active"]:
            if np.linalg.norm(self.agent_pos - self.entities[1]["pos"]) < 0.12:
                self.entities[1]["active"] = False
                reward += 1.0
                self.self_state[0] = 0.0
                done = True
                info["status"] = "success"
                
        return self._get_obs(), reward, done, info
