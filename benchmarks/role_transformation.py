import numpy as np
from benchmarks.base import BaseBenchmark

class RoleTransformationEnv(BaseBenchmark):
    """
    8. Role Transformation / action becoming goal.
    Agent must bypass a blocked wall by outputting action u > 0.5 (open door)
    to transform the blocked state (impasse) and reach the reward behind it.
    """
    def setup_task(self):
        # Door (Entity 0)
        self.entities[0]["pos"] = np.array([0.0, 0.3], dtype=np.float32)
        self.entities[0]["feats"] = np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32)
        
        # Reward (Entity 1)
        self.entities[1]["pos"] = np.array([0.0, 0.8], dtype=np.float32)
        self.entities[1]["feats"] = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32)
        
        self.context = np.array([1.0, 0.0], dtype=np.float32)
        self.door_opened = False
        
    def step(self, action):
        dx, dy, u = action[0], action[1], action[2]
        
        # Clip action limits
        dx = np.clip(dx, -0.15, 0.15)
        dy = np.clip(dy, -0.15, 0.15)
        
        # Check wall block
        target_y = self.agent_pos[1] + dy
        if target_y > 0.25 and not self.door_opened:
            # Check if invention key u > 0.5 is applied
            if u > 0.5:
                self.door_opened = True
                
        if not self.door_opened and target_y > 0.25:
            # Movement is blocked: agent hits wall at y = 0.25
            self.agent_pos[0] = np.clip(self.agent_pos[0] + dx, -1.0, 1.0)
            self.agent_pos[1] = 0.25
        else:
            self.agent_pos += np.array([dx, dy])
            self.agent_pos = np.clip(self.agent_pos, -1.0, 1.0)
            
        # Basic decays
        self.step_count += 1
        self.self_state[0] = np.clip(self.self_state[0] + 0.02, 0.0, 1.0)
        self.self_state[1] = np.clip(self.self_state[1] + 0.01, 0.0, 1.0)
        
        reward = 0.0
        done = False
        info = {"status": "running"}
        
        # Reward contact
        if self.entities[1]["active"]:
            if np.linalg.norm(self.agent_pos - self.entities[1]["pos"]) < 0.12:
                self.entities[1]["active"] = False
                reward += 1.0
                self.self_state[0] = 0.0
                done = True
                info["status"] = "success"
                
        if self.self_state[0] >= 1.0 or self.self_state[1] >= 1.0:
            self.viability = 0.0
            done = True
            info["status"] = "death"
            
        if self.step_count >= self.max_steps:
            done = True
            if info["status"] == "running":
                info["status"] = "timeout"
                
        return self._get_obs(), reward, done, info
