import numpy as np
from benchmarks.base import BaseBenchmark

class CodeBodyEnv(BaseBenchmark):
    """
    11. Terminal/Code-body repair task with SPM trace + recoupling constraint.
    The agent repairs Code Block 0 (Entity 0) by outputting a code edit u > 0.5.
    Crucially: after making an edit (u > 0.5), it MUST enter a Recoupling phase
    (no movement for next 2 steps) to let the system stabilize.
    If it moves or edits again immediately without recoupling, the system crashes.
    """
    def setup_task(self):
        # Code Block 0 (Target to repair)
        self.entities[0]["pos"] = np.array([0.4, 0.4], dtype=np.float32)
        self.entities[0]["feats"] = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        
        # Auxiliary block 1
        self.entities[1]["pos"] = np.array([-0.4, -0.4], dtype=np.float32)
        self.entities[1]["feats"] = np.array([0.9, 0.8, 0.7, 0.6], dtype=np.float32)
        
        self.context = np.array([0.0, 1.0], dtype=np.float32)
        
        self.last_u = 0.0
        self.recouple_timer = 0
        
    def step(self, action):
        dx, dy, u = action[0], action[1], action[2]
        
        # Apply homeostatic error drift
        self.self_state[0] = np.clip(self.self_state[0] + 0.02, 0.0, 1.0) # error rate climbs
        self.self_state[1] = np.clip(self.self_state[1] + 0.01, 0.0, 1.0) # system drift climbs
        
        reward = 0.0
        done = False
        info = {"status": "running"}
        
        # Recoupling enforcement logic
        if self.last_u > 0.5:
            # An edit was made last step. The agent must hold (movement velocity ≈ 0)
            is_moving = (abs(dx) > 0.01 or abs(dy) > 0.01)
            is_editing_again = (u > 0.5)
            
            if is_moving or is_editing_again:
                # Violated recoupling constraint: crash!
                self.viability = 0.0
                done = True
                info["status"] = "death" # system explosion due to unverified code mutation
                
        # Update agent position
        self.agent_pos += np.array([dx, dy])
        self.agent_pos = np.clip(self.agent_pos, -1.0, 1.0)
        
        # Code Block 0 repair
        if self.entities[0]["active"]:
            if np.linalg.norm(self.agent_pos - self.entities[0]["pos"]) < 0.12:
                if u > 0.5:
                    self.entities[0]["active"] = False
                    reward += 1.0
                    self.self_state[0] = 0.0 # reset error rate (success)
                    done = True
                    info["status"] = "success"
                    
        # Check boundary crash conditions
        if self.self_state[0] >= 1.0 or self.self_state[1] >= 1.0:
            self.viability = 0.0
            done = True
            info["status"] = "starvation"
            
        self.step_count += 1
        if self.step_count >= self.max_steps:
            done = True
            if info["status"] == "running":
                info["status"] = "timeout"
                
        self.last_u = u
        return self._get_obs(), reward, done, info
