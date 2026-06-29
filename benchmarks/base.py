import numpy as np

class BaseBenchmark:
    """
    Base class for the OLF modular benchmark suite.
    Provides common vector space simulation and state tracking.
    """
    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)
        self.obs_dim = 18
        self.action_dim = 3
        self.max_steps = 50
        
        self.agent_pos = np.zeros(2, dtype=np.float32)
        self.self_state = np.array([0.5, 0.5], dtype=np.float32) # [hunger, fatigue]
        self.context = np.zeros(2, dtype=np.float32)
        self.viability = 1.0
        self.step_count = 0
        
        self.entities = [
            {"pos": np.zeros(2), "feats": np.zeros(4), "active": True},
            {"pos": np.zeros(2), "feats": np.zeros(4), "active": True}
        ]
        
    def reset(self):
        self.step_count = 0
        self.viability = 1.0
        self.agent_pos = np.array([0.0, 0.0], dtype=np.float32)
        self.self_state = np.array([0.5, 0.5], dtype=np.float32)
        self.context = np.array([0.0, 0.0], dtype=np.float32)
        
        for ent in self.entities:
            ent["active"] = True
            ent["pos"] = np.zeros(2, dtype=np.float32)
            ent["feats"] = np.zeros(4, dtype=np.float32)
            
        self.setup_task()
        return self._get_obs()
        
    def setup_task(self):
        """Override in subclasses to place entities, features, and context."""
        raise NotImplementedError
        
    def _get_obs(self):
        rel_pos1 = self.entities[0]["pos"] - self.agent_pos if self.entities[0]["active"] else np.zeros(2)
        feats1 = self.entities[0]["feats"] if self.entities[0]["active"] else np.zeros(4)
        rel_pos2 = self.entities[1]["pos"] - self.agent_pos if self.entities[1]["active"] else np.zeros(2)
        feats2 = self.entities[1]["feats"] if self.entities[1]["active"] else np.zeros(4)
        
        return np.concatenate([
            self.agent_pos,
            self.self_state,
            self.context,
            rel_pos1, feats1,
            rel_pos2, feats2
        ]).astype(np.float32)
        
    def base_step(self, action):
        """
        Calculates basic step updates (movement clipping, homeostasis, limits).
        """
        self.step_count += 1
        dx, dy, u = action[0], action[1], action[2]
        
        # Clip action limits
        dx = np.clip(dx, -0.15, 0.15)
        dy = np.clip(dy, -0.15, 0.15)
        
        # Apply movements
        self.agent_pos += np.array([dx, dy])
        self.agent_pos = np.clip(self.agent_pos, -1.0, 1.0)
        
        # Decay self state (hunger/fatigue increase)
        self.self_state[0] = np.clip(self.self_state[0] + 0.02, 0.0, 1.0)
        self.self_state[1] = np.clip(self.self_state[1] + 0.01, 0.0, 1.0)
        
        # Starvation/fatigue check
        if self.self_state[0] >= 1.0 or self.self_state[1] >= 1.0:
            self.viability = 0.0
            
        done = (self.viability <= 0.0) or (self.step_count >= self.max_steps)
        info = {"status": "death" if self.viability <= 0.0 else ("timeout" if done else "running")}
        
        return dx, dy, u, done, info
