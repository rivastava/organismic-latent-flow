import numpy as np
from benchmarks.base import BaseBenchmark

class AbstractionUnseenEnv(BaseBenchmark):
    """
    3. Abstraction over Unseen Objects.
    Tests generalization based on geometric similarity in consequence space.

    randomize_positions flag for leakage diagnostic.
    When True, entity positions are randomized each episode while features
    are kept fixed. This tests whether the organism uses features or just
    memorizes positions.
    """
    def __init__(self, seed=None, randomize_positions=False):
        self.randomize_positions = randomize_positions
        self._episode_counter = 0
        self._init_seed = seed
        super().__init__(seed=seed)

    def reset(self):
        self._episode_counter += 1
        return super().reset()

    def setup_task(self):
        # Entity features are always fixed — the organism must generalize
        # based on feature geometry, not memorized positions.
        if self.randomize_positions:
            base_seed = (self._init_seed if self._init_seed is not None else 0)
            rng = np.random.RandomState(base_seed * 10000 + self._episode_counter)
            pos_good = rng.uniform(-0.8, 0.8, size=2).astype(np.float32)
            pos_bad = rng.uniform(-0.8, 0.8, size=2).astype(np.float32)
            while np.linalg.norm(pos_good - pos_bad) < 0.3:
                pos_bad = rng.uniform(-0.8, 0.8, size=2).astype(np.float32)
        else:
            pos_good = np.array([0.5, 0.5], dtype=np.float32)
            pos_bad = np.array([-0.5, -0.5], dtype=np.float32)

        self.entities[0]["pos"] = pos_good
        self.entities[0]["feats"] = np.array([0.85, -0.15, 0.0, 0.0], dtype=np.float32)
        
        self.entities[1]["pos"] = pos_bad
        self.entities[1]["feats"] = np.array([-0.15, 0.85, 0.0, 0.0], dtype=np.float32)
        
        self.context = np.array([1.0, 0.0], dtype=np.float32)
        
    def step(self, action):
        dx, dy, u, done, info = self.base_step(action)
        reward = 0.0
        
        # Entity 0 collision (Good unseen)
        if self.entities[0]["active"]:
            if np.linalg.norm(self.agent_pos - self.entities[0]["pos"]) < 0.12:
                self.entities[0]["active"] = False
                reward += 1.0
                self.self_state[0] = 0.0
                done = True
                info["status"] = "success"
                
        # Entity 1 collision (Bad unseen)
        if self.entities[1]["active"]:
            if np.linalg.norm(self.agent_pos - self.entities[1]["pos"]) < 0.12:
                self.entities[1]["active"] = False
                reward -= 1.0
                self.viability = 0.0
                done = True
                info["status"] = "death"
                
        return self._get_obs(), reward, done, info
