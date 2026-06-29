import numpy as np
from benchmarks.base import BaseBenchmark

class SelfStateMeaningEnv(BaseBenchmark):
    """
    6. Self-state-dependent meaning (v0.3.2.7 — homeostatic consequence grounding).

    Actions change internal body variables.  The usefulness of objects
    depends entirely on the agent's internal needs — not because the
    reward says so, but because the *body consequence* differs.

    Same action + different internal state → different body consequence.
    Reward = body stabilization signal (need reduction), NOT a task label.

    Correct OLF framing:
        same world movement + different internal state
        → different body consequence
        → different latent deformation
        → different future action pressure

    Wrong framing (what we had before):
        if hungry choose food        ← task label reward
        if tired choose bed          ← task label reward
    """
    def setup_task(self):
        self.entities[0]["pos"] = np.array([0.5, 0.5], dtype=np.float32)
        self.entities[0]["feats"] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)  # Food signature

        self.entities[1]["pos"] = np.array([-0.5, -0.5], dtype=np.float32)
        self.entities[1]["feats"] = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)  # Bed signature

        # Pick high hunger [0.9, 0.1] or high fatigue [0.1, 0.9] randomly
        self.self_state = self.rng.choice([np.array([0.9, 0.1]), np.array([0.1, 0.9])]).astype(np.float32)
        self.context = np.array([1.0, 0.0], dtype=np.float32)

    def step(self, action):
        dx, dy, u, done, info = self.base_step(action)
        reward = 0.0

        # --- Entity 0 (Food): body consequence = hunger reduction ---
        # Food reduces hunger ONLY when hunger is high (the need is active).
        # When hunger is already low, food does nothing — body consequence is zero.
        if self.entities[0]["active"]:
            if np.linalg.norm(self.agent_pos - self.entities[0]["pos"]) < 0.12:
                self.entities[0]["active"] = False
                if self.self_state[0] > 0.4:  # hunger is meaningfully high
                    old_hunger = self.self_state[0]
                    reduction = min(0.7, self.self_state[0])
                    self.self_state[0] = max(0.0, self.self_state[0] - reduction)
                    reward = old_hunger - self.self_state[0]

        # --- Entity 1 (Bed): body consequence = fatigue reduction ---
        # Bed reduces fatigue ONLY when fatigue is high (the need is active).
        if self.entities[1]["active"]:
            if np.linalg.norm(self.agent_pos - self.entities[1]["pos"]) < 0.12:
                self.entities[1]["active"] = False
                if self.self_state[1] > 0.4:  # fatigue is meaningfully high
                    old_fatigue = self.self_state[1]
                    reduction = min(0.7, self.self_state[1])
                    self.self_state[1] = max(0.0, self.self_state[1] - reduction)
                    reward = old_fatigue - self.self_state[1]

        # Success = the dominant need that was high is now resolved
        # (e.g. hunger was high → food resolved it, or fatigue was high → bed resolved it)
        if self.self_state[0] > self.self_state[1]:
            # Was hungry: success = hunger resolved
            if self.self_state[0] < 0.3:
                done = True
                info["status"] = "success"
        else:
            # Was tired: success = fatigue resolved
            if self.self_state[1] < 0.3:
                done = True
                info["status"] = "success"

        return self._get_obs(), reward, done, info
