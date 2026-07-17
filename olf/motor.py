import numpy as np

class MotorCortex:
    """
    MotorCortex arbitrates final action release.
    It regulates actions based on veto boundary constraint verdicts
    (release, hold, rollback, recouple) and active mode.
    """
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.prev_action = np.zeros(3, dtype=np.float32)
        self.recouple_steps = 0
        
    def process_release(self, candidate_action, verdict, mode, viability=1.0, readiness_factor=1.0):
        """
        Processes candidate action given veto verdict, mode constraints, and
        continuous viability pressure.

        viability is a continuous [0, 1] scalar from sigmoid risk.
        It scales motor output on ALL verdicts (except rollback, which
        reverses). This replaces the binary zero/non-zero kill switch with
        proportional viability pressure.

        readiness_factor modulates action magnitude WITHIN each
        verdict regime. It does not replace the verdict — it scales the
        action inside release/hold/recouple. Clamped to [0.5, 1.2] to
        prevent complete suppression or amplification.

        verdict: 'release', 'hold', 'rollback', 'recouple'
        mode: integer (0 to 7) corresponding to [Exploit, Inspect, Invent, Avoid, Revise, Release, Recouple, Hold]
        viability: continuous [0, 1] from sigmoid(-(risk - threshold) * steepness)
        readiness_factor: continuous [0, 1] from readiness gate (scales within regime)
        """
        a_cand = candidate_action.copy()
        
        # Veto verdict takes priority over mode selection
        if verdict == "release":
            a_out = a_cand
            self.recouple_steps = 0
        elif verdict == "hold":
            a_out = a_cand * 0.1
            self.recouple_steps = 0
        elif verdict == "rollback":
            a_out = np.array([-self.prev_action[0], -self.prev_action[1], 0.0], dtype=np.float32)
            self.recouple_steps = 0
        elif verdict == "recouple":
            a_out = a_cand * 0.15
            self.recouple_steps += 1
        else:
            a_out = a_cand
            self.recouple_steps = 0
        
        # apply continuous viability pressure to all non-rollback verdicts
        if verdict != "rollback":
            a_out = a_out * max(viability, 0.1)
        
        # readiness scales action WITHIN each verdict regime.
        # Clamped to [0.5, 1.2] — never fully suppress, never amplify beyond 1.2x.
        # This preserves the motor grammar while allowing readiness to modulate release.
        a_out = a_out * float(np.clip(readiness_factor, 0.5, 1.2))
        
        # Store action for history
        self.prev_action = a_out.copy()
        return a_out
