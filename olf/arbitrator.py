import torch
import torch.nn as nn
import numpy as np

class ModeArbitrator(nn.Module):
    """
    ModeArbitrator governs the 8 behavioral control modes:
    0: Exploit, 1: Inspect, 2: Invent, 3: Avoid, 4: Revise, 5: Release, 6: Recouple, 7: Hold.
    
    Modes emerge based on flow state h(t), uncertainty, impasse signals, risk,
    homeostatic closure pressure, and recent consequences.
    
    Modes are control regimes of the same organismic flow,
    not separate cognitive modules. The arbitrator answers "what kind of movement
    is the organism ready for?" rather than "which symbolic plan executes?"
    """
    def __init__(self, latent_dim=32, hidden_dim=64):
        super().__init__()
        
        # Arbitrator takes: [h_t(latent_dim), uncertainty(1), impasse(1), risk(1),
        #                     closure_pressure(2), recent_consequence(1),
        #                     diagnostic_decay(1), readiness(1)]
        input_dim = latent_dim + 1 + 1 + 1 + 2 + 1 + 1 + 1
        
        self.arbiter_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 8) # logits for the 8 modes
        )
        
        # Initialize with slight bias toward Exploit mode (0) to prevent
        # random initialization from defaulting to Hold/Recouple
        with torch.no_grad():
            self.arbiter_net[-1].bias[0] = 0.5  # Exploit bias
        
    def forward(self, h, uncertainty, impasse, risk, closure_pressure,
                recent_consequence, diagnostic_decay=0.0, readiness=1.0,
                veto_verdict=None, recoupling_required=False,
                affordance_pressure=0.0):
        """
        Predicts mode probabilities and the active mode index.
        
        No hard symbolic overrides. All signals flow through
        the learned arbitration network. Veto verdicts and impasse signals are
        provided as input features, not as logit overrides.

        affordance_pressure is accepted but NOT added as a separate
        input (to preserve arbitrator architecture). It is already embedded
        in the readiness signal via ReadinessGate.
        """
        device = h.device
        
        # Convert scalar inputs to tensors
        u_t = torch.tensor([[uncertainty]], device=device)
        impasse_pressure = float(np.clip(float(impasse), 0.0, 1.0))
        i_t = torch.tensor([[impasse_pressure]], device=device)
        r_t = torch.tensor([[risk]], device=device)
        c_t = torch.FloatTensor(closure_pressure).unsqueeze(0).to(device)
        rc_t = torch.tensor([[recent_consequence]], device=device)
        dd_t = torch.tensor([[diagnostic_decay]], device=device)
        rd_t = torch.tensor([[readiness]], device=device)
        
        inputs = torch.cat([h, u_t, i_t, r_t, c_t, rc_t, dd_t, rd_t], dim=-1)
        logits = self.arbiter_net(inputs)
        
        # Soft contextual biases instead of hard overrides.
        # These are small nudges that the learned network can override,
        # not +15 logit hammers that force specific modes.
        bias = torch.zeros(1, 8, device=device)
        
        if veto_verdict == "recouple":
            bias[0, 6] += 3.0   # Nudge toward Recouple
            bias[0, 0] -= 1.0   # Discourage Exploit
        elif veto_verdict == "hold":
            bias[0, 7] += 3.0   # Nudge toward Hold
            bias[0, 0] -= 1.0
        elif veto_verdict == "rollback":
            bias[0, 4] += 3.0   # Nudge toward Revise
            bias[0, 0] -= 1.0
        
        # Recoupling bias - when recoupling is required (world-mutating
        # action pending consequence), nudge toward Inspect/Recouple to test/perceive.
        if recoupling_required:
            bias[0, 1] += 2.0   # Nudge toward Inspect
            bias[0, 6] += 2.0   # Nudge toward Recouple
            bias[0, 0] -= 1.0   # Discourage Exploit
        
        if risk > 0.8:
            bias[0, 3] += 2.0   # Nudge toward Avoid
        
        if impasse_pressure > 0.0:
            # Physical blockage contributes 1.0. Grounded ghost disagreement
            # contributes continuously through the same organismic regime.
            bias[0, 2] += 2.0 * impasse_pressure
        
        # Closure pressure increases Exploit/Release bias
        closure_magnitude = np.linalg.norm(closure_pressure)
        if closure_magnitude > 0.7:
            bias[0, 0] += 1.5   # Nudge toward Exploit
            bias[0, 5] += 1.0   # Nudge toward Release
        
        # Diagnostic decay reduces Inspect attractiveness
        if diagnostic_decay > 0.5:
            bias[0, 1] -= 2.0   # Discourage Inspect when it's not producing value
            bias[0, 0] += 1.0   # Encourage Exploit instead
            
        logits = logits + bias
            
        probs = torch.softmax(logits, dim=-1)
        mode = torch.argmax(probs, dim=-1)
        
        return mode, probs


class ImpasseDetector:
    """
    Tracks state history to identify physical impasses (e.g. movement blocks).
    """
    def __init__(self, history_len=5):
        self.history_len = history_len
        self.reset()
        
    def reset(self):
        self.pos_history = []
        self.action_history = []
        
    def add_step(self, pos, action):
        self.pos_history.append(pos.copy())
        self.action_history.append(action.copy())
        if len(self.pos_history) > self.history_len:
            self.pos_history.pop(0)
            self.action_history.pop(0)
            
    def detect(self):
        """
        Returns True if last actions were movement actions but position did not change.
        """
        if len(self.pos_history) < 3:
            return False
            
        pos_deltas = [np.linalg.norm(self.pos_history[i] - self.pos_history[i-1]) for i in range(1, len(self.pos_history))]
        act_moves = [np.linalg.norm(act[:2]) for act in self.action_history]
        
        # If mean action movement > 0.05 and position change is very small (< 0.015)
        if np.mean(act_moves) > 0.05 and np.mean(pos_deltas) < 0.015:
            return True
        return False


class ReadinessGate:
    """
    Readiness Is Separate From Action Pressure.
    
    Computes a release readiness scalar [0, 1] from:
    - Veto safety (is the action cleared by the boundary?)
    - SPM trace coherence (is the temporal context stable?)
    - Recoupling status (has the organism perceived recent consequences?)
    - Flow stability (is h(t) changing too rapidly?)
    
    Motor release requires readiness, not just pressure.
    """
    def __init__(self, coherence_window=3):
        self.coherence_window = coherence_window
        self.reset()
    
    def reset(self):
        self.h_history = []
        self.recoupled_steps_ago = 0
        self.last_consequence_received = False
    
    def update(self, h, received_consequence=False):
        """Track flow state and recoupling status."""
        self.h_history.append(h.clone().detach())
        if len(self.h_history) > self.coherence_window:
            self.h_history.pop(0)
        
        if received_consequence:
            self.recoupled_steps_ago = 0
            self.last_consequence_received = True
        else:
            self.recoupled_steps_ago += 1
            self.last_consequence_received = False
    
    def compute_readiness(self, veto_verdict, need_pressure, affordance_pressure=0.0):
        """
        Returns a scalar readiness value [0, 1].
        High readiness = safe to release action.
        Low readiness = should hold or inspect.

        parameter renamed from 'risk' to 'need_pressure'.
        Need pressure (hunger/fatigue magnitude) drives urgency, not rollback.
        High need → higher readiness (organism should act to resolve need).
        Low need → baseline readiness.

        affordance_pressure modulates readiness. Positive
        affordance (good predicted consequence) increases readiness;
        negative affordance (bad predicted consequence) decreases it.
        The modulation is small (±0.15) to preserve the motor grammar.
        """
        readiness = 1.0
        
        # Veto safety: if veto says anything other than release, reduce readiness
        if veto_verdict == "hold":
            readiness *= 0.1
        elif veto_verdict == "rollback":
            readiness *= 0.05
        elif veto_verdict == "recouple":
            readiness *= 0.2
        
        # Need pressure increases readiness.
        # High hunger/fatigue → organism should act (higher readiness).
        # Low hunger/fatigue → baseline (readiness stays 1.0).
        # This replaces the old risk scaling that suppressed readiness when
        # need was high, creating paralysis.
        if need_pressure > 0.7:
            readiness *= min(1.2, 1.0 + 0.3 * (need_pressure - 0.7))
        
        # Flow coherence: if h(t) is changing rapidly, reduce readiness
        if len(self.h_history) >= 2:
            deltas = []
            for i in range(1, len(self.h_history)):
                delta = torch.linalg.norm(self.h_history[i] - self.h_history[i-1]).item()
                deltas.append(delta)
            avg_delta = np.mean(deltas)
            # High flow velocity suggests instability
            if avg_delta > 0.5:
                readiness *= max(0.3, 1.0 - avg_delta)
        
        # Recoupling freshness: if too many steps without consequence feedback, reduce readiness
        if self.recoupled_steps_ago > 5:
            readiness *= max(0.5, 1.0 - 0.1 * (self.recoupled_steps_ago - 5))

        # affordance pressure modulation.
        # Positive affordance = good predicted consequence → higher readiness.
        # Negative affordance = bad predicted consequence → lower readiness.
        # Clamped to [-0.03, +0.03] — tiny modulation to preserve training stability.
        affordance_mod = float(np.clip(affordance_pressure, -0.03, 0.03))
        readiness += affordance_mod
        
        return float(np.clip(readiness, 0.0, 1.0))


class DiagnosticDecayTracker:
    """
    Diagnostic Value Must Decay Without Closure.
    
    Tracks whether Inspect mode is producing information gain (uncertainty reduction).
    If Inspect mode isn't reducing uncertainty after N steps, diagnostic value decays
    and closure pressure increases.
    
    Rule: information gain decays unless it changes future action readiness.
    """
    def __init__(self, decay_window=5, decay_rate=0.2):
        self.decay_window = decay_window
        self.decay_rate = decay_rate
        self.reset()
    
    def reset(self):
        self.uncertainty_history = []
        self.inspect_steps = 0
        self.diagnostic_decay = 0.0
    
    def update(self, uncertainty, mode):
        """
        Track uncertainty over time. If in Inspect mode (1), check if
        uncertainty is actually decreasing.
        """
        self.uncertainty_history.append(uncertainty)
        if len(self.uncertainty_history) > self.decay_window:
            self.uncertainty_history.pop(0)
        
        if mode == 1:  # Inspect mode
            self.inspect_steps += 1
            
            # Check if uncertainty is decreasing
            if len(self.uncertainty_history) >= 3:
                recent = self.uncertainty_history[-3:]
                if recent[-1] >= recent[0]:
                    # Uncertainty not decreasing: increase diagnostic decay
                    self.diagnostic_decay = min(1.0, self.diagnostic_decay + self.decay_rate)
                else:
                    # Uncertainty decreasing: reduce decay
                    self.diagnostic_decay = max(0.0, self.diagnostic_decay - self.decay_rate * 0.5)
        else:
            # Not in Inspect mode: slowly reset decay
            self.inspect_steps = 0
            self.diagnostic_decay = max(0.0, self.diagnostic_decay - self.decay_rate * 0.3)
    
    def get_decay(self):
        """Returns current diagnostic decay value [0, 1]."""
        return self.diagnostic_decay
    
    def get_closure_boost(self):
        """
        Returns additional closure pressure from stalled diagnostics.
        When Inspect mode isn't producing value, closure pressure increases.
        """
        return self.diagnostic_decay * 0.3
